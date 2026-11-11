"""Persistent vector index - LlamaIndex + ChromaDB (local, file-based).

Key ideas:
* Vectors, chunk text and metadata are stored in a local ChromaDB database
  (chromadb.PersistentClient) under STORAGE_DIR/chroma - no server needed.
  On restart the index is reopened with VectorStoreIndex.from_vector_store,
  so no embeddings are re-created.
* manifest.json maps every indexed file (by SHA-256 hash) to the ids of the
  LlamaIndex documents created from it. A file whose hash is already in the
  manifest is never embedded again.
* New files are added with insert_nodes() - only the new chunks are embedded;
  the rest of the index is untouched.

Crash safety (ChromaDB and manifest.json cannot be updated atomically together):
* manifest.json is the source of truth. It is written atomically (temp file +
  replace) and only AFTER the new chunks are safely stored in ChromaDB.
* A modified file is replaced in this order: (1) extract, chunk, embed and
  store the new version (old version untouched), (2) commit the manifest,
  (3) delete the old chunks. A failure in (1) or (2) keeps the old version.
* On every load, _recover() (local, no API calls) deletes "orphan" chunks that
  the manifest does not know (leftovers of an interrupted update) and drops
  manifest entries whose chunks are missing, so they are re-indexed next sync.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.utils import metadata_dict_to_node

import config
from rag.errors import friendly_api_error
from rag.document_loader import (
    DocumentLoadError,
    compute_bytes_hash,
    compute_file_hash,
    extract_documents,
    iter_supported_files,
    relative_path_for,
    split_documents,
    topic_for,
)


def create_openai_embed_model():
    """OpenAI embedding model. Constructing it makes no API call."""
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(model=config.EMBED_MODEL, max_retries=2, timeout=60)


def open_chroma_collection(chroma_dir: Path):
    """Open (or create) the local ChromaDB collection. No network access.

    Telemetry is disabled; distance is cosine (semantic similarity).
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(chroma_dir), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        config.CHROMA_COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )


@dataclass
class AddResult:
    status: str          # "added" | "duplicate" | "error"
    file_name: str
    message: str
    num_chunks: int = 0


class IndexManager:
    def __init__(self, storage_dir: Path | None = None, embed_model=None):
        self.storage_dir = Path(storage_dir or config.STORAGE_DIR)
        self.chroma_dir = self.storage_dir / "chroma"
        self.manifest_path = self.storage_dir / config.MANIFEST_FILENAME
        self.embed_model = embed_model or create_openai_embed_model()
        self.index: VectorStoreIndex | None = None
        self.collection = None
        self.manifest: dict[str, dict] = {}
        self.loaded_from_storage = False
        self.load_error: str | None = None
        # Results of the last sync, keyed by relative path (for the UI).
        self.last_sync_results: dict[str, AddResult] = {}
        # Bumped on every change, so the cached BM25 index is rebuilt when needed.
        self._version = 0
        self._keyword_cache = None
        # Messages about what _recover() fixed on the last load (for the UI).
        self.recovery_messages: list[str] = []

    # ------------------------------------------------------------------ load
    def load_or_create(self) -> VectorStoreIndex | None:
        """Open the ChromaDB index (creating it if needed). No API calls.

        If the stored data is corrupt, load_error is set and the files are
        left untouched (nothing is overwritten or deleted).
        """
        from llama_index.vector_stores.chroma import ChromaVectorStore

        self.load_error = None
        self._version += 1
        existed = (self.chroma_dir / "chroma.sqlite3").exists()
        try:
            self.manifest = self._read_manifest()
            self.collection = open_chroma_collection(self.chroma_dir)
            self.index = VectorStoreIndex.from_vector_store(
                ChromaVectorStore(chroma_collection=self.collection),
                embed_model=self.embed_model,
            )
            self.recovery_messages = self._recover()
        except Exception as error:
            self.index = None
            self.collection = None
            self.manifest = {}
            self.loaded_from_storage = False
            self.load_error = f"טעינת האינדקס מהתיקייה storage נכשלה: {error}"
            return None
        self.loaded_from_storage = existed
        return self.index

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict) -> None:
        """Atomic write: a reader sees either the old or the new file, never half."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _commit(self, manifest: dict) -> None:
        """Write the manifest to disk first; switch the in-memory copy only on success."""
        self._write_manifest(manifest)
        self.manifest = manifest
        self._version += 1

    def _persist(self) -> None:
        """ChromaDB writes to disk by itself; only the manifest is saved here."""
        self._commit(dict(self.manifest))

    def _recover(self) -> list[str]:
        """Repair inconsistencies left by an interrupted operation. No API calls.

        * chunks whose document is not in the manifest -> deleted (orphans)
        * manifest entries whose chunks are missing/incomplete -> dropped, so
          the file is indexed again on the next sync
        * a leftover manifest temp file (interrupted write) -> deleted
        """
        messages = []
        tmp = self.manifest_path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
            messages.append("נמצא קובץ זמני מכתיבה שנקטעה, והוא נמחק.")

        data = self.collection.get(include=["metadatas"])
        chunks_by_ref: dict[str, list[str]] = {}
        for chunk_id, md in zip(data["ids"], data["metadatas"]):
            ref = (md or {}).get("document_id") or (md or {}).get("ref_doc_id") or ""
            chunks_by_ref.setdefault(ref, []).append(chunk_id)

        known_refs = {ref for e in self.manifest.values() for ref in e.get("ref_doc_ids", [])}
        orphan_ids = [cid for ref, ids in chunks_by_ref.items()
                      if ref not in known_refs for cid in ids]
        if orphan_ids:
            self.collection.delete(ids=orphan_ids)
            messages.append("נוקו נתונים יתומים שנשארו מעדכון שנקטע.")

        manifest = dict(self.manifest)
        for file_hash, entry in list(manifest.items()):
            stored = sum(len(chunks_by_ref.get(ref, [])) for ref in entry.get("ref_doc_ids", []))
            if stored != entry.get("num_chunks", -1):
                ids = [cid for ref in entry.get("ref_doc_ids", [])
                       for cid in chunks_by_ref.get(ref, [])]
                if ids:
                    self.collection.delete(ids=ids)
                del manifest[file_hash]
                messages.append(
                    f"המסמך '{entry.get('file_name')}' לא היה שלם במאגר הידע, "
                    "והוא יתווסף מחדש בסנכרון הבא.")
        if manifest != self.manifest:
            self._commit(manifest)
        return messages

    def _require_index(self) -> VectorStoreIndex:
        if self.index is None and self.load_error is None:
            self.load_or_create()
        if self.index is None:
            # Never build a new index on top of a corrupt one.
            raise RuntimeError(self.load_error or "האינדקס אינו זמין.")
        return self.index

    # ------------------------------------------------------------- queries
    def find_by_hash(self, file_hash: str) -> dict | None:
        return self.manifest.get(file_hash)

    def _find_by_path(self, path: Path) -> str | None:
        key = str(path.resolve())
        for file_hash, entry in self.manifest.items():
            if entry.get("path") == key:
                return file_hash
        return None

    def stats(self) -> dict:
        self._require_index()
        return {
            "num_documents": len(self.manifest),
            "num_chunks": self.collection.count(),
            "loaded_from_storage": self.loaded_from_storage,
        }

    def num_vectors(self) -> int:
        """Number of stored embeddings (one per chunk) - for consistency checks."""
        self._require_index()
        return self.collection.count()

    def keyword_index(self):
        """BM25 index over all stored chunks (built locally, cached until the
        index changes). No API calls."""
        from rag.hybrid import KeywordIndex

        if self._keyword_cache is None or self._keyword_cache[0] != self._version:
            self._keyword_cache = (self._version, KeywordIndex(self.all_nodes()))
        return self._keyword_cache[1]

    def all_nodes(self) -> list:
        """All stored chunks as LlamaIndex nodes (text + metadata). No API calls."""
        self._require_index()
        data = self.collection.get(include=["documents", "metadatas"])
        return [metadata_dict_to_node(md, text=text)
                for text, md in zip(data["documents"], data["metadatas"])]

    def is_indexed(self, path: Path) -> bool:
        try:
            return compute_file_hash(path) in self.manifest
        except OSError:
            return False

    def list_documents(self) -> list[dict]:
        return sorted(
            ({"hash": h, **entry} for h, entry in self.manifest.items()),
            key=lambda e: (e.get("source", ""), e.get("file_name", "")),
        )

    # ---------------------------------------------------------------- write
    def add_file(self, path: Path, source: str = "knowledge_base",
                 topic: str | None = None,
                 relative_path: str | None = None,
                 replaces: str | None = None) -> AddResult:
        """Index a single file, unless identical content is already indexed.

        replaces: hash of the previous version of the same file. The previous
        version stays in the index until the new one is fully stored and the
        manifest is committed; on any failure it is kept.
        """
        index = self._require_index()
        path = Path(path)
        kept = " הגרסה הקודמת של המסמך נשמרה באינדקס." if replaces else ""

        try:
            file_hash = compute_file_hash(path)
        except OSError as error:
            return AddResult("error", path.name, f"לא ניתן לקרוא את הקובץ: {error}{kept}")

        existing = self.find_by_hash(file_hash)
        if existing:
            if replaces and replaces in self.manifest:
                # The file now has the same content as another indexed file:
                # its old version is obsolete; the content is indexed already.
                self._drop_entry(replaces)
            return AddResult(
                "duplicate",
                path.name,
                f"המסמך כבר קיים במאגר (בשם '{existing['file_name']}').",
                existing.get("num_chunks", 0),
            )

        try:
            documents = extract_documents(path, source=source, file_hash=file_hash,
                                          topic=topic, relative_path=relative_path)
        except DocumentLoadError as error:
            return AddResult("error", path.name, f"{error}{kept}")

        nodes = split_documents(documents)
        new_refs = [d.id_ for d in documents]
        # Step 1 - store the new version. The old version is not touched.
        try:
            # Leftovers of an earlier interrupted attempt with the same content
            # (same deterministic document ids) are removed first - no duplicates.
            self._delete_ref_docs(new_refs)
            # Only the new nodes are embedded here (API call when using OpenAI).
            index.insert_nodes(nodes)
        except Exception as error:
            self._try_delete(new_refs)       # roll back a partial insert
            return AddResult(
                "error", path.name,
                f"שגיאה ביצירת Embeddings או בשמירה ל־ChromaDB: "
                f"{friendly_api_error(error)}{kept}",
            )

        # Step 2 - commit: the manifest now points to the new version.
        manifest = dict(self.manifest)
        manifest[file_hash] = {
            "file_name": path.name,
            "path": str(path.resolve()),
            "source": source,
            "topic": topic,
            "relative_path": relative_path or path.name,
            "ref_doc_ids": new_refs,
            "num_chunks": len(nodes),
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
        old_entry = manifest.pop(replaces, None) if replaces else None
        try:
            self._commit(manifest)
        except Exception as error:
            self._try_delete(new_refs)       # roll back; old version stays valid
            return AddResult(
                "error", path.name,
                f"שמירת רשימת המסמכים (manifest) נכשלה: {error}.{kept}",
            )

        # Step 3 - remove the old version. If this fails, the chunks are
        # orphans and _recover() deletes them on the next load.
        if old_entry:
            self._try_delete(old_entry.get("ref_doc_ids", []))
            message = "המסמך עודכן במאגר הידע."
        else:
            message = "המסמך נוסף לספרייה ולמאגר הידע."
        return AddResult("added", path.name, message, len(nodes))

    def _delete_ref_docs(self, ref_doc_ids: list[str]) -> None:
        """Delete all chunks of the given LlamaIndex documents (no API call)."""
        for ref_doc_id in ref_doc_ids:
            self.index.vector_store.delete(ref_doc_id)

    def _try_delete(self, ref_doc_ids: list[str]) -> None:
        """Best-effort delete; anything left behind is cleaned by _recover()."""
        try:
            self._delete_ref_docs(ref_doc_ids)
        except Exception:
            pass

    def _drop_entry(self, file_hash: str) -> None:
        """Commit the removal in the manifest first, then delete the chunks."""
        entry = self.manifest.get(file_hash)
        if not entry:
            return
        manifest = dict(self.manifest)
        del manifest[file_hash]
        self._commit(manifest)
        self._try_delete(entry.get("ref_doc_ids", []))

    def remove_document(self, file_hash: str) -> None:
        """Remove a document's chunks from the index (no API call)."""
        self._require_index()
        self._drop_entry(file_hash)

    def entry_for_path(self, path: Path) -> dict | None:
        """Manifest entry for this path (possibly an older version of the file)."""
        file_hash = self._find_by_path(Path(path))
        return self.manifest.get(file_hash) if file_hash else None

    def _relink(self, file_hash: str, new_path: Path, topic: str | None,
                relative_path: str) -> None:
        """A file was renamed or moved to another topic folder: update its
        name and topic everywhere without re-embedding.

        ChromaDB keeps the metadata twice: as flat fields (used for filtering)
        and inside "_node_content" (used to rebuild the node). Both are
        updated; the embeddings are not touched.
        """
        entry = self.manifest[file_hash]
        ref_ids = list(entry.get("ref_doc_ids", []))
        if ref_ids:
            found = self.collection.get(where={"document_id": {"$in": ref_ids}},
                                        include=["metadatas"])
            new_metadatas = []
            for md in found["metadatas"]:
                md = dict(md)
                content = json.loads(md["_node_content"])
                for target in (md, content["metadata"]):
                    target["file_name"] = new_path.name
                    target["relative_path"] = relative_path
                    if topic:
                        target["topic"] = topic
                    else:
                        target.pop("topic", None)
                if not topic:
                    md["topic"] = None      # None removes the field in ChromaDB
                md["_node_content"] = json.dumps(content)
                new_metadatas.append(md)
            if found["ids"]:
                self.collection.update(ids=found["ids"], metadatas=new_metadatas)
        entry["file_name"] = new_path.name
        entry["path"] = str(new_path.resolve())
        entry["topic"] = topic
        entry["relative_path"] = relative_path
        self._persist()

    def sync_folder(self, folder: Path, source: str,
                    stop_on_error: bool = False) -> list[AddResult]:
        """Bring the index in line with a folder.

        * unchanged files (same hash) -> skipped, no embeddings
        * new files                   -> embedded and added
        * modified files (same path, new hash) -> new version stored first;
          the old version is removed only after that succeeded
        * files deleted from the folder -> their chunks removed
        * renamed / moved between sub-folders -> name and topic updated only

        Sub-folders are scanned recursively; the first-level sub-folder name
        is stored as the document's topic.
        """
        self._require_index()
        folder = Path(folder)
        results = []
        hashes = {str(p.resolve()): compute_file_hash(p)
                  for p in iter_supported_files(folder)}

        # 1. Entries whose file is gone from this folder: if the same content
        #    still exists under another name (renamed file) just update the
        #    name - no API call; otherwise remove the chunks.
        root = folder.resolve()
        for file_hash, entry in list(self.manifest.items()):
            entry_path = entry.get("path", "")
            if not Path(entry_path).is_relative_to(root) or entry_path in hashes:
                continue
            renamed_to = next((p for p, h in hashes.items() if h == file_hash), None)
            if renamed_to:
                self._relink(file_hash, Path(renamed_to), topic_for(renamed_to, root),
                             relative_path_for(renamed_to, root))
            else:
                self.remove_document(file_hash)

        # 2. New or modified files.
        for path_key, file_hash in hashes.items():
            if file_hash in self.manifest:
                continue  # unchanged (or a duplicate of an indexed file)
            path = Path(path_key)
            old_hash = self._find_by_path(path)     # set when the file was modified
            result = self.add_file(path, source=source, topic=topic_for(path, root),
                                   relative_path=relative_path_for(path, root),
                                   replaces=old_hash)
            results.append(result)
            self.last_sync_results[path.relative_to(root).as_posix()] = result
            if stop_on_error and result.status == "error":
                break       # e.g. no network: don't try (and pay for) the next files

        return results

    def add_uploaded_file(self, file_name: str, data: bytes,
                          uploads_dir: Path | None = None) -> AddResult:
        """Save an uploaded file to uploads/ and index it.

        Duplicates (by content hash) are detected *before* anything is
        written to disk, so a duplicate upload leaves no trace.
        """
        uploads_dir = Path(uploads_dir or config.UPLOADS_DIR)
        safe_name = Path(file_name).name  # strip any directory part
        suffix = Path(safe_name).suffix.lower()

        if suffix not in config.SUPPORTED_EXTENSIONS:
            return AddResult(
                "error", safe_name,
                f"סוג הקובץ '{suffix or 'ללא סיומת'}' אינו נתמך. "
                "ניתן להעלות קובצי PDF, DOCX או TXT.",
            )
        if not data:
            return AddResult("error", safe_name, f"הקובץ '{safe_name}' ריק.")

        self._require_index()
        existing = self.find_by_hash(compute_bytes_hash(data))
        if existing:
            return AddResult(
                "duplicate", safe_name,
                f"המסמך כבר קיים במאגר (בשם '{existing['file_name']}').",
                existing.get("num_chunks", 0),
            )

        uploads_dir.mkdir(parents=True, exist_ok=True)
        destination = uploads_dir / safe_name
        counter = 1
        while destination.exists():  # different content, same name
            destination = uploads_dir / f"{Path(safe_name).stem}_{counter}{suffix}"
            counter += 1
        destination.write_bytes(data)

        result = self.add_file(destination, source="uploads")
        if result.status != "added":
            destination.unlink(missing_ok=True)  # don't keep unusable files
        return result
