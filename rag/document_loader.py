"""Loading, text extraction and chunking of documents (PDF / DOCX / TXT)."""

import hashlib
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

import config

# Metadata that is useful for tracking but should not be embedded or sent to
# the LLM (saves tokens and keeps embeddings focused on content).
_INTERNAL_METADATA_KEYS = ["doc_id", "source", "relative_path"]


class DocumentLoadError(Exception):
    """Raised when a file cannot be turned into indexable text.

    The message is in Hebrew and is meant to be shown to the user as-is.
    """


def compute_file_hash(path: Path) -> str:
    """SHA-256 of the file contents, used to detect duplicate documents."""
    return compute_bytes_hash(Path(path).read_bytes())


def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pdf(path: Path) -> list[tuple[str, int | None]]:
    try:
        reader = PdfReader(str(path))
        pages = [
            (page.extract_text() or "", number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as error:
        raise DocumentLoadError(
            f"הקובץ '{path.name}' פגום או שאינו קובץ PDF תקין ({error})."
        ) from error

    if not any(text.strip() for text, _ in pages):
        raise DocumentLoadError(
            f"לא ניתן לחלץ טקסט מהקובץ '{path.name}'. "
            "ייתכן שזהו PDF סרוק (תמונה בלבד). "
            "יש להעלות גרסה של המסמך שמכילה טקסט."
        )
    return pages


def _read_docx(path: Path) -> list[tuple[str, int | None]]:
    try:
        import docx  # python-docx

        document = docx.Document(str(path))
    except Exception as error:
        raise DocumentLoadError(
            f"הקובץ '{path.name}' פגום או שאינו קובץ DOCX תקין ({error})."
        ) from error

    parts = [p.text for p in document.paragraphs if p.text.strip()]
    # Text inside tables is not part of document.paragraphs.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return [("\n".join(parts), None)]


def _read_txt(path: Path) -> list[tuple[str, int | None]]:
    data = path.read_bytes()
    # UTF-8 first; cp1255 is the common legacy Hebrew encoding on Windows.
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            return [(data.decode(encoding), None)]
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(
        f"לא ניתן לפענח את הקידוד של הקובץ '{path.name}'. יש לשמור אותו כ־UTF-8."
    )


_READERS = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}


def extract_documents(path: Path, source: str = "knowledge_base",
                      file_hash: str | None = None,
                      topic: str | None = None,
                      relative_path: str | None = None) -> list[Document]:
    """Extract text from one file into LlamaIndex Documents.

    PDF files produce one Document per page (keeps the page number for
    citations); DOCX and TXT produce a single Document. `topic` (the
    sub-folder name, e.g. "Arduino") is stored in the metadata.

    Raises DocumentLoadError for unsupported, empty, corrupt or text-less files.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"סוג הקובץ '{suffix or 'ללא סיומת'}' אינו נתמך. "
            "ניתן להעלות קובצי PDF, DOCX או TXT."
        )
    if not path.is_file() or path.stat().st_size == 0:
        raise DocumentLoadError(f"הקובץ '{path.name}' ריק.")

    file_hash = file_hash or compute_file_hash(path)
    documents = []

    for text, page in _READERS[suffix](path):
        if not text.strip():
            continue
        metadata = {
            "file_name": path.name,
            "doc_id": file_hash,
            "source": source,
            "relative_path": relative_path or path.name,
        }
        if topic:
            metadata["topic"] = topic
        if page is not None:
            metadata["page"] = page

        documents.append(
            Document(
                text=text,
                id_=f"{file_hash}_{page or 0}",
                metadata=metadata,
                excluded_embed_metadata_keys=list(_INTERNAL_METADATA_KEYS),
                excluded_llm_metadata_keys=list(_INTERNAL_METADATA_KEYS),
            )
        )

    if not documents:
        raise DocumentLoadError(f"לא נמצא טקסט בקובץ '{path.name}'.")
    return documents


def get_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def split_documents(documents: list[Document]):
    """Split Documents into Nodes (chunks). Metadata is inherited by each node."""
    if not documents:
        return []
    return get_splitter().get_nodes_from_documents(documents)


def _is_hidden_or_temp(path: Path, root: Path) -> bool:
    """Skip hidden folders/files (.git, .ipynb_checkpoints) and Office lock files (~$x.docx)."""
    parts = path.relative_to(root).parts
    return any(p.startswith(".") for p in parts) or path.name.startswith("~$")


def iter_supported_files(folder: Path) -> list[Path]:
    """All supported files in a folder, including sub-folders (recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not _is_hidden_or_temp(p, folder)
    )


def relative_path_for(path: Path, root: Path) -> str:
    """Path relative to the documents root, with forward slashes (e.g. "STM32/gpio.pdf")."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def topic_for(path: Path, root: Path) -> str | None:
    """Topic = name of the first-level sub-folder under root, or None.

    knowledge_base/Arduino/pwm.pdf          -> "Arduino"
    knowledge_base/STM32/HAL/gpio.pdf       -> "STM32"
    knowledge_base/test_study.txt           -> None
    """
    parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    return parts[0] if len(parts) > 1 else None


def load_and_split_documents(folder: Path | None = None):
    """Load every supported file in a folder and split it into chunks.

    Files that cannot be read are skipped. Returns (documents, nodes).
    Kept for the existing "read and split" preview in app.py; it does not
    create embeddings.
    """
    folder = Path(folder or config.KNOWLEDGE_BASE_DIR)
    folder.mkdir(exist_ok=True)

    documents = []
    for path in iter_supported_files(folder):
        try:
            documents.extend(extract_documents(
                path, topic=topic_for(path, folder),
                relative_path=relative_path_for(path, folder)))
        except DocumentLoadError:
            continue

    return documents, split_documents(documents)
