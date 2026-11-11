"""Crash-safe document updates (mocked embeddings, temp folders, no API calls).

Scenario mirrors the real knowledge base: 24 documents, one of them is
modified; the other 23 must stay exactly as they were.
"""

import json

import pytest
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.index_manager import IndexManager
from tests.conftest import CountingEmbedding, FailingEmbedding

OLD = "Python: האופרטור // הוא חלוקת שלמים. 7 // 2 == 3."
NEW = "Python: האופרטור // הוא Floor Division. עם float התוצאה float: 7.0 // 2 == 3.0."
TARGET = "Python/basics.txt"


class SimulatedCrash(BaseException):
    """Like the process being killed: not caught by `except Exception`."""


def make_kb(kb):
    (kb / "Python").mkdir()
    (kb / TARGET).write_text(OLD, encoding="utf-8")
    for i in range(23):
        folder = kb / f"Topic{i % 8}"
        folder.mkdir(exist_ok=True)
        (folder / f"doc{i}.txt").write_text(f"מסמך מספר {i} על נושא {i % 8}. תוכן ייחודי {i}.",
                                            encoding="utf-8")


def manager(storage, embed=None):
    m = IndexManager(storage_dir=storage, embed_model=embed or CountingEmbedding(embed_dim=8))
    m.load_or_create()
    return m


def snapshot(m):
    """{chunk_id: (text, file_name)} straight from ChromaDB."""
    data = m.collection.get(include=["documents", "metadatas"])
    return {i: (doc, json.loads(md["_node_content"])["metadata"]["file_name"])
            for i, doc, md in zip(data["ids"], data["documents"], data["metadatas"])}


def texts_of(m, file_name):
    return [text for text, name in snapshot(m).values() if name == file_name]


def assert_consistent(m):
    assert m.num_vectors() == sum(e["num_chunks"] for e in m.manifest.values())
    texts = [(t, n) for t, n in snapshot(m).values()]
    assert len(texts) == len(set(texts)), "duplicate chunks"


def others(m):
    return {i: v for i, v in snapshot(m).items() if v[1] != "basics.txt"}


@pytest.fixture
def indexed(dirs):
    kb, _, storage = dirs
    make_kb(kb)
    m = manager(storage)
    m.sync_folder(kb, "knowledge_base")
    assert len(m.manifest) == 24
    before_others = others(m)
    manifest_before = (storage / "manifest.json").read_text(encoding="utf-8")
    (kb / TARGET).write_text(NEW, encoding="utf-8")          # the file is modified
    return kb, storage, before_others, manifest_before


# ------------------------------------------------------------------ success
def test_successful_update_replaces_version_without_duplicates(indexed):
    kb, storage, before_others, _ = indexed
    embed = CountingEmbedding(embed_dim=8)
    m = manager(storage, embed)
    results = m.sync_folder(kb, "knowledge_base")

    assert [(r.file_name, r.status) for r in results] == [("basics.txt", "added")]
    assert "עודכן" in results[0].message
    assert texts_of(m, "basics.txt") == [NEW]                 # old chunks gone
    assert embed.call_count == 1                              # only the modified file
    assert others(m) == before_others                         # 23 others untouched
    assert len(m.manifest) == 24
    assert_consistent(m)
    reloaded = manager(storage)
    assert reloaded.recovery_messages == []
    assert texts_of(reloaded, "basics.txt") == [NEW]


# ------------------------------------------------------- embedding failure
def test_embedding_failure_keeps_previous_version(indexed):
    kb, storage, before_others, manifest_before = indexed
    m = manager(storage, FailingEmbedding(embed_dim=8))
    results = m.sync_folder(kb, "knowledge_base")

    assert results[0].status == "error"
    assert "הגרסה הקודמת של המסמך נשמרה" in results[0].message
    assert texts_of(m, "basics.txt") == [OLD]
    assert others(m) == before_others
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert_consistent(m)

    # When the connection is back, the next sync performs the update.
    fixed = manager(storage)
    fixed.sync_folder(kb, "knowledge_base")
    assert texts_of(fixed, "basics.txt") == [NEW]
    assert_consistent(fixed)


# ------------------------------------------------------- ChromaDB failure
def test_chromadb_failure_mid_insert_rolls_back(indexed, monkeypatch):
    kb, storage, before_others, manifest_before = indexed
    m = manager(storage)
    original_add = ChromaVectorStore.add

    def add_then_fail(self, nodes, **kwargs):
        original_add(self, nodes[:1], **kwargs)              # part of the batch is written
        raise RuntimeError("simulated ChromaDB write failure")

    monkeypatch.setattr(ChromaVectorStore, "add", add_then_fail)
    results = m.sync_folder(kb, "knowledge_base")
    monkeypatch.setattr(ChromaVectorStore, "add", original_add)

    assert results[0].status == "error" and "ChromaDB" in results[0].message
    assert texts_of(m, "basics.txt") == [OLD]                 # partial new chunk rolled back
    assert others(m) == before_others
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert_consistent(m)


# ------------------------------------------------------- manifest failure
def test_manifest_write_failure_rolls_back(indexed, monkeypatch):
    kb, storage, before_others, manifest_before = indexed
    m = manager(storage)

    def fail(manifest):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(m, "_write_manifest", fail)
    results = m.sync_folder(kb, "knowledge_base")

    assert results[0].status == "error" and "manifest" in results[0].message
    assert "הגרסה הקודמת של המסמך נשמרה" in results[0].message
    assert texts_of(m, "basics.txt") == [OLD]
    assert others(m) == before_others
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert_consistent(m)
    assert manager(storage).recovery_messages == []          # nothing left to repair


# --------------------------------------------- interruption during the swap
def test_crash_after_new_chunks_before_commit_recovers_old_version(indexed, monkeypatch):
    kb, storage, before_others, manifest_before = indexed
    m = manager(storage)

    def crash(manifest):
        raise SimulatedCrash()

    monkeypatch.setattr(m, "_write_manifest", crash)
    with pytest.raises(SimulatedCrash):
        m.sync_folder(kb, "knowledge_base")
    # State on disk right after the "crash": new chunks are orphans in ChromaDB.
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before

    restarted = manager(storage)                             # next start-up
    assert any("יתומים" in msg for msg in restarted.recovery_messages)
    assert texts_of(restarted, "basics.txt") == [OLD]
    assert others(restarted) == before_others
    assert_consistent(restarted)

    restarted.sync_folder(kb, "knowledge_base")              # update completes
    assert texts_of(restarted, "basics.txt") == [NEW]
    assert_consistent(restarted)


def test_crash_after_commit_before_old_deleted_recovers_new_version(indexed, monkeypatch):
    kb, storage, before_others, _ = indexed
    m = manager(storage)
    old_refs = set(next(e for e in m.manifest.values()
                        if e["file_name"] == "basics.txt")["ref_doc_ids"])
    original = IndexManager._delete_ref_docs

    def crash_on_old(self, ref_doc_ids):
        if set(ref_doc_ids) & old_refs:
            raise SimulatedCrash()
        return original(self, ref_doc_ids)

    monkeypatch.setattr(IndexManager, "_delete_ref_docs", crash_on_old)
    with pytest.raises(SimulatedCrash):
        m.sync_folder(kb, "knowledge_base")
    monkeypatch.setattr(IndexManager, "_delete_ref_docs", original)

    restarted = manager(storage)
    assert any("יתומים" in msg for msg in restarted.recovery_messages)
    assert texts_of(restarted, "basics.txt") == [NEW]        # committed version wins
    assert others(restarted) == before_others
    assert_consistent(restarted)


def test_rollback_failure_is_cleaned_on_next_start(indexed, monkeypatch):
    """Embedding fails AND the rollback delete fails: recovery still cleans up."""
    kb, storage, before_others, _ = indexed
    m = manager(storage)
    original_add = ChromaVectorStore.add

    def add_then_fail(self, nodes, **kwargs):
        original_add(self, nodes, **kwargs)
        raise RuntimeError("fail after write")

    monkeypatch.setattr(ChromaVectorStore, "add", add_then_fail)
    monkeypatch.setattr(IndexManager, "_try_delete", lambda self, refs: None)
    assert m.sync_folder(kb, "knowledge_base")[0].status == "error"
    monkeypatch.undo()

    restarted = manager(storage)
    assert any("יתומים" in msg for msg in restarted.recovery_messages)
    assert texts_of(restarted, "basics.txt") == [OLD]
    assert others(restarted) == before_others
    assert_consistent(restarted)


# ------------------------------------------------------------ other repairs
def test_entry_with_missing_chunks_is_reindexed(dirs):
    kb, _, storage = dirs
    make_kb(kb)
    m = manager(storage)
    m.sync_folder(kb, "knowledge_base")
    entry = next(e for e in m.manifest.values() if e["file_name"] == "basics.txt")
    m.collection.delete(where={"document_id": {"$in": entry["ref_doc_ids"]}})   # chunks lost

    restarted = manager(storage)
    assert any("לא היה שלם" in msg for msg in restarted.recovery_messages)
    assert len(restarted.manifest) == 23
    embed = CountingEmbedding(embed_dim=8)
    again = manager(storage, embed)
    again.sync_folder(kb, "knowledge_base")
    assert embed.call_count == 1 and len(again.manifest) == 24
    assert_consistent(again)


def test_leftover_temp_manifest_is_removed(dirs):
    kb, _, storage = dirs
    make_kb(kb)
    manager(storage).sync_folder(kb, "knowledge_base")
    (storage / "manifest.tmp").write_text("{ half written", encoding="utf-8")
    restarted = manager(storage)
    assert not (storage / "manifest.tmp").exists()
    assert len(restarted.manifest) == 24
    assert_consistent(restarted)


def test_consistent_index_is_not_touched_by_recovery(dirs):
    kb, _, storage = dirs
    make_kb(kb)
    m = manager(storage)
    m.sync_folder(kb, "knowledge_base")
    before = snapshot(m)
    restarted = manager(storage)
    assert restarted.recovery_messages == []
    assert snapshot(restarted) == before


def test_deleted_file_crash_between_commit_and_delete(dirs, monkeypatch):
    kb, _, storage = dirs
    make_kb(kb)
    m = manager(storage)
    m.sync_folder(kb, "knowledge_base")
    (kb / "Topic0" / "doc0.txt").unlink()
    monkeypatch.setattr(IndexManager, "_delete_ref_docs",
                        lambda self, refs: (_ for _ in ()).throw(SimulatedCrash()))
    with pytest.raises(SimulatedCrash):
        m.sync_folder(kb, "knowledge_base")
    monkeypatch.undo()
    restarted = manager(storage)
    assert len(restarted.manifest) == 23
    assert "doc0.txt" not in {n for _, n in snapshot(restarted).values()}
    assert_consistent(restarted)


def test_stop_on_error_stops_after_first_failure(dirs):
    """Used by the paid test script's --strict mode: no further embedding attempts."""
    kb, _, storage = dirs
    make_kb(kb)

    class CountingFailure(FailingEmbedding):
        def _get_text_embeddings(self, texts):
            attempts.append(1)
            raise RuntimeError("no network")

    attempts = []
    m = manager(storage, CountingFailure(embed_dim=8))
    results = m.sync_folder(kb, "knowledge_base", stop_on_error=True)
    assert len(results) == 1 and results[0].status == "error"
    assert len(attempts) == 1
    assert m.num_vectors() == 0 and m.manifest == {}
