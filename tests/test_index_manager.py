"""Local tests for the persistent index (mock embeddings, no API calls)."""

import json

from rag.index_manager import IndexManager
from tests.conftest import CountingEmbedding, FailingEmbedding, make_blank_pdf

TEXT_A = "ממוצע חשבוני: סכום כל הערכים חלקי מספר הערכים."
TEXT_B = "חציון: הערך האמצעי בקבוצת נתונים מסודרת."


def new_manager(storage, embed_model=None):
    manager = IndexManager(storage_dir=storage,
                           embed_model=embed_model or CountingEmbedding(embed_dim=8))
    manager.load_or_create()
    return manager


def test_empty_index_created_without_storage(dirs):
    _, _, storage = dirs
    manager = new_manager(storage)
    assert manager.loaded_from_storage is False
    assert manager.stats() == {"num_documents": 0, "num_chunks": 0,
                               "loaded_from_storage": False}


def test_add_file_persists_index_and_manifest(dirs, embed_model):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")

    manager = new_manager(storage, embed_model)
    result = manager.add_file(kb / "a.txt")

    assert result.status == "added"
    assert embed_model.call_count == result.num_chunks == 1
    assert (storage / "chroma" / "chroma.sqlite3").exists()
    assert manager.num_vectors() == 1
    manifest = json.loads((storage / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest.values())[0]["file_name"] == "a.txt"


def test_reload_from_storage_does_not_reembed(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "b.txt").write_text(TEXT_B, encoding="utf-8")
    first = new_manager(storage)
    first.sync_folder(kb, "knowledge_base")

    fresh_embed = CountingEmbedding(embed_dim=8)
    second = new_manager(storage, fresh_embed)
    results = second.sync_folder(kb, "knowledge_base")

    assert second.loaded_from_storage is True
    assert results == []                       # nothing new
    assert fresh_embed.call_count == 0         # no embeddings recreated
    assert second.stats()["num_documents"] == 2
    assert second.stats()["num_chunks"] == 2


def test_duplicate_content_detected_by_hash(dirs, embed_model):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "copy_of_a.txt").write_text(TEXT_A, encoding="utf-8")
    manager = new_manager(storage, embed_model)

    assert manager.add_file(kb / "a.txt").status == "added"
    result = manager.add_file(kb / "copy_of_a.txt")
    assert result.status == "duplicate"
    assert "a.txt" in result.message
    assert embed_model.call_count == 1
    assert manager.stats()["num_chunks"] == 1


def test_adding_one_file_only_embeds_that_file(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    new_manager(storage).sync_folder(kb, "knowledge_base")

    (kb / "b.txt").write_text(TEXT_B, encoding="utf-8")
    embed = CountingEmbedding(embed_dim=8)
    manager = new_manager(storage, embed)
    results = manager.sync_folder(kb, "knowledge_base")

    assert [r.file_name for r in results] == ["b.txt"]
    assert embed.call_count == 1
    assert manager.stats()["num_chunks"] == 2


def test_modified_file_replaces_old_chunks(dirs):
    kb, _, storage = dirs
    path = kb / "a.txt"
    path.write_text(TEXT_A, encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")

    path.write_text(TEXT_B, encoding="utf-8")
    manager.sync_folder(kb, "knowledge_base")

    texts = [n.text for n in manager.all_nodes()]
    assert texts == [TEXT_B]
    assert manager.stats()["num_documents"] == 1


def test_deleted_file_removed_from_index(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "b.txt").write_text(TEXT_B, encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")

    (kb / "b.txt").unlink()
    manager.sync_folder(kb, "knowledge_base")
    assert manager.stats()["num_chunks"] == 1

    reloaded = new_manager(storage)
    assert reloaded.stats()["num_chunks"] == 1


def test_upload_saved_to_uploads_and_indexed(dirs, embed_model):
    _, uploads, storage = dirs
    manager = new_manager(storage, embed_model)

    result = manager.add_uploaded_file("notes.txt", TEXT_A.encode("utf-8"), uploads)
    assert result.status == "added"
    assert (uploads / "notes.txt").exists()
    entry = manager.list_documents()[0]
    assert entry["source"] == "uploads"

    node = manager.all_nodes()[0]
    assert node.metadata["file_name"] == "notes.txt"
    assert node.metadata["source"] == "uploads"


def test_duplicate_upload_not_saved(dirs, embed_model):
    _, uploads, storage = dirs
    manager = new_manager(storage, embed_model)
    manager.add_uploaded_file("notes.txt", TEXT_A.encode("utf-8"), uploads)

    result = manager.add_uploaded_file("renamed.txt", TEXT_A.encode("utf-8"), uploads)
    assert result.status == "duplicate"
    assert not (uploads / "renamed.txt").exists()
    assert embed_model.call_count == 1


def test_same_name_different_content_gets_new_name(dirs):
    _, uploads, storage = dirs
    manager = new_manager(storage)
    manager.add_uploaded_file("notes.txt", TEXT_A.encode("utf-8"), uploads)
    result = manager.add_uploaded_file("notes.txt", TEXT_B.encode("utf-8"), uploads)

    assert result.status == "added"
    assert (uploads / "notes.txt").read_text(encoding="utf-8") == TEXT_A
    assert (uploads / "notes_1.txt").read_text(encoding="utf-8") == TEXT_B


def test_invalid_uploads_rejected_and_not_kept(dirs, tmp_path):
    _, uploads, storage = dirs
    manager = new_manager(storage)

    assert manager.add_uploaded_file("x.exe", b"MZ", uploads).status == "error"
    assert manager.add_uploaded_file("empty.txt", b"", uploads).status == "error"
    assert manager.add_uploaded_file("bad.pdf", b"garbage", uploads).status == "error"

    scan = make_blank_pdf(tmp_path / "scan.pdf")
    result = manager.add_uploaded_file("scan.pdf", scan.read_bytes(), uploads)
    assert result.status == "error"
    assert "סרוק" in result.message

    assert not uploads.exists() or list(uploads.iterdir()) == []
    assert manager.stats()["num_documents"] == 0


def test_path_traversal_in_upload_name(dirs, tmp_path):
    _, uploads, storage = dirs
    manager = new_manager(storage)
    manager.add_uploaded_file("../../evil.txt", TEXT_A.encode("utf-8"), uploads)
    assert (uploads / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_embedding_api_error_is_handled(dirs):
    kb, uploads, storage = dirs
    manager = new_manager(storage, FailingEmbedding(embed_dim=8))

    result = manager.add_uploaded_file("a.txt", TEXT_A.encode("utf-8"), uploads)
    assert result.status == "error"
    assert "Embeddings" in result.message
    assert manager.stats() == {"num_documents": 0, "num_chunks": 0,
                               "loaded_from_storage": False}
    assert not (uploads / "a.txt").exists()


# --- Consistency: no stale chunks, failures don't corrupt the index ---------

def _assert_consistent(manager):
    """Stored chunks, vectors and manifest all describe the same chunks."""
    expected = sum(e["num_chunks"] for e in manager.manifest.values())
    assert len(manager.all_nodes()) == expected
    assert manager.num_vectors() == expected


def test_no_stale_chunks_after_update_including_reload(dirs):
    kb, _, storage = dirs
    path = kb / "a.txt"
    path.write_text(TEXT_A, encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")

    path.write_text(TEXT_B, encoding="utf-8")
    manager.sync_folder(kb, "knowledge_base")
    _assert_consistent(manager)

    reloaded = new_manager(storage)
    _assert_consistent(reloaded)
    texts = [n.text for n in reloaded.all_nodes()]
    assert texts == [TEXT_B]
    assert reloaded.num_vectors() == 1


def test_no_stale_chunks_after_delete_including_reload(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "b.txt").write_text(TEXT_B, encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")
    (kb / "a.txt").unlink()
    manager.sync_folder(kb, "knowledge_base")

    reloaded = new_manager(storage)
    _assert_consistent(reloaded)
    names = {n.metadata["file_name"] for n in reloaded.all_nodes()}
    assert names == {"b.txt"}


def test_multi_page_pdf_update_removes_all_old_pages(dirs, monkeypatch):
    """PDFs become one LlamaIndex document per page - all pages must be removed."""
    from rag import document_loader

    pages = {"v1": [("עמוד ראשון ממוצע", 1), ("עמוד שני חציון", 2)],
             "v2": [("גרסה חדשה שכיח", 1)]}
    monkeypatch.setitem(document_loader._READERS, ".pdf",
                        lambda p: pages[p.read_text(encoding="utf-8")])

    kb, _, storage = dirs
    pdf = kb / "book.pdf"
    pdf.write_text("v1", encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")
    assert manager.num_vectors() == 2
    assert len(next(iter(manager.manifest.values()))["ref_doc_ids"]) == 2

    pdf.write_text("v2", encoding="utf-8")
    manager.sync_folder(kb, "knowledge_base")
    _assert_consistent(manager)
    nodes = list(manager.all_nodes())
    assert [n.text for n in nodes] == ["גרסה חדשה שכיח"]
    assert nodes[0].metadata["page"] == 1


def test_failed_indexing_keeps_previous_index_intact(dirs, tmp_path):
    kb, uploads, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    good = new_manager(storage)
    good.sync_folder(kb, "knowledge_base")
    manifest_before = (storage / "manifest.json").read_text(encoding="utf-8")

    failing = IndexManager(storage_dir=storage, embed_model=FailingEmbedding(embed_dim=8))
    failing.load_or_create()
    result = failing.add_uploaded_file("b.txt", TEXT_B.encode("utf-8"), uploads)
    assert result.status == "error"
    _assert_consistent(failing)

    # Files on disk were not touched by the failed attempt.
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before
    reloaded = new_manager(storage)
    _assert_consistent(reloaded)
    assert reloaded.stats()["num_documents"] == 1


def test_one_bad_file_does_not_stop_sync(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "empty.txt").write_bytes(b"")
    make_blank_pdf(kb / "scan.pdf")
    (kb / "b.txt").write_text(TEXT_B, encoding="utf-8")

    manager = new_manager(storage)
    results = {r.file_name: r.status for r in manager.sync_folder(kb, "knowledge_base")}
    assert results == {"a.txt": "added", "b.txt": "added",
                       "empty.txt": "error", "scan.pdf": "error"}
    assert "סרוק" in manager.last_sync_results["scan.pdf"].message
    _assert_consistent(manager)


def test_corrupt_storage_is_reported_not_overwritten(dirs):
    kb, uploads, storage = dirs
    (storage / "chroma").mkdir(parents=True)
    (storage / "chroma" / "chroma.sqlite3").write_text("{ this is not json", encoding="utf-8")

    manager = IndexManager(storage_dir=storage, embed_model=CountingEmbedding(embed_dim=8))
    assert manager.load_or_create() is None
    assert manager.load_error

    import pytest
    with pytest.raises(RuntimeError):
        manager.add_uploaded_file("a.txt", TEXT_A.encode("utf-8"), uploads)
    assert (storage / "chroma" / "chroma.sqlite3").read_text(encoding="utf-8") == "{ this is not json"


def test_renamed_file_keeps_chunks_without_reembedding(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    new_manager(storage).sync_folder(kb, "knowledge_base")

    (kb / "a.txt").rename(kb / "renamed.txt")
    embed = CountingEmbedding(embed_dim=8)
    manager = new_manager(storage, embed)
    manager.sync_folder(kb, "knowledge_base")

    assert embed.call_count == 0
    _assert_consistent(manager)
    reloaded = new_manager(storage)
    node = reloaded.all_nodes()[0]
    assert node.metadata["file_name"] == "renamed.txt"
    assert reloaded.list_documents()[0]["file_name"] == "renamed.txt"


def test_deleting_original_keeps_identical_copy_indexed(dirs):
    kb, _, storage = dirs
    (kb / "a.txt").write_text(TEXT_A, encoding="utf-8")
    (kb / "copy.txt").write_text(TEXT_A, encoding="utf-8")
    manager = new_manager(storage)
    manager.sync_folder(kb, "knowledge_base")
    assert manager.stats()["num_documents"] == 1   # duplicate stored once

    (kb / "a.txt").unlink()
    embed = CountingEmbedding(embed_dim=8)
    manager = new_manager(storage, embed)
    manager.sync_folder(kb, "knowledge_base")
    assert embed.call_count == 0
    assert manager.list_documents()[0]["file_name"] == "copy.txt"
    _assert_consistent(manager)


def test_embedding_error_message_never_contains_key(dirs, monkeypatch):
    fake_key = "sk-proj-THISISAFAKEKEY1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)

    class LeakyEmbedding(CountingEmbedding):
        def _get_text_embeddings(self, texts):
            raise RuntimeError(f"Incorrect API key provided: {fake_key}")

    _, uploads, storage = dirs
    manager = new_manager(storage, LeakyEmbedding(embed_dim=8))
    result = manager.add_uploaded_file("a.txt", TEXT_A.encode("utf-8"), uploads)
    assert result.status == "error"
    assert fake_key not in result.message
    assert "THISISAFAKEKEY" not in result.message
