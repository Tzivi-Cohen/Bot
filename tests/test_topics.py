"""Sub-folder (topic) support, SSL handling and key checks - no API calls.

The documents created here are temporary test files in a temp folder,
never in the real knowledge_base.
"""

import ssl

import httpx
import openai
import pytest

import config
from rag.document_loader import iter_supported_files, relative_path_for, topic_for
from rag.errors import friendly_api_error
from rag.index_manager import IndexManager
from rag.query_engine import RagEngine, RagError
from tests.conftest import CountingEmbedding, FailingEmbedding, FakeLLM, KeywordEmbedding

ARDUINO = "Arduino: הפונקציה analogWrite מייצרת אות PWM על פין דיגיטלי."
STM32 = "STM32: הפונקציה HAL_GPIO_WritePin משנה את מצב הפין."
PYTHON = "Python: רשימה (list) היא מבנה נתונים שניתן לשנות."


def topic_embedding():
    return KeywordEmbedding(vocab=["Arduino", "analogWrite", "PWM", "STM32",
                                   "HAL_GPIO", "Python", "list", "רשימה"])


def build_kb(kb):
    (kb / "Arduino").mkdir()
    (kb / "STM32" / "HAL").mkdir(parents=True)
    (kb / "Python").mkdir()
    (kb / "Arduino" / "pwm.txt").write_text(ARDUINO, encoding="utf-8")
    (kb / "STM32" / "HAL" / "gpio.txt").write_text(STM32, encoding="utf-8")
    (kb / "Python" / "lists.txt").write_text(PYTHON, encoding="utf-8")
    (kb / "root_file.txt").write_text("קובץ בשורש התיקייה.", encoding="utf-8")


def manager_for(storage, embed=None):
    m = IndexManager(storage_dir=storage, embed_model=embed or CountingEmbedding(embed_dim=8))
    m.load_or_create()
    return m


def node_meta(manager):
    return {n.metadata["file_name"]: n.metadata for n in manager.all_nodes()}


# --------------------------------------------------------------- scanning
def test_recursive_scan_skips_hidden_and_temp_files(dirs):
    kb, _, _ = dirs
    build_kb(kb)
    (kb / ".git").mkdir()
    (kb / ".git" / "x.txt").write_text("hidden", encoding="utf-8")
    (kb / "Python" / "~$draft.docx").write_bytes(b"lock")
    (kb / "Python" / ".gitkeep").write_bytes(b"")

    names = {p.relative_to(kb).as_posix() for p in iter_supported_files(kb)}
    assert names == {"Arduino/pwm.txt", "STM32/HAL/gpio.txt",
                     "Python/lists.txt", "root_file.txt"}


def test_topic_and_relative_path_helpers(tmp_path):
    assert topic_for(tmp_path / "STM32" / "HAL" / "gpio.pdf", tmp_path) == "STM32"
    assert topic_for(tmp_path / "a.txt", tmp_path) is None
    assert relative_path_for(tmp_path / "STM32" / "HAL" / "gpio.pdf", tmp_path) == "STM32/HAL/gpio.pdf"


def test_topic_and_relative_path_saved_in_metadata(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")

    meta = node_meta(manager)
    assert meta["gpio.txt"]["topic"] == "STM32"
    assert meta["gpio.txt"]["relative_path"] == "STM32/HAL/gpio.txt"
    assert meta["pwm.txt"]["topic"] == "Arduino"
    assert "topic" not in meta["root_file.txt"]
    entry = next(e for e in manager.manifest.values() if e["file_name"] == "gpio.txt")
    assert entry["topic"] == "STM32" and entry["relative_path"] == "STM32/HAL/gpio.txt"


def test_topic_sent_to_model_but_internal_path_is_not(dirs):
    from llama_index.core.schema import MetadataMode

    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")
    node = next(n for n in manager.all_nodes()
                if n.metadata["file_name"] == "gpio.txt")
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        content = node.get_content(metadata_mode=mode)
        assert "STM32" in content and "gpio.txt" in content
        assert "STM32/HAL/gpio.txt" not in content


# ----------------------------------------------------------- incremental
def test_duplicate_across_topic_folders_indexed_once(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    (kb / "STM32" / "copy_of_pwm.txt").write_text(ARDUINO, encoding="utf-8")
    embed = CountingEmbedding(embed_dim=8)
    manager = manager_for(storage, embed)
    manager.sync_folder(kb, "knowledge_base")

    assert manager.stats()["num_documents"] == 4
    assert embed.call_count == 4
    names = [e["file_name"] for e in manager.manifest.values()]
    assert names.count("pwm.txt") + names.count("copy_of_pwm.txt") == 1


def test_new_file_in_subfolder_only_embeds_that_file(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager_for(storage).sync_folder(kb, "knowledge_base")

    (kb / "Python" / "dicts.txt").write_text("Python: מילון (dict) שומר זוגות.", encoding="utf-8")
    embed = CountingEmbedding(embed_dim=8)
    manager = manager_for(storage, embed)
    results = manager.sync_folder(kb, "knowledge_base")

    assert [r.file_name for r in results] == ["dicts.txt"]
    assert embed.call_count == 1
    assert "Python/dicts.txt" in manager.last_sync_results


def test_reload_with_subfolders_does_not_reembed(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager_for(storage).sync_folder(kb, "knowledge_base")

    embed = CountingEmbedding(embed_dim=8)
    manager = manager_for(storage, embed)
    assert manager.sync_folder(kb, "knowledge_base") == []
    assert embed.call_count == 0
    assert manager.loaded_from_storage
    assert node_meta(manager)["gpio.txt"]["topic"] == "STM32"


def test_move_between_topics_updates_metadata_without_embeddings(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager_for(storage).sync_folder(kb, "knowledge_base")

    (kb / "Embedded-Systems").mkdir()
    (kb / "STM32" / "HAL" / "gpio.txt").rename(kb / "Embedded-Systems" / "gpio_basics.txt")
    embed = CountingEmbedding(embed_dim=8)
    manager = manager_for(storage, embed)
    manager.sync_folder(kb, "knowledge_base")
    assert embed.call_count == 0

    meta = node_meta(manager_for(storage))["gpio_basics.txt"]   # after reload
    assert meta["topic"] == "Embedded-Systems"
    assert meta["relative_path"] == "Embedded-Systems/gpio_basics.txt"
    assert "gpio.txt" not in node_meta(manager)


def test_move_from_subfolder_to_root_removes_topic(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")
    (kb / "Python" / "lists.txt").rename(kb / "lists.txt")
    manager.sync_folder(kb, "knowledge_base")
    assert "topic" not in node_meta(manager)["lists.txt"]


def test_delete_file_in_subfolder(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")

    (kb / "STM32" / "HAL" / "gpio.txt").unlink()
    manager.sync_folder(kb, "knowledge_base")
    reloaded = manager_for(storage)
    assert "gpio.txt" not in node_meta(reloaded)
    assert reloaded.num_vectors() == len(reloaded.all_nodes()) == 3


def test_deleting_whole_topic_folder(dirs):
    import shutil

    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")
    shutil.rmtree(kb / "STM32")
    manager.sync_folder(kb, "knowledge_base")
    assert {m.get("topic") for m in node_meta(manager).values()} == {"Arduino", "Python", None}


def test_uploads_sync_does_not_touch_knowledge_base_entries(dirs):
    kb, uploads, storage = dirs
    build_kb(kb)
    manager = manager_for(storage)
    manager.sync_folder(kb, "knowledge_base")
    uploads.mkdir()
    manager.sync_folder(uploads, "uploads")          # empty uploads folder
    assert manager.stats()["num_documents"] == 4


def test_error_in_subfolder_keeps_index_intact(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager_for(storage).sync_folder(kb, "knowledge_base")
    manifest_before = (storage / "manifest.json").read_text(encoding="utf-8")

    (kb / "Arduino" / "new.txt").write_text("Arduino: digitalRead.", encoding="utf-8")
    failing = IndexManager(storage_dir=storage, embed_model=FailingEmbedding(embed_dim=8))
    failing.load_or_create()
    results = failing.sync_folder(kb, "knowledge_base")

    assert [r.status for r in results] == ["error"]
    assert (storage / "manifest.json").read_text(encoding="utf-8") == manifest_before
    reloaded = manager_for(storage)
    assert reloaded.stats()["num_documents"] == 4
    assert reloaded.num_vectors() == 4


# ----------------------------------------------------------- multi-topic RAG
def test_rag_retrieves_from_the_right_topic(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage, topic_embedding())
    manager.sync_folder(kb, "knowledge_base")

    llm = FakeLLM(response_text="analogWrite מייצרת PWM [1].")
    engine = RagEngine(manager.index, llm=llm)

    arduino = engine.ask("איך מייצרים PWM ב־Arduino עם analogWrite?")
    assert arduino.sources[0].file_name == "pwm.txt"
    assert arduino.sources[0].topic == "Arduino"
    assert "נושא: Arduino" in llm.prompts[-1]

    python = engine.ask("מה זה list ב־Python?")
    assert python.sources[0].topic == "Python"

    stm = engine.retrieve("מה עושה HAL_GPIO ב־STM32?")
    assert stm[0].topic == "STM32"


def test_rag_refuses_question_outside_all_topics(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage, topic_embedding())
    manager.sync_folder(kb, "knowledge_base")
    llm = FakeLLM(response_text=config.NO_ANSWER_MESSAGE)
    result = RagEngine(manager.index, llm=llm).ask("מי זכה במונדיאל 2018?")
    assert result.found is False and result.sources == []


# ------------------------------------------------------------------- SSL
def _openai_ssl_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    error = openai.APIConnectionError(request=request)
    error.__cause__ = ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    return error


def test_real_openai_ssl_error_class_gets_ssl_message():
    message = friendly_api_error(_openai_ssl_error())
    assert "SSL" in message and "אנטי־וירוס" in message


def test_ssl_error_while_answering(dirs):
    kb, _, storage = dirs
    build_kb(kb)
    manager = manager_for(storage, topic_embedding())
    manager.sync_folder(kb, "knowledge_base")
    llm = FakeLLM(error=_openai_ssl_error())
    with pytest.raises(RagError, match="SSL"):
        RagEngine(manager.index, llm=llm).ask("מה זה PWM?")


def test_ssl_error_while_indexing_is_reported(dirs):
    class SslFailingEmbedding(CountingEmbedding):
        def _get_text_embeddings(self, texts):
            raise _openai_ssl_error()

    _, uploads, storage = dirs
    manager = manager_for(storage, SslFailingEmbedding(embed_dim=8))
    result = manager.add_uploaded_file("a.txt", PYTHON.encode("utf-8"), uploads)
    assert result.status == "error" and "SSL" in result.message
    assert manager.stats()["num_chunks"] == 0


def test_truststore_active_and_verification_still_enabled():
    """config injects truststore; certificates and host names are still checked."""
    import config  # noqa: F401  (import triggers the injection)

    context = ssl.create_default_context()
    assert type(context).__module__.startswith("truststore")
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_no_verify_false_anywhere_in_source():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for path in [*root.glob("*.py"), *(root / "rag").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "verify=False" not in text and "CERT_NONE" not in text, path


# ------------------------------------------------------------- API key
@pytest.mark.parametrize("value, expected", [
    ("", "missing"),
    ("   ", "missing"),
    ("sk-short123", "incomplete"),
    ("sk-" + "a" * 60 + " ", "incomplete"),
    ('"sk-' + "a" * 60 + '"', "incomplete"),
    ("not-a-key-" + "a" * 60, "incomplete"),
    ("sk-proj-" + "A1_b-" * 30, "ok"),
])
def test_api_key_status(monkeypatch, value, expected):
    monkeypatch.setenv("OPENAI_API_KEY", value)
    assert config.api_key_status() == expected


def test_models_and_limits_match_requirements():
    from rag.index_manager import create_openai_embed_model
    from rag.query_engine import create_openai_llm

    assert config.TOP_K == 3
    assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (512, 50)
    embed = create_openai_embed_model()       # constructing makes no API call
    llm = create_openai_llm()
    assert embed.model_name == "text-embedding-3-small"
    assert llm.model == "gpt-4o-mini"
    assert llm.max_tokens == 600 and llm.temperature == 0
    assert embed.max_retries == 2 and llm.max_retries == 2
