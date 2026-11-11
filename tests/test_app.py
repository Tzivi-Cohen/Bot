"""UI tests with Streamlit AppTest - fake embeddings, fake LLM, temp folders.

A fake key is set so the UI enables the chat; no real OpenAI object is ever
created because both factories are replaced with fakes.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from rag import index_manager, query_engine, summarizer
from tests.conftest import FakeLLM, KeywordEmbedding, make_blank_pdf

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app_env(dirs, monkeypatch):
    kb, uploads, storage = dirs
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", kb)
    monkeypatch.setattr(config, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(config, "STORAGE_DIR", storage)
    monkeypatch.setattr(index_manager, "create_openai_embed_model", KeywordEmbedding)

    def guard():
        raise AssertionError("real OpenAI LLM must not be created in tests")

    monkeypatch.setattr(query_engine, "create_openai_llm", guard)

    def summary_guard(max_tokens):
        raise AssertionError("real summary LLM must not be created in tests")

    monkeypatch.setattr(summarizer, "create_summary_llm", summary_guard)
    st.cache_resource.clear()
    yield kb, uploads, storage
    st.cache_resource.clear()


def use_llm(monkeypatch, text):
    llm = FakeLLM(response_text=text)
    monkeypatch.setattr(query_engine, "create_openai_llm", lambda: llm)
    return llm


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [t.value for t in at.text]
    parts += [e.value for e in (*at.info, *at.warning, *at.error, *at.success)]
    return "\n".join(str(p) for p in parts)


def ltr(name):
    """How a Latin name appears in the HTML (bidi isolated)."""
    return f'<bdi dir="ltr">{name}</bdi>'


def open_folder(at, folder_key):
    at.session_state["view"] = "folder"
    at.session_state["folder"] = folder_key
    at.run()
    assert not at.exception, at.exception
    return at


def button(at, key):
    return next(b for b in at.button if b.key == key)


def capture_search(monkeypatch):
    """Records the text each RagEngine.retrieve() call searched for."""
    searched = []
    original = query_engine.RagEngine.retrieve

    def spy(self, question):
        searched.append(question)
        return original(self, question)

    monkeypatch.setattr(query_engine.RagEngine, "retrieve", spy)
    return searched


def button_keys(at, prefix):
    return [b.key for b in at.button if b.key and b.key.startswith(prefix)]


def test_without_api_key_chat_is_disabled(app_env):
    at = run_app()
    assert any("OPENAI_API_KEY" in w.value for w in at.warning)
    assert at.chat_input[0].disabled
    assert at.metric[0].value == "0"


def test_knowledge_base_indexed_and_status_shown(app_env, monkeypatch):
    kb, _, storage = app_env
    (kb / "median.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    make_blank_pdf(kb / "scan.pdf")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")

    at = run_app()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"מסמכים במאגר": "1", "נושאים": "1"}
    text = all_text(open_folder(at, "__general__"))
    assert ltr("median.txt") in text and "✅ במאגר הידע" in text
    assert ltr("scan.pdf") in text and "סרוק" in text
    assert (storage / "chroma" / "chroma.sqlite3").exists()


def test_index_loaded_from_disk_on_restart(app_env, monkeypatch):
    kb, _, _ = app_env
    (kb / "median.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    run_app()

    st.cache_resource.clear()  # simulate a new server process
    at = run_app()
    assert any("נטען בהצלחה" in s.value for s in at.success)


def test_chat_shows_answer_and_sources(app_env, monkeypatch):
    kb, _, _ = app_env
    (kb / "median.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    (kb / "mean.txt").write_text("ממוצע הוא סכום חלקי כמות.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    llm = use_llm(monkeypatch, "החציון הוא הערך האמצעי [1].")

    at = run_app()
    at.chat_input[0].set_value("מה זה חציון?").run()
    assert not at.exception

    text = all_text(at)
    assert "החציון הוא הערך האמצעי [1]." in text
    assert f"📄 {ltr('median.txt')}" in text
    assert "מקורות (1)" in [e.label for e in at.expander]
    assert len(llm.prompts) == 1

    # "open document" from the source navigates to the file view
    button(at, "src_1_0").click().run()
    assert at.session_state["view"] == "file"
    assert at.session_state["file"].endswith("median.txt")
    assert "חציון הוא הערך האמצעי" in all_text(at)


def test_chat_no_answer_has_no_sources(app_env, monkeypatch):
    kb, _, _ = app_env
    (kb / "median.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    use_llm(monkeypatch, config.NO_ANSWER_MESSAGE)

    at = run_app()
    at.chat_input[0].set_value("איך מכינים עוגה?").run()
    assert config.NO_ANSWER_MESSAGE in all_text(at)
    assert not at.expander


def test_chat_api_error_shown(app_env, monkeypatch):
    from tests.conftest import AuthenticationError

    kb, _, _ = app_env
    (kb / "median.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    llm = FakeLLM(error=AuthenticationError("401"))
    monkeypatch.setattr(query_engine, "create_openai_llm", lambda: llm)

    at = run_app()
    at.chat_input[0].set_value("מה זה חציון?").run()
    assert not at.exception
    assert any("מפתח ה־API" in e.value for e in at.error)


def test_corrupt_index_shows_error(app_env):
    _, _, storage = app_env
    (storage / "chroma").mkdir(parents=True)
    (storage / "chroma" / "chroma.sqlite3").write_text("{ broken", encoding="utf-8")
    at = run_app()
    assert any("טעינת האינדקס" in e.value for e in at.error)
    assert at.chat_input[0].disabled


def test_document_list_without_api_key(app_env):
    kb, _, _ = app_env
    (kb / "notes.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    make_blank_pdf(kb / "scan.pdf")

    at = open_folder(run_app(), "__general__")
    text = all_text(at)
    assert ltr("notes.txt") in text and "🕓 תקין, עדיין לא נוסף למאגר הידע" in text
    assert ltr("scan.pdf") in text and "⚠️" in text and "סרוק" in text


def test_document_list_shown_even_if_index_is_corrupt(app_env):
    kb, _, storage = app_env
    (kb / "notes.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    (storage / "chroma").mkdir(parents=True)
    (storage / "chroma" / "chroma.sqlite3").write_text("{ broken", encoding="utf-8")
    assert "notes.txt" in all_text(run_app())


def test_restart_does_not_reembed_existing_documents(app_env, monkeypatch):
    from tests.conftest import CountingEmbedding

    kb, _, _ = app_env
    (kb / "a.txt").write_text("ממוצע הוא סכום חלקי כמות.", encoding="utf-8")
    (kb / "b.txt").write_text("חציון הוא הערך האמצעי.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")

    first = CountingEmbedding(embed_dim=8)
    monkeypatch.setattr(index_manager, "create_openai_embed_model", lambda: first)
    run_app()
    assert first.call_count == 2

    second = CountingEmbedding(embed_dim=8)
    monkeypatch.setattr(index_manager, "create_openai_embed_model", lambda: second)
    st.cache_resource.clear()          # new server process
    at = run_app()
    at.run()                           # and a normal page refresh
    assert second.call_count == 0
    assert {m.label: m.value for m in at.metric}["מסמכים במאגר"] == "2"


def test_document_list_grouped_by_topic(app_env):
    kb, _, _ = app_env
    (kb / "Arduino").mkdir()
    (kb / "Arduino" / "pwm.txt").write_text("Arduino PWM", encoding="utf-8")
    (kb / "STM32").mkdir()
    (kb / "STM32" / "gpio.txt").write_text("STM32 GPIO", encoding="utf-8")

    at = run_app()
    text = all_text(at)
    assert f'<div class="sm-folder-title">{ltr("Arduino")}</div>' in text
    assert f'<div class="sm-folder-title">{ltr("STM32")}</div>' in text
    assert button_keys(at, "file_open_") == []      # home shows folder cards, not files
    assert len(button_keys(at, "open_folder_")) == 2

    button(at, "open_folder_0").click().run()      # cards follow the topic order: Arduino first
    assert at.session_state["folder"] == "Arduino"
    assert button_keys(at, "file_open_") == ["file_open_folder_0"]   # only pwm.txt
    assert ltr("pwm.txt") in all_text(at)


def test_latin_topic_names_are_direction_isolated(app_env):
    """"C++" inside RTL text must be wrapped in LRI/PDI so it isn't shown as "++C"."""
    kb, _, _ = app_env
    (kb / "C++").mkdir()
    (kb / "C++" / "classes.txt").write_text("C++ constructor", encoding="utf-8")
    assert f'<div class="sm-folder-title">{ltr("C++")}</div>' in all_text(run_app())


def test_incomplete_key_warning(app_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-short")
    at = run_app()
    assert any("נראה לא שלם" in w.value for w in at.warning)
    assert "sk-short" not in all_text(at)


def test_modified_file_with_failed_update_shows_previous_version(app_env, monkeypatch):
    """Without a key the update can't run: the list shows the old version is still indexed."""
    from tests.conftest import KeywordEmbedding

    kb, _, storage = app_env
    (kb / "notes.txt").write_text("גרסה ראשונה", encoding="utf-8")
    from rag.index_manager import IndexManager
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")
    (kb / "notes.txt").write_text("גרסה שנייה", encoding="utf-8")

    text = all_text(open_folder(run_app(), "__general__"))     # no API key -> no sync
    assert ltr("notes.txt") in text and "🔄" in text and "הגרסה הקודמת" in text


def test_recovery_message_shown_in_status(app_env):
    from tests.conftest import KeywordEmbedding
    from rag.index_manager import IndexManager

    kb, _, storage = app_env
    (kb / "a.txt").write_text("טקסט", encoding="utf-8")
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")
    (storage / "manifest.tmp").write_text("{ half", encoding="utf-8")

    at = run_app()
    assert any("שחזור אוטומטי" in i.value for i in at.info)


# ------------------------------------------------------------- new library UI
STUDY_TEXT = """Python – יסודות
================

1. חלוקה
--------
האופרטור // מחזיר int עבור שני int: 7 // 2 == 3.

    print(7 // 2)

- נקודה ראשונה
- נקודה שנייה
"""


def make_python_folder(kb):
    (kb / "Python").mkdir()
    (kb / "Python" / "basics.txt").write_text(STUDY_TEXT, encoding="utf-8")
    (kb / "Arduino").mkdir()
    (kb / "Arduino" / "pwm.txt").write_text("Arduino analogWrite PWM", encoding="utf-8")


def test_navigation_folder_file_and_back(app_env):
    kb, _, _ = app_env
    make_python_folder(kb)
    at = run_app()
    button(at, "open_folder_0").click().run()            # Python is first in topic order
    assert at.session_state["view"] == "folder"
    button(at, "file_open_folder_0").click().run()
    assert at.session_state["view"] == "file"
    text = all_text(at)
    assert "<h1>" in text and "<h2>" in text and "<li>" in text
    assert '7 // 2 == 3</bdi>' in text                     # inline code is LTR-isolated
    assert at.code and at.code[0].value == "print(7 // 2)" and at.code[0].language == "python"
    assert "ספריית הלימוד" in text and ltr("basics.txt") in text   # breadcrumbs

    button(at, "nav_back").click().run()
    assert at.session_state["view"] == "folder"
    button(at, "nav_back").click().run()
    assert at.session_state["view"] == "home"


def test_search_finds_file_by_content_and_filters(app_env):
    kb, _, _ = app_env
    make_python_folder(kb)
    at = run_app()
    at.text_input(key="search").set_value("analogWrite").run()
    text = all_text(at)
    assert "תוצאה אחת" in text and ltr("pwm.txt") in text
    assert button_keys(at, "file_open_search_") == ["file_open_search_0"]
    at.text_input(key="search").set_value("nothing-like-this").run()
    assert any("לא נמצאו מסמכים מתאימים" in i.value for i in at.info)


def test_ask_about_file_focus_adds_topic_hint(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    llm = use_llm(monkeypatch, "תשובה [1].")
    searched = capture_search(monkeypatch)
    at = run_app()
    at.session_state["view"] = "file"
    at.session_state["folder"] = "Python"
    at.session_state["file"] = str((kb / "Python" / "basics.txt").resolve())
    at.run()
    button(at, "focus_file").click().run()
    assert "שאלות על" in all_text(at)
    at.chat_input[0].set_value("מה עושה //?").run()
    assert "שאלה: מה עושה //?\n" in llm.prompts[-1]          # original question for the model
    assert searched[-1] == "מה עושה //? (Python)"                # topic hint used for retrieval
    button(at, "clear_focus").click().run()
    assert at.session_state["focus"] is None


def test_suggestion_button_asks_question(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    llm = use_llm(monkeypatch, "תשובה [1].")
    at = run_app()
    button(at, "suggest_0").click().run()
    assert len(llm.prompts) == 1
    assert len(at.session_state["messages"]) == 2


def test_chat_history_kept_during_session(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    use_llm(monkeypatch, "תשובה [1].")
    at = run_app()
    at.chat_input[0].set_value("שאלה ראשונה").run()
    button(at, "open_folder_0").click().run()             # navigate in the library
    at.chat_input[0].set_value("שאלה שנייה").run()
    roles = [m["role"] for m in at.session_state["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


# ------------------------------------------------------------------ summaries
SUMMARY = """# Python
## חלוקה
האופרטור // [1]
```python
print(7 // 2)
```"""


def use_summary_llm(monkeypatch, text=SUMMARY):
    llm = FakeLLM(response_text=text)
    monkeypatch.setattr(summarizer, "create_summary_llm", lambda max_tokens: llm)
    return llm


def keyed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")


def test_topic_summary_button_shows_cost_and_waits_for_approval(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "לא אמור להיקרא")
    sum_llm = use_summary_llm(monkeypatch)
    at = run_app()
    open_folder(at, "Python")
    button(at, "summarize_folder").click().run()

    text = all_text(at)
    assert "סיכום הנושא:" in text and "נמצא מסמך אחד בנושא." in text
    assert "המערכת מוכנה להכין סיכום של חומר הלימוד." in text
    for technical in ("עלות", "$", "טוקן", "קריאות API"):
        assert technical not in text
    assert button(at, "confirm_summary").label == "יצירת סיכום"
    assert sum_llm.prompts == [] and rag_llm.prompts == []        # nothing paid yet

    button(at, "confirm_summary").click().run()
    assert len(sum_llm.prompts) == 1 and rag_llm.prompts == []
    assert "print(7 // 2)" in sum_llm.prompts[0]                  # whole file material sent
    last = at.session_state["messages"][-1]
    assert last["summary"] and last["found"] and "## חלוקה" in last["content"]
    assert [s["file_name"] for s in last["summary_sources"]] == ["basics.txt"]
    assert "מקורות הסיכום (1)" in [e.label for e in at.expander]
    assert at.session_state["summary_request"] is None

    button(at, "sumsrc_1_0").click().run()                         # open the source
    assert at.session_state["view"] == "file"
    assert at.session_state["file"].endswith("basics.txt")


def test_cancel_summary_makes_no_call(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    sum_llm = use_summary_llm(monkeypatch)
    at = open_folder(run_app(), "Python")
    button(at, "summarize_folder").click().run()
    button(at, "cancel_summary").click().run()
    assert sum_llm.prompts == [] and at.session_state["summary_request"] is None
    assert not button_keys(at, "confirm_summary")


def test_chat_summary_request_is_routed_to_right_topic_without_call(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "x")
    sum_llm = use_summary_llm(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("הסכם לי את החומר בפייתון בצורה טובה ומקצועית").run()
    assert at.session_state["summary_request"] == {"scope": "topic", "key": "Python"}
    assert rag_llm.prompts == [] and sum_llm.prompts == []
    assert f"סיכום הנושא: {ltr('Python')}" in all_text(at)
    button(at, "confirm_summary").click().run()
    assert len(sum_llm.prompts) == 1 and "analogWrite" not in sum_llm.prompts[0]   # not Arduino


def test_ambiguous_summary_asks_for_topic(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "תשובה רגילה")
    sum_llm = use_summary_llm(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("סכם לי בבקשה").run()
    assert rag_llm.prompts == [] and sum_llm.prompts == []
    assert set(button_keys(at, "choose_")) == {"choose_1_Python", "choose_1_Arduino"}
    button(at, "choose_1_Arduino").click().run()
    assert at.session_state["summary_request"] == {"scope": "topic", "key": "Arduino"}
    assert sum_llm.prompts == []


def test_ambiguous_summary_can_be_sent_as_regular_question(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "תשובה [1].")
    at = run_app()
    at.chat_input[0].set_value("מה זה הסכם?").run()
    button(at, "as_question_1").click().run()
    assert len(rag_llm.prompts) == 1 and "שאלה: מה זה הסכם?" in rag_llm.prompts[0]


def test_file_summary_uses_only_that_file(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    (kb / "Python" / "more.txt").write_text("קובץ נוסף על פונקציות", encoding="utf-8")
    keyed(monkeypatch)
    sum_llm = use_summary_llm(monkeypatch, "# סיכום הקובץ [1]")
    at = run_app()
    at.session_state["view"], at.session_state["folder"] = "file", "Python"
    at.session_state["file"] = str((kb / "Python" / "basics.txt").resolve())
    at.run()
    button(at, "summarize_file").click().run()
    assert at.session_state["summary_request"]["scope"] == "file"
    button(at, "confirm_summary").click().run()
    assert len(sum_llm.prompts) == 1
    assert "basics.txt" in sum_llm.prompts[0] and "more.txt" not in sum_llm.prompts[0]


def test_summary_cannot_run_without_api_key(app_env, monkeypatch):
    kb, _, storage = app_env
    make_python_folder(kb)
    from rag.index_manager import IndexManager
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")                   # indexed, but no key now
    at = open_folder(run_app(), "Python")
    button(at, "summarize_folder").click().run()
    assert button(at, "confirm_summary").disabled


def test_regular_question_still_uses_hybrid_search_with_name_mapping(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "תשובה [1].")
    sum_llm = use_summary_llm(monkeypatch)
    searched = capture_search(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("מה עושה analogWrite בארדואינו?").run()
    assert sum_llm.prompts == [] and len(rag_llm.prompts) == 1
    prompt = rag_llm.prompts[0]
    assert "שאלה: מה עושה analogWrite בארדואינו?\n" in prompt          # model sees the original
    assert "Arduino" in searched[-1] and "ארדואינו" not in searched[-1]  # local name mapping
    assert prompt.count("[1] (קובץ:") == 1                                # regular Q&A prompt
    sources = at.session_state["messages"][-1]["sources"]
    assert any(s.keyword_rank is not None for s in sources)              # BM25 half of hybrid used


# ------------------------------------------- regression: no technical details
TECHNICAL = ("מקטע", "Chunk", "chunk", "טוקן", "$", "קריאות API", "עלות", "אינדקס")


def assert_no_technical_text(at):
    text = all_text(at) + "\n".join(str(m.label) + str(m.value) for m in at.metric)
    text = text.split("</style>", 1)[-1]               # ignore the CSS block
    for word in TECHNICAL:
        assert word not in text, word


def test_no_chunks_tokens_or_costs_anywhere(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-" + "x" * 50)    # well-formed fake key
    use_llm(monkeypatch, "תשובה [1].")
    use_summary_llm(monkeypatch)
    at = run_app()
    assert_no_technical_text(at)                                   # home + status window
    pill = next(m.value for m in at.markdown if "sm-pill ok" in m.value)
    assert "מאגר הידע מוכן • 2 מסמכים • 2 נושאים" in pill          # real numbers

    open_folder(at, "Python")
    assert_no_technical_text(at)                                   # folder view
    button(at, "summarize_folder").click().run()
    assert_no_technical_text(at)                                   # summary card
    button(at, "confirm_summary").click().run()
    assert_no_technical_text(at)                                   # summary message
    at.text_input(key="search").set_value("פייתון").run()
    assert_no_technical_text(at)                                   # search results
    button(at, "file_open_search_0").click().run()
    assert_no_technical_text(at)                                   # document view


def test_hebrew_topic_search_in_ui(app_env):
    kb, _, _ = app_env
    make_python_folder(kb)
    at = run_app()
    for query in ("פייתון", "בפייתון", "Python"):
        at.text_input(key="search").set_value(query).run()
        assert button_keys(at, "file_open_search_") == ["file_open_search_0"], query
        assert "נמצא בנושא" in all_text(at)
        assert button_keys(at, "goto_topic_")                       # "go to folder" button
    button(at, button_keys(at, "goto_topic_")[0]).click().run()
    assert at.session_state["view"] == "folder" and at.session_state["folder"] == "Python"


def test_follow_up_question_keeps_previous_topic(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    use_llm(monkeypatch, "תשובה [1].")
    searched = capture_search(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("מה זה לולאה בפייתון?").run()
    at.chat_input[0].set_value("תן לי דוגמה מתוך החומר").run()
    assert "Python" in searched[0] and "פייתון" not in searched[0]
    # no content of its own -> the previous question's subject and topic are searched
    assert searched[1].startswith("תן לי דוגמה מתוך החומר")
    assert "לולאה" in searched[1] and "Python" in searched[1] and "Loops" in searched[1]


def test_document_view_formats_lists_labels_and_code(app_env):
    kb, _, _ = app_env
    (kb / "C++").mkdir()
    (kb / "C++" / "demo.txt").write_text(
        "C++ – מחלקות\n============\n\nשלושה מקרים:\n1) ראשון\n2) שני\n\n"
        "שאלה 1: מה יודפס?\nפתרון: \u200f10 20.\n\n    std::cout << x;\n", encoding="utf-8")
    at = run_app()
    at.session_state["view"], at.session_state["folder"] = "file", "C++"
    at.session_state["file"] = str((kb / "C++" / "demo.txt").resolve())
    at.run()
    text = all_text(at)
    assert "<ol><li>ראשון</li><li>שני</li></ol>" in text
    assert "<strong>שאלה 1:</strong>" in text and "<strong>פתרון:</strong>" in text
    assert "\u200f" not in text.split("</style>", 1)[-1]
    assert at.code[0].value == "std::cout << x;" and at.code[0].language == "cpp"
    assert '<h1><bdi dir="ltr">C++</bdi> – מחלקות</h1>' in text


TREES_TEXT = """מבני נתונים – עצים בינאריים
===========================

1. מבוא
-------
לכל צומת בעץ בינארי יש עד שני ילדים.

2. סריקות עץ (Tree Traversals)
-------------------------------
    def inorder(node, out):            # שמאל, שורש, ימין
        if node:
            inorder(node.left, out)

סריקה לפי רמות (Level Order / BFS) משתמשת בתור.
"""


def test_sub_topic_summary_from_chat_without_choosing_a_topic(app_env, monkeypatch):
    """"אפשר סיכום של סריקות עצים באלגוריתמים": a card for the sub-topic (not
    "איזה נושא לסכם?"), no call before approval, then one call with that material."""
    kb, _, _ = app_env
    make_python_folder(kb)
    (kb / "Data-Structures-and-Algorithms").mkdir()
    (kb / "Data-Structures-and-Algorithms" / "trees.txt").write_text(TREES_TEXT, encoding="utf-8")
    keyed(monkeypatch)
    rag_llm = use_llm(monkeypatch, "x")
    sum_llm = use_summary_llm(monkeypatch, "# סיכום\nסריקה לפי רמות משתמשת בתור [1].")
    at = run_app()
    at.chat_input[0].set_value("אפשר סיכום של סריקות עצים באלגוריתמים").run()
    assert at.session_state["summary_request"] == {
        "scope": "subtopic", "key": "סריקות עצים", "topic": "Data-Structures-and-Algorithms"}
    text = all_text(at)
    assert "סיכום הנושא: סריקות עצים" in text and "איזה נושא לסכם" not in text
    assert "נמצא חומר מתאים במסמך אחד." in text
    assert rag_llm.prompts == [] and sum_llm.prompts == []          # nothing sent yet
    button(at, "confirm_summary").click().run()
    assert len(sum_llm.prompts) == 1
    assert "Tree Traversals" in sum_llm.prompts[0] and "analogWrite" not in sum_llm.prompts[0]


def test_sub_topic_summary_not_found_is_said_clearly(app_env, monkeypatch):
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    sum_llm = use_summary_llm(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("סכם לי את רשתות הנוירונים הקוונטיות").run()
    assert "לא נמצא במאגר הידע חומר על" in all_text(at)
    assert not button_keys(at, "confirm_summary") or at.button(key="confirm_summary").disabled
    assert sum_llm.prompts == []


def test_explanation_question_and_follow_up_in_chat(app_env, monkeypatch):
    """"אני לא מבינה את..." is searched without the request words; a follow-up
    gets the previous question as context for the model."""
    kb, _, _ = app_env
    (kb / "Data-Structures-and-Algorithms").mkdir()
    (kb / "Data-Structures-and-Algorithms" / "trees.txt").write_text(TREES_TEXT, encoding="utf-8")
    keyed(monkeypatch)
    llm = use_llm(monkeypatch, "תשובה [1].")
    searched = capture_search(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("אני לא מבינה את עניין הסריקות בעצים").run()
    assert "מבינה" not in searched[0] and "Tree Traversals" in searched[0]
    assert "שאלה: אני לא מבינה את עניין הסריקות בעצים" in llm.prompts[-1]
    at.chat_input[0].set_value("תסבירי את זה שוב").run()
    assert "הסריקות בעצים" in searched[1]
    assert "השאלה הקודמת בשיחה (רק להבנת ההקשר): אני לא מבינה את עניין הסריקות בעצים" in llm.prompts[-1]


def test_new_question_does_not_inherit_the_previous_topic(app_env, monkeypatch):
    """"איך עובדות פסיקות?" after a Python question is a new subject - the search
    must not get "Python" added (it pulled unrelated chunks)."""
    kb, _, _ = app_env
    make_python_folder(kb)
    keyed(monkeypatch)
    use_llm(monkeypatch, "תשובה [1].")
    searched = capture_search(monkeypatch)
    at = run_app()
    at.chat_input[0].set_value("מה זה לולאה בפייתון?").run()
    at.chat_input[0].set_value("איך עובדות פסיקות?").run()
    assert "Python" in searched[0]
    assert "Python" not in searched[1] and "Interrupt" in searched[1]
    at.chat_input[0].set_value("ומה לגבי לולאות?").run()          # a real follow-up
    assert "Loops" in searched[2]
