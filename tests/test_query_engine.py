"""RAG engine tests with fake embeddings and a fake LLM (no API calls)."""

import pytest

import config
from rag.index_manager import IndexManager
from rag.query_engine import RagEngine, RagError
from tests.conftest import AuthenticationError, FailingEmbedding, FakeLLM, KeywordEmbedding

# Temporary test documents (created in a temp folder, never in knowledge_base).
DOCS = {
    "mean.txt": "ממוצע חשבוני הוא סכום כל הערכים חלקי מספר הערכים. ממוצע של 10, 20, 30 הוא 20.",
    "median.txt": "חציון הוא הערך האמצעי בקבוצת נתונים מסודרת. החציון של 5, 10, 20 הוא 10.",
    "mode.txt": "שכיח הוא הערך שמופיע בתדירות הגבוהה ביותר. השכיח של 2, 2, 3, 4 הוא 2.",
    "std.txt": "סטיית תקן מודדת את פיזור הנתונים סביב הממוצע.",
    "prob.txt": "הסתברות היא מספר בין 0 ל־1 המבטא סיכוי להתרחשות מאורע.",
}


@pytest.fixture
def manager(dirs):
    kb, _, storage = dirs
    for name, text in DOCS.items():
        (kb / name).write_text(text, encoding="utf-8")
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")
    return m


def test_retrieves_top_k_3(manager):
    llm = FakeLLM()
    result = RagEngine(manager.index, llm=llm).ask("מה זה חציון?")

    assert len(result.retrieved) == config.TOP_K == 3
    prompt = llm.prompts[0]
    assert "[1]" in prompt and "[3]" in prompt and "[4]" not in prompt


def test_most_relevant_chunk_ranked_first(manager):
    engine = RagEngine(manager.index, llm=FakeLLM())
    assert engine.retrieve("איך מחשבים חציון?")[0].file_name == "median.txt"
    assert engine.retrieve("מהו השכיח?")[0].file_name == "mode.txt"


def test_prompt_contains_only_retrieved_chunks_and_rules(manager):
    llm = FakeLLM()
    RagEngine(manager.index, llm=llm).ask("מה זה חציון?")
    prompt = llm.prompts[0]

    assert "אך ורק על סמך המקורות" in prompt
    assert config.NO_ANSWER_MESSAGE in prompt
    assert "מה זה חציון?" in prompt
    assert "קובץ: median.txt" in prompt
    # Not every document is sent to the model - only Top-K.
    sent = sum(f"קובץ: {name}" in prompt for name in DOCS)
    assert sent == 3


def test_sources_are_the_cited_chunks(manager):
    llm = FakeLLM(response_text="החציון הוא הערך האמצעי [1].")
    result = RagEngine(manager.index, llm=llm).ask("מה זה חציון?")

    assert result.found is True
    assert [s.file_name for s in result.sources] == ["median.txt"]
    assert "הערך האמצעי" in result.sources[0].text
    assert result.sources[0].score is not None


def test_answer_without_citations_shows_all_retrieved(manager):
    llm = FakeLLM(response_text="החציון הוא הערך האמצעי.")
    result = RagEngine(manager.index, llm=llm).ask("מה זה חציון?")
    assert len(result.sources) == 3


def test_no_answer_returns_fixed_message_without_sources(manager):
    llm = FakeLLM(response_text=config.NO_ANSWER_MESSAGE)
    result = RagEngine(manager.index, llm=llm).ask("איך מכינים עוגת שוקולד?")

    assert result.found is False
    assert result.answer == config.NO_ANSWER_MESSAGE
    assert result.sources == []          # no invented sources
    assert len(result.retrieved) == 3    # still available for debugging


def test_no_answer_with_extra_text_is_normalised(manager):
    llm = FakeLLM(response_text=f"{config.NO_ANSWER_MESSAGE} אולי כדאי לחפש ברשת.")
    result = RagEngine(manager.index, llm=llm).ask("איך מכינים עוגה?")
    assert result.answer == config.NO_ANSWER_MESSAGE


def test_empty_index_does_not_call_llm(dirs):
    _, _, storage = dirs
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    m.load_or_create()
    llm = FakeLLM()
    result = RagEngine(m.index, llm=llm).ask("מה זה חציון?")

    assert result.found is False
    assert llm.prompts == []


def test_empty_question_rejected(manager):
    llm = FakeLLM()
    with pytest.raises(RagError):
        RagEngine(manager.index, llm=llm).ask("   ")
    assert llm.prompts == []


def test_llm_api_error_becomes_friendly_message(manager):
    llm = FakeLLM(error=AuthenticationError("401"))
    with pytest.raises(RagError, match="מפתח ה־API"):
        RagEngine(manager.index, llm=llm).ask("מה זה חציון?")


def test_embedding_api_error_on_question(dirs, manager):
    manager.index._embed_model = FailingEmbedding(embed_dim=7)
    with pytest.raises(RagError, match="שגיאה"):
        RagEngine(manager.index, llm=FakeLLM()).ask("מה זה חציון?")


def test_question_about_newly_uploaded_document(dirs, manager):
    _, uploads, _ = dirs
    result = manager.add_uploaded_file(
        "cake.txt", "מתכון עוגה: קמח, ביצים וסוכר. אופים 30 דקות.".encode("utf-8"),
        uploads,
    )
    assert result.status == "added"

    llm = FakeLLM(response_text="אופים 30 דקות [1].")
    answer = RagEngine(manager.index, llm=llm).ask("כמה זמן אופים את העוגה?")
    assert answer.sources[0].file_name == "cake.txt"


def test_engine_works_after_reload_from_storage(dirs, manager):
    _, _, storage = dirs
    reloaded = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    reloaded.load_or_create()
    engine = RagEngine(reloaded.index, llm=FakeLLM())
    assert engine.retrieve("מה זה חציון?")[0].file_name == "median.txt"


def test_pdf_page_shown_in_context():
    from rag.query_engine import Source, format_context

    text = format_context([Source(1, "book.pdf", "טקסט", 0.8, page=4)])
    assert "קובץ: book.pdf, עמוד 4" in text


def test_partial_answer_with_sources_is_kept(manager):
    llm = FakeLLM(response_text=f"החציון הוא הערך האמצעי [1]. {config.NO_ANSWER_MESSAGE}")
    result = RagEngine(manager.index, llm=llm).ask("מה זה חציון ומה זה ממוצע הנדסי?")
    assert result.found and result.sources                     # not thrown away as a refusal


def test_refusal_citing_nonexistent_source_is_still_a_refusal(manager):
    llm = FakeLLM(response_text=f"{config.NO_ANSWER_MESSAGE} [9]")
    assert RagEngine(manager.index, llm=llm).ask("מה זה עוגה?").found is False


def test_search_query_is_used_for_retrieval_only(manager):
    llm = FakeLLM()
    engine = RagEngine(manager.index, llm=llm)
    engine.ask("שאלה בעברית בלי מילות מפתח", search_query="חציון")
    assert "שאלה: שאלה בעברית בלי מילות מפתח" in llm.prompts[0]   # model sees the original
    assert "קובץ: median.txt" in llm.prompts[0]                   # retrieval used "חציון"
