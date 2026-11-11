"""Hybrid retrieval (ChromaDB vectors + local BM25, fused with RRF) - no API calls."""

import pytest

import config
from rag.hybrid import KeywordIndex, reciprocal_rank_fusion, tokenize
from rag.index_manager import IndexManager
from rag.query_engine import RagEngine
from tests.conftest import FakeLLM, KeywordEmbedding


# ---------------------------------------------------------------- tokenizer
def test_tokenize_hebrew_prefixes_and_maqaf():
    tokens = tokenize("לשדות של האובייקט ב־Python")
    assert "לשדות" in tokens and "שדות" in tokens          # prefix ל removed as extra token
    assert "האובייקט" in tokens and "אובייקט" in tokens
    assert "python" in tokens                              # maqaf separates, lower-cased
    assert "של" not in tokens                              # stopword


def test_tokenize_code_identifiers_and_operators():
    tokens = tokenize("HAL_GPIO_WritePin(GPIOA, 5); x // 2; a ** b; p->x; std::move; a == b")
    for expected in ["hal_gpio_writepin", "gpioa", "//", "**", "->", "::", "==", "std", "move"]:
        assert expected in tokens, expected


# --------------------------------------------------------------------- BM25
def _nodes(texts):
    from llama_index.core.schema import TextNode
    return [TextNode(text=t, id_=f"n{i}") for i, t in enumerate(texts)]


def test_bm25_ranks_exact_term_first_and_ignores_unrelated():
    idx = KeywordIndex(_nodes(["analogWrite מייצרת PWM", "digitalRead קורא פין", "רשימה ב־Python"]))
    results = idx.search("מה עושה analogWrite?", 5)
    assert results[0][0].node_id == "n0"
    assert all(node.node_id != "n2" for node, _ in results)   # no shared term -> not returned


def test_bm25_no_match_returns_empty():
    idx = KeywordIndex(_nodes(["Arduino PWM", "STM32 GPIO"]))
    assert idx.search("Scrum Product Owner", 5) == []
    assert KeywordIndex([]).search("anything", 5) == []


# ---------------------------------------------------------------------- RRF
def test_rrf_no_duplicates_and_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "a"]])
    assert len(fused) == len(set(fused)) == 4
    assert fused[0] in ("a", "c")          # found by both lists
    assert fused.index("d") > fused.index("c")


def test_rrf_single_list_keeps_order():
    assert reciprocal_rank_fusion([["x", "y", "z"], []]) == ["x", "y", "z"]


# ------------------------------------------------------- hybrid engine (index)
VOCAB = ["Arduino", "STM32", "Python", "C++", "constructor", "PWM", "list", "קבל"]


def build(dirs, docs):
    kb, _, storage = dirs
    for rel, text in docs.items():
        path = kb / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding(vocab=VOCAB))
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")
    return m


def engines(m):
    return (RagEngine(m.index, llm=FakeLLM()),
            RagEngine(m.index, llm=FakeLLM(), keyword_index=m.keyword_index()))


# The vector model (KeywordEmbedding) only "understands" the words in VOCAB.
# The member-initializer chunk shares no VOCAB words with the question, so the
# vector search ranks it low - like the real failure of question 2.
DOCS = {
    "C++/init_list.txt": "Member Initializer List: מאתחלים שדות לפני גוף הבנאי, שדות const חייבים אתחול כך.",
    "C++/classes.txt": "C++ constructor C++ constructor – מחלקה ובנאי.",
    "C++/destructor.txt": "C++ constructor constructor destructor – הריסת אובייקט.",
    "C++/copy.txt": "C++ constructor constructor copy – העתקת אובייקט.",
    "C++/move.txt": "C++ constructor constructor move – העברת משאבים.",
    "Electronics/caps.txt": "קבל קבל constructor – טעינה ופריקה.",
    "Arduino/pwm.txt": "Arduino PWM analogWrite – פינים 3, 5, 6, 9, 10, 11.",
    "Python/lists.txt": "Python list – רשימה ניתנת לשינוי.",
}


def test_bm25_rescues_chunk_that_vector_search_misses(dirs):
    m = build(dirs, DOCS)
    vector_only, hybrid = engines(m)
    question = "C++ constructor: למה מאתחלים שדות לפני גוף הבנאי?"

    assert "init_list.txt" not in [s.file_name for s in vector_only.retrieve(question)]
    top = hybrid.retrieve(question)
    assert "init_list.txt" in [s.file_name for s in top]
    assert len(top) == config.TOP_K


def test_bm25_without_keyword_overlap_does_not_change_results(dirs):
    """When BM25 finds nothing, the final order is exactly the vector order."""
    m = build(dirs, DOCS)
    _, hybrid = engines(m)
    unrelated = "Scrum"                  # no chunk contains this word
    assert m.keyword_index().search(unrelated, 20) == []

    top = hybrid.retrieve(unrelated)
    assert [s.vector_rank for s in top] == [1, 2, 3]
    assert all(s.keyword_rank is None for s in top)


def test_bm25_agreeing_with_vector_keeps_vector_winner_first(dirs):
    """Both methods point to the same chunk: the result is unchanged."""
    m = build(dirs, DOCS)
    vector_only, hybrid = engines(m)
    question = "Arduino PWM analogWrite"
    assert vector_only.retrieve(question)[0].file_name == "pwm.txt"
    assert hybrid.retrieve(question)[0].file_name == "pwm.txt"


def test_misleading_keyword_does_not_push_out_strong_vector_match(dirs):
    """BM25 favours a wrong chunk, but the vector top-1 stays in the final Top-3."""
    m = build(dirs, DOCS)
    vector_only, hybrid = engines(m)
    question = "Python list טעינה ופריקה"     # vector -> lists.txt, keywords -> caps.txt
    best_vector = vector_only.retrieve(question)[0]
    assert best_vector.file_name == "lists.txt"
    assert best_vector.node_id in [s.node_id for s in hybrid.retrieve(question)]


def test_hybrid_results_are_unique_and_ranked(dirs):
    m = build(dirs, DOCS)
    _, hybrid = engines(m)
    top = hybrid.retrieve("C++ constructor שדות בנאי")
    ids = [s.node_id for s in top]
    assert len(ids) == len(set(ids)) == 3
    assert [s.number for s in top] == [1, 2, 3]
    assert any(s.vector_rank is not None for s in top)
    assert any(s.keyword_rank is not None for s in top)


def test_function_names_and_english_terms(dirs):
    docs = dict(DOCS)
    docs["STM32/gpio.txt"] = "STM32: HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET) מדליקה פין."
    m = build(dirs, docs)
    _, hybrid = engines(m)
    top = hybrid.retrieve("מה עושה HAL_GPIO_WritePin?")
    assert top[0].file_name == "gpio.txt"


def test_hybrid_works_after_reload(dirs):
    m = build(dirs, DOCS)
    _, _, storage = dirs
    reloaded = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding(vocab=VOCAB))
    reloaded.load_or_create()
    assert reloaded.loaded_from_storage
    assert len(reloaded.keyword_index()) == m.num_vectors()
    top = RagEngine(reloaded.index, llm=FakeLLM(),
                    keyword_index=reloaded.keyword_index()).retrieve(
        "C++ constructor: למה מאתחלים שדות לפני גוף הבנאי?")
    assert "init_list.txt" in [s.file_name for s in top]


def test_keyword_index_rebuilt_after_upload(dirs):
    m = build(dirs, DOCS)
    _, uploads, _ = dirs
    before = m.keyword_index()
    assert m.keyword_index() is before                     # cached while nothing changes
    m.add_uploaded_file("servo.txt", "Arduino Servo.h servo.write זווית".encode("utf-8"), uploads)
    after = m.keyword_index()
    assert after is not before and len(after) == len(before) + 1
    top = RagEngine(m.index, llm=FakeLLM(), keyword_index=after).retrieve("servo.write")
    assert top[0].file_name == "servo.txt"


def test_hybrid_can_be_disabled(dirs, monkeypatch):
    m = build(dirs, DOCS)
    vector_only, hybrid = engines(m)
    monkeypatch.setattr(config, "HYBRID_SEARCH", False)
    q = "C++ constructor: למה מאתחלים שדות לפני גוף הבנאי?"
    assert [s.node_id for s in hybrid.retrieve(q)] == [s.node_id for s in vector_only.retrieve(q)]


def test_refusal_and_no_sources_still_work_with_hybrid(dirs):
    m = build(dirs, DOCS)
    llm = FakeLLM(response_text=config.NO_ANSWER_MESSAGE)
    result = RagEngine(m.index, llm=llm, keyword_index=m.keyword_index()).ask("מהי Scrum?")
    assert result.found is False and result.sources == []
    assert len(llm.prompts) == 1                           # still exactly one LLM call


def test_empty_index_with_hybrid_does_not_call_llm(dirs):
    _, _, storage = dirs
    m = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding(vocab=VOCAB))
    m.load_or_create()
    llm = FakeLLM()
    result = RagEngine(m.index, llm=llm, keyword_index=m.keyword_index()).ask("Arduino?")
    assert result.found is False and llm.prompts == []
