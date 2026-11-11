"""Free-form questions, follow-ups and sub-topic summaries - local rules only.

No API calls: plain functions, fake LLM and fake embeddings. The same behavior
against the REAL study material is checked in tests/test_regression_real.py.
"""

from llama_index.core.schema import TextNode

from rag import summarizer, topics
from rag.glossary import build_glossary, find_terms
from rag.hybrid import KeywordIndex, hebrew_stem, tokenize
from rag.query_engine import RagEngine
from tests.conftest import FakeLLM


def node(text, topic="Data-Structures-and-Algorithms", doc="d1", start=0, name="trees.txt"):
    return TextNode(text=text, metadata={"topic": topic, "doc_id": doc, "file_name": name},
                    start_char_idx=start)


# ------------------------------------------------------- Hebrew inflections
def test_hebrew_stem_matches_singular_plural_and_final_letters():
    assert hebrew_stem("סריקות") == hebrew_stem("סריקה") == "סריק"
    assert hebrew_stem("עצים") == hebrew_stem("עץ") == "עצ"
    assert hebrew_stem("פסיקות") == hebrew_stem("פסיקה")


def test_tokenize_inflected_question_matches_the_material():
    question = set(tokenize("הסריקות בעצים"))
    material = set(tokenize("סריקה של עץ"))
    assert {"סריק", "עצ"} <= question & material


def test_conversation_words_are_not_searched():
    # "מבינה" without its "מ" is "בינה" (artificial intelligence) and "מידע" is "ידע"
    tokens = tokenize("אני לא מבינה, יש לך מידע על זה?")
    assert not {"בינה", "ידע", "מבינה", "מידע", "אני", "לך"} & set(tokens)


def test_each_query_word_counts_once():
    """The stem adds matches, not weight: a Hebrew word with several forms
    ("מגדירות" + its stem) scores like an English word with one form - before the
    fix it counted twice and outweighed "analogWrite"."""
    nodes = [node("מגדירות x", doc="a"), node("analogwrite x", doc="b"), node("y z", doc="c")]
    scores = {n.metadata["doc_id"]: s for n, s in KeywordIndex(nodes).search("מגדירות analogWrite", 3)}
    assert abs(scores["a"] - scores["b"]) < 1e-9


# ---------------------------------------------------------------- glossary
MATERIAL = [
    node("2. סריקות עץ (Tree Traversals)\nסריקה לפי רמות (Level Order / BFS) משתמשת בתור.", start=0),
    node("העמסת פונקציות (Overloading) מאפשרת כמה פונקציות עם אותו שם.", topic="C++", doc="c"),
    node("מחלק מתח (Voltage Divider) בנוי משני נגדים.", topic="Electronics", doc="e"),
]


def test_glossary_is_learned_from_the_material():
    pairs = {(t.hebrew, t.english) for t in build_glossary(MATERIAL)}
    assert ("סריקות עץ", "Tree Traversals") in pairs
    assert ("העמסת פונקציות", "Overloading") in pairs           # construct form kept whole
    assert ("פונקציות", "Overloading") not in pairs
    assert ("מחלק מתח", "Voltage Divider") in pairs


def test_glossary_matches_any_inflection_and_knows_the_topic():
    glossary = build_glossary(MATERIAL)
    found = find_terms("אני לא מבינה את עניין הסריקות בעצים", glossary)
    assert [t.english for t in found] == ["Tree Traversals"]
    assert found[0].topics == {"Data-Structures-and-Algorithms"}
    assert [t.english for t in find_terms("מה זה מחלק מתח?", glossary)] == ["Voltage Divider"]
    assert find_terms("מה זה חציון?", glossary) == []


# ---------------------------------------------------- free-form questions
def test_request_phrasing_is_removed_from_the_search_only():
    assert topics.strip_request_phrases("אני לא מבינה את עניין הסריקות בעצים") == "הסריקות בעצים"
    assert topics.strip_request_phrases("הסבר לי את רשימת האתחול") == "רשימת האתחול"
    assert topics.strip_request_phrases("תוכלי להסביר לי מה זה בנאי?") == "מה זה בנאי"
    assert topics.strip_request_phrases("מה ההבדל בין / ל־// ב־Python?") == "מה ההבדל בין / ל־// ב־Python?"


def test_retrieval_query_adds_the_materials_terms_and_topic():
    glossary = build_glossary(MATERIAL)
    query = topics.retrieval_query("אני לא מבינה את עניין הסריקות בעצים", glossary=glossary)
    assert "Tree Traversals" in query and "מבינה" not in query
    assert topics.infer_topic("הסריקות בעצים", glossary) == "Data-Structures-and-Algorithms"
    assert topics.infer_topic("מה זה חציון?", glossary) is None


def test_explanation_question_finds_the_section_with_other_words():
    nodes = MATERIAL + [node("בינה מלאכותית ומודלי שפה גדולים - מידע כללי", topic=None, doc="llm")]
    index = KeywordIndex(nodes)
    query = topics.retrieval_query("אני לא מבינה את עניין הסריקות בעצים", glossary=index.glossary)
    assert index.search(query, 1)[0][0].get_content().startswith("2. סריקות עץ")


# --------------------------------------------------------------- follow-ups
def test_follow_up_detection():
    assert topics.is_follow_up("ומה לגבי ירושה?")
    assert topics.is_follow_up("אבל יש לך מידע על עצים?")
    assert topics.is_follow_up("תסבירי את זה שוב")
    assert not topics.is_follow_up("מהו analogWrite?")
    assert not topics.is_follow_up("מה זה בנאי?")


def test_follow_up_without_content_searches_the_previous_subject():
    previous = "אני לא מבינה את עניין הסריקות בעצים"
    query = topics.retrieval_query("תסבירי את זה שוב, במילים פשוטות", previous=previous)
    assert "הסריקות בעצים" in query


def test_follow_up_on_the_same_subject_keeps_it():
    previous = "אני לא מבינה את עניין הסריקות בעצים"
    assert "הסריקות" in topics.retrieval_query("אבל יש לך מידע על עצים באלגוריתמים", previous=previous)


def test_follow_up_on_a_new_subject_keeps_only_the_topic():
    query = topics.retrieval_query("ומה לגבי מחסנית?", "Data-Structures-and-Algorithms",
                                   previous="מה זה רשימה מקושרת?")
    assert "מקושרת" not in query and "מבני נתונים" in query


def test_previous_question_is_given_to_the_model_as_context(dirs):
    from rag.index_manager import IndexManager
    from tests.conftest import KeywordEmbedding

    kb, _, storage = dirs
    (kb / "stats.txt").write_text("חציון הוא הערך האמצעי. ממוצע הוא סכום חלקי מספר.", encoding="utf-8")
    manager = IndexManager(storage_dir=storage, embed_model=KeywordEmbedding())
    manager.load_or_create()
    manager.sync_folder(kb, "knowledge_base")
    llm = FakeLLM(response_text="תשובה [1].")
    engine = RagEngine(manager.index, llm=llm)
    engine.ask("ומה לגבי ממוצע?", previous_question="מה זה חציון?")
    assert "השאלה הקודמת בשיחה (רק להבנת ההקשר): מה זה חציון?" in llm.prompts[-1]
    engine.ask("מה זה חציון?")
    assert "השאלה הקודמת" not in llm.prompts[-1]


# --------------------------------------------------- sub-topic summaries
def test_summary_of_a_sub_topic_is_not_a_whole_topic_summary():
    decision = topics.route("אפשר סיכום של סריקות עצים באלגוריתמים")
    assert decision == {"kind": "summary", "scope": "subtopic", "subject": "סריקות עצים",
                        "topic": "Data-Structures-and-Algorithms"}
    # no topic named: no "choose a topic" question - inferred later from the material
    assert topics.route("סכמי לי את סריקות העצים")["scope"] == "subtopic"


def test_whole_topic_summaries_are_unchanged():
    assert topics.route("הסכם לי את החומר בפייתון בצורה טובה ומקצועית") == \
        {"kind": "summary", "scope": "topic", "topic": "Python"}
    assert topics.route("תסכם לי בבקשה את כל החומר של מבני נתונים")["scope"] == "topic"
    assert topics.route("סכם לי בבקשה") == {"kind": "choose_topic", "candidates": []}
    assert topics.route("summary of Arduino")["topic"] == "Arduino"


class SubtopicManager:
    """Stand-in for IndexManager: a manifest and a keyword index over real files."""

    def __init__(self, tmp_path, texts):
        from rag.document_loader import compute_file_hash

        self.manifest, nodes = {}, []
        for i, (name, topic, chunks) in enumerate(texts):
            path = tmp_path / name
            path.write_text("\n".join(chunks), encoding="utf-8")
            file_hash = compute_file_hash(path)
            self.manifest[file_hash] = {"file_name": name, "path": str(path), "topic": topic}
            start = 0
            for chunk in chunks:
                nodes.append(TextNode(text=chunk, start_char_idx=start,
                                      metadata={"topic": topic, "doc_id": file_hash, "file_name": name}))
                start += len(chunk) + 1
        self._index = KeywordIndex(nodes)

    def keyword_index(self):
        return self._index


TREES = [
    "עצים בינאריים - מבוא. לכל צומת יש עד שני ילדים.",
    "2. סריקות עץ (Tree Traversals)\ndef inorder(node): ...",
    "def postorder(node): ...\nסריקה לפי רמות (Level Order) משתמשת בתור.",
    "3. עץ חיפוש בינארי (BST) - הכנסה וחיפוש.",
]


def test_sub_topic_plan_uses_only_the_relevant_chunks(tmp_path):
    manager = SubtopicManager(tmp_path, [
        ("trees.txt", "Data-Structures-and-Algorithms", TREES),
        ("sort.txt", "Data-Structures-and-Algorithms", ["מיון מהיר (Quick Sort) ומיון מיזוג."]),
        ("cpp.txt", "C++", ["מחלקות ובנאים ב־C++."]),
    ])
    plan = summarizer.plan_subtopic_summary(manager, "סריקות עצים")
    assert plan.usable and plan.scope == "subtopic" and plan.calls == 1
    assert plan.topic == "Data-Structures-and-Algorithms"             # inferred, not asked
    assert [s.file_name for s in plan.sources] == ["trees.txt"]
    text = plan.sources[0].text
    assert "Tree Traversals" in text and "Level Order" in text          # the section + its continuation
    assert "מיון" not in text and "מחלקות" not in text


def test_sub_topic_not_in_the_material_is_reported_not_invented(tmp_path):
    manager = SubtopicManager(tmp_path, [("trees.txt", "Data-Structures-and-Algorithms", TREES)])
    plan = summarizer.plan_subtopic_summary(manager, "רשתות נוירונים קוונטיות")
    assert not plan.usable and plan.calls == 0
    assert plan.skipped == ['לא נמצא במאגר הידע חומר על "רשתות נוירונים קוונטיות".']


def test_sub_topic_summary_prompt_names_the_sub_topic(tmp_path):
    manager = SubtopicManager(tmp_path, [("trees.txt", "Data-Structures-and-Algorithms", TREES)])
    plan = summarizer.plan_subtopic_summary(manager, "סריקות עצים", "Data-Structures-and-Algorithms")
    llm = FakeLLM(response_text="# סיכום\nסריקות עץ [1].")
    result = summarizer.run_summary(plan, make_llm=lambda max_tokens: llm)
    assert len(llm.prompts) == 1 and result.found
    assert 'הסיכום הוא של הנושא "סריקות עצים"' in llm.prompts[0]
    assert "Tree Traversals" in llm.prompts[0]
