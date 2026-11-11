"""Regression tests against the REAL knowledge_base files - read-only, no API.

They do not open ChromaDB or change the index: the library status comes from
storage/manifest.json, and the keyword search (the BM25 half of the hybrid
search) is built in memory from the real documents.
"""

import json
from pathlib import Path

import pytest

import config
from rag.document_loader import (
    compute_file_hash,
    extract_documents,
    iter_supported_files,
    relative_path_for,
    split_documents,
    topic_for,
)
from rag.hybrid import KeywordIndex
from rag.topics import retrieval_query
from ui import library as lib

KB = config.KNOWLEDGE_BASE_DIR
MANIFEST = config.STORAGE_DIR / config.MANIFEST_FILENAME
pytestmark = pytest.mark.skipif(not (KB / "Python").is_dir(), reason="real knowledge_base not present")


class ManifestOnly:
    """Read-only stand-in for IndexManager (no ChromaDB)."""

    def __init__(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}

    def find_by_hash(self, file_hash):
        return self.manifest.get(file_hash)

    def entry_for_path(self, path):
        key = str(Path(path).resolve())
        return next((e for e in self.manifest.values() if e.get("path") == key), None)


@pytest.fixture(scope="module")
def catalog():
    return lib.build_catalog(ManifestOnly(), KB, config.UPLOADS_DIR, lambda p, h: None)


def read(path):
    return "".join(d.text for d in extract_documents(path))


def names(results):
    return [item.name for item, _, _ in results]


TOPIC_FILES = {
    "Python": {"python_basics_variables_conditions_loops.txt", "python_files_and_exceptions.txt",
               "python_functions_and_classes.txt"},
    "C++": {"cpp_classes_constructors_destructors.txt", "cpp_inheritance_and_polymorphism.txt",
            "cpp_references_and_resource_management.txt"},
    "Arduino": {"arduino_analog_sensors_pwm.txt", "arduino_digital_io_buttons_leds.txt",
                "arduino_serial_communication.txt"},
    "STM32": {"stm32_gpio_and_interrupts.txt", "stm32_timers_and_adc.txt", "stm32_uart_spi_i2c.txt"},
}


# ------------------------------------------------------------- library search
@pytest.mark.parametrize("query, topic", [
    ("פייתון", "Python"), ("Python", "Python"), ("בפייתון", "Python"), ("python", "Python"),
    ("C++", "C++"), ("סי פלוס פלוס", "C++"),
    ("Arduino", "Arduino"), ("ארדואינו", "Arduino"), ("STM32", "STM32"),
])
def test_topic_search_returns_exactly_the_topic_documents(catalog, query, topic):
    results = lib.search_files(catalog, query, read)
    assert set(names(results)) == TOPIC_FILES[topic]
    assert all(reason.startswith("בנושא") for _, _, reason in results)


@pytest.mark.parametrize("query, must_include", [
    ("מחלקות", {"cpp_classes_constructors_destructors.txt", "python_functions_and_classes.txt"}),
    ("ירושה", {"cpp_inheritance_and_polymorphism.txt"}),
    ("analogWrite", {"arduino_analog_sensors_pwm.txt"}),
    ("חוק אוהם", {"voltage_current_resistance_ohms_law.txt"}),
])
def test_concept_search_finds_relevant_documents(catalog, query, must_include):
    found = set(names(lib.search_files(catalog, query, read)))
    assert must_include <= found


def test_topic_plus_word_searches_only_inside_the_topic(catalog):
    found = names(lib.search_files(catalog, "לולאות בפייתון", read))
    assert found and set(found) <= TOPIC_FILES["Python"]
    assert "python_basics_variables_conditions_loops.txt" in found


@pytest.mark.parametrize("query, expected_first", [
    ("cpp_classes_constructors_destructors.txt", "cpp_classes_constructors_destructors.txt"),
    ("python_basics", "python_basics_variables_conditions_loops.txt"),
    ("stm32_uart", "stm32_uart_spi_i2c.txt"),
])
def test_file_name_search(catalog, query, expected_first):
    results = lib.search_files(catalog, query, read)
    assert names(results)[0] == expected_first and results[0][2] == "בשם הקובץ"


def test_unrelated_query_returns_nothing(catalog):
    assert lib.search_files(catalog, "Scrum Product Owner", read) == []


# --------------------------------------------------- bot retrieval (BM25 half)
@pytest.fixture(scope="module")
def keyword_index():
    nodes = []
    for path in iter_supported_files(KB):
        docs = extract_documents(path, topic=topic_for(path, KB),
                                 relative_path=relative_path_for(path, KB),
                                 file_hash=compute_file_hash(path))
        nodes += split_documents(docs)
    return KeywordIndex(nodes)


@pytest.mark.parametrize("question, context_topic, answer_text", [
    ("מהי מחלקה ב־C++?", None, "מחלקה מגדירה טיפוס חדש"),
    ("מהו בנאי?", None, "Constructor הוא פונקציה עם שם המחלקה"),
    ("מה ההבדל בין public ל־private?", None, "private – נגיש רק מתוך המחלקה"),
    ("הסבר לי לולאות בפייתון", None, "לולאת for עוברת על איברים"),
    ("מה עושה analogWrite בארדואינו?", None, "analogWrite(pin, value) מייצרת אות PWM"),
    ("מהו חוק אוהם?", None, "V = I × R"),
    ("איך עובדת פסיקה ב־STM32?", None, "EXTI (External Interrupt/Event Controller)"),
    ("מה זה מצביע בשפת C?", None, "מצביע הוא משתנה שמכיל כתובת"),
])
def test_answer_chunk_is_retrieved_in_top3(keyword_index, question, context_topic, answer_text):
    hits = keyword_index.search(retrieval_query(question, context_topic), 3)
    assert any(answer_text in node.get_content() for node, _ in hits), \
        [n.metadata["file_name"] for n, _ in hits]


def test_follow_up_question_uses_previous_topic(keyword_index):
    """"תן לי דוגמה של לולאה" after a Python question: a Python loop example is found."""
    import re

    hits = keyword_index.search(retrieval_query("תן לי דוגמה של לולאה", "Python"), 3)
    assert any(node.metadata.get("topic") == "Python"
               and re.search(r"^\s*(for|while) .*:", node.get_content(), re.M)
               for node, _ in hits)


# ----------------------------------------------------------- document display
def test_real_document_display_structure():
    text = (KB / "Data-Structures-and-Algorithms" / "binary_trees.txt").read_text(encoding="utf-8")
    blocks = lib.parse_study_text(text, "python")
    kinds = {b[0] for b in blocks}
    assert {"title", "para", "bullets", "numbered", "code"} <= kinds
    numbered = next(b[1] for b in blocks if b[0] == "numbered")
    assert numbered[0].startswith("הצומת הוא עלה")                    # "1) ..." became a list item
    questions = [b[1][0] for b in blocks if b[0] == "para" and b[1][0].startswith("שאלה")]
    solutions = [b[1][0] for b in blocks if b[0] == "para" and b[1][0].startswith("פתרון")]
    assert len(questions) == len(solutions) == 4                     # each in its own paragraph


def test_direction_marks_removed_for_display_only():
    path = KB / "C++" / "cpp_references_and_resource_management.txt"
    raw = path.read_text(encoding="utf-8")
    assert "‏" in raw                                            # file itself unchanged
    for block in lib.parse_study_text(raw, "cpp"):
        assert "‏" not in str(block[1])
    assert "‏" not in lib.bidi_html(raw)


def test_code_blocks_keep_their_language():
    text = (KB / "C++" / "cpp_classes_constructors_destructors.txt").read_text(encoding="utf-8")
    codes = [b for b in lib.parse_study_text(text, "cpp") if b[0] == "code"]
    assert codes and all(b[2] == "cpp" for b in codes)
    py = (KB / "Python" / "python_basics_variables_conditions_loops.txt").read_text(encoding="utf-8")
    assert all(b[2] == "python" for b in lib.parse_study_text(py, "python") if b[0] == "code")


# ------------------------------------- free-form questions (real material)
# The questions that got "לא מצאתי מידע" although the material exists, and more
# concepts phrased in other words than the documents. (question, previous
# question, file of the answer, text that must be in a top-3 chunk)
FREE_QUESTIONS = [
    ("אני לא מבינה את עניין הסריקות בעצים", None, "binary_trees.txt", "סריקות עץ (Tree Traversals)"),
    ("אבל יש לך מידע על עצים באלגוריתמים", "אני לא מבינה את עניין הסריקות בעצים",
     "binary_trees.txt", "סריקות עץ (Tree Traversals)"),
    ("תסבירי את זה שוב, במילים פשוטות", "אני לא מבינה את עניין הסריקות בעצים",
     "binary_trees.txt", "סריקות עץ (Tree Traversals)"),
    ("הסבירי לי את רשימת האתחול בבנאי", None, "cpp_classes_constructors_destructors.txt", "Initializer"),
    ("אני לא מבינה מה זה מחלק מתח", None, "voltage_current_resistance_ohms_law.txt", "מחלק מתח"),
    ("תוכלי להסביר לי מה זה קטע קריטי?", None, "memory_and_interrupts_in_embedded.txt", "Critical Section"),
    ("מה זה ערימת מינימום?", None, "binary_trees.txt", "Min-Heap"),
    ("אני לא מבינה את עניין המיון היציב", None, "sorting_and_searching.txt", "יציב"),
    ("ומה לגבי ירושה?", "מה זה בנאי ב־C++?", "cpp_inheritance_and_polymorphism.txt", None),
]


@pytest.mark.parametrize("question, previous, answer_file, answer_text", FREE_QUESTIONS)
def test_free_form_question_finds_the_material(keyword_index, question, previous, answer_file, answer_text):
    from rag.topics import infer_topic

    glossary = keyword_index.glossary
    context = infer_topic(previous, glossary) if previous else None
    query = retrieval_query(question, context, glossary=glossary, previous=previous)
    hits = keyword_index.search(query, 3)
    assert any(node.metadata["file_name"] == answer_file
               and (answer_text is None or answer_text in node.get_content())
               for node, _ in hits), (query, [n.metadata["file_name"] for n, _ in hits])


def test_glossary_from_real_material(keyword_index):
    pairs = {(t.hebrew, t.english) for t in keyword_index.glossary}
    assert {("סריקות עץ", "Tree Traversals"), ("מחלק מתח", "Voltage Divider"),
            ("קטע קריטי", "Critical Section"), ("העמסת פונקציות", "Overloading")} <= pairs
    assert len(pairs) > 150                                    # learned, not a hand-made list


class RealManifest(ManifestOnly):
    def __init__(self, keyword_index):
        super().__init__()
        self._keywords = keyword_index

    def keyword_index(self):
        return self._keywords


def test_sub_topic_summary_of_tree_traversals(keyword_index):
    from rag import summarizer
    from rag.topics import route

    decision = route("אפשר סיכום של סריקות עצים באלגוריתמים")
    assert decision["scope"] == "subtopic" and decision["subject"] == "סריקות עצים"
    plan = summarizer.plan_subtopic_summary(RealManifest(keyword_index), decision["subject"],
                                            decision["topic"])
    assert plan.usable and plan.calls == 1 and plan.topic == "Data-Structures-and-Algorithms"
    assert [s.file_name for s in plan.sources] == ["binary_trees.txt"]
    text = plan.sources[0].text
    assert "סריקות עץ (Tree Traversals)" in text and "level_order" in text   # section + its code
    assert plan.sources[0].tokens < 4000                       # part of the file, not all of it


def test_sub_topic_not_in_real_material(keyword_index):
    from rag import summarizer

    plan = summarizer.plan_subtopic_summary(RealManifest(keyword_index), "רשתות נוירונים קוונטיות")
    assert not plan.usable and plan.calls == 0 and plan.skipped


# ------------------------- every topic, loose Hebrew, English terms, follow-ups
IRQ_FILES = {"memory_and_interrupts_in_embedded.txt", "stm32_gpio_and_interrupts.txt"}
ALL_TOPICS = [
    # the reported question ("איך עובד פסיקות?" - got "לא מצאתי") and its variants
    ("איך עובד פסיקות?", None, IRQ_FILES),
    ("איך עובדות פסיקות?", None, IRQ_FILES),
    ("מה זה פסיקה?", None, IRQ_FILES),
    ("תסביר לי על Interrupts", None, IRQ_FILES),
    ("מה התפקיד של NVIC?", None, IRQ_FILES),
    ("איך מגדירים טיימר בSTM32", None, {"stm32_timers_and_adc.txt"}),
    ("איך עושים לולאה בפייתון", None, {"python_basics_variables_conditions_loops.txt"}),
    ("מה זה חריגה בפייתון", None, {"python_files_and_exceptions.txt"}),
    ("מה זה המצביעים בשפת סי", None, {"c_pointers_and_arrays.txt"}),
    ("איך מקצים זיכרון דינמי ב C", None, {"c_memory_management.txt"}),
    ("איך עובד ירושה", None, {"cpp_inheritance_and_polymorphism.txt"}),
    ("מה זה רפרנס בסי פלוס פלוס", None, {"cpp_references_and_resource_management.txt"}),
    ("איך עובד מחסנית", None, {"linked_lists_stacks_queues.txt"}),
    ("איך עובד מיון מהיר", None, {"sorting_and_searching.txt"}),
    ("איך קוראים חיישן אנלוגי בארדואינו", None, {"arduino_analog_sensors_pwm.txt"}),
    ("איך שולחים נתונים בסריאל", None, {"arduino_serial_communication.txt", "stm32_uart_spi_i2c.txt"}),
    ("מה זה חוק אוהם", None, {"voltage_current_resistance_ohms_law.txt"}),
    ("איך עובד טרנזיסטור", None, {"diodes_and_transistors.txt"}),
    ("מה התפקיד של קבל", None, {"resistors_and_capacitors.txt"}),
    ("למה צריך volatile", None, {"embedded_c_essentials.txt"}),
    ("מה זה מערכת זמן אמת", None, {"communication_and_real_time_systems.txt"}),
    ("תן לי דוגמה", "איך עובד פסיקות?", IRQ_FILES),
    ("ומה לגבי SPI?", "מה זה UART ב STM32", {"stm32_uart_spi_i2c.txt"}),
    ("ובפייתון?", "מה זה לולאה בשפת C", {"python_basics_variables_conditions_loops.txt"}),
    ("איך עובד פסיקות?", "מה זה רשימה בפייתון?", IRQ_FILES),     # new subject: no "Python" leak
]


@pytest.mark.parametrize("question, previous, files", ALL_TOPICS)
def test_every_topic_is_found_like_in_the_app(keyword_index, question, previous, files):
    """Same steps as app.py: the previous topic is used only for a follow-up."""
    from rag import topics

    glossary = keyword_index.glossary
    follow = bool(previous) and (topics.is_follow_up(question)
                                 or topics.continues_previous(question, previous))
    context = topics.infer_topic(previous, glossary, keyword_index) if follow else None
    query = topics.retrieval_query(question, context, glossary=glossary, previous=previous)
    found = [node.metadata["file_name"] for node, _ in keyword_index.search(query, 3)]
    assert set(found) & files, (query, found)
    if previous and not follow:
        assert "Python" not in query


def test_topic_is_recognized_without_its_name(keyword_index):
    from rag.topics import infer_topic

    g = keyword_index.glossary
    assert infer_topic("איך עובדות פסיקות?", g, keyword_index) in {"Embedded-Systems", "STM32"}
    assert infer_topic("מה התפקיד של NVIC?", g, keyword_index) in {"Embedded-Systems", "STM32"}
    assert infer_topic("איך עובד טרנזיסטור", g, keyword_index) == "Electronics"
    assert infer_topic("תן לי דוגמה", g, keyword_index) is None        # no subject of its own


def test_glossary_keeps_real_pairs_and_drops_chance_ones(keyword_index):
    pairs = {(t.hebrew, t.english) for t in keyword_index.glossary}
    assert ("פסיקה", "Interrupt") in pairs and ("פסיקות", "Interrupts") in pairs
    assert ("נתונים", "Attributes") not in pairs                     # one chance phrase
