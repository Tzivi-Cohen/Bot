"""The real-API check (tests/manual_api_check.py) - checked locally, no API call.

The assignment asks for at least 5 questions, including one whose answer is
not in the material, with the retrieved sources. These tests make sure the
question set stays complete and that the script runs the same pipeline as the app.
"""

import json
from pathlib import Path

import config

TESTS = Path(__file__).resolve().parent
SPEC = json.loads((TESTS / "api_questions.json").read_text(encoding="utf-8"))
QUESTIONS = SPEC["questions"]


def test_at_least_five_questions_all_with_text():
    assert len([q for q in QUESTIONS if q["question"].strip()]) >= 5


def test_one_question_is_not_in_the_material():
    no_answer = [q for q in QUESTIONS if q["expect_no_answer"]]
    assert no_answer and all(q["expected_answer"] == config.NO_ANSWER_MESSAGE for q in no_answer)


def test_answerable_questions_name_an_existing_source_file():
    names = {p.name for folder in (config.KNOWLEDGE_BASE_DIR, config.UPLOADS_DIR) if folder.exists()
             for p in folder.rglob("*") if p.is_file()}
    for q in QUESTIONS:
        if not q["expect_no_answer"]:
            assert q["expected_source"], q["id"]
            if q["expected_source"] != "LLM.txt" or (config.UPLOADS_DIR / "LLM.txt").exists():
                assert q["expected_source"] in names, q["expected_source"]


def test_script_uses_the_apps_search_text():
    """The paid check must test the system as it is: same search text as app.py."""
    source = (TESTS / "manual_api_check.py").read_text(encoding="utf-8")
    assert "topics.retrieval_query(" in source and "search_query=search_text" in source
    assert "glossary=keyword_index.glossary" in source
