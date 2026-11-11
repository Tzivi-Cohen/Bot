"""Topic / file summaries (rag/summarizer.py) - mocked LLM, temp folders, no API."""

import pytest

import config
from rag import summarizer as sm
from rag.index_manager import IndexManager
from tests.conftest import AuthenticationError, CountingEmbedding, FakeLLM

PY1 = "Python – יסודות\n=====\nמשתנים ולולאות.\n\n    for i in range(3):\n        print(i)\n"
PY2 = "Python – פונקציות\n=====\nהגדרת פונקציה עם def.\n\n    def add(a, b):\n        return a + b\n"
ARD = "Arduino – PWM\n=====\nanalogWrite מייצרת PWM.\n\n    analogWrite(9, 128);\n"


@pytest.fixture
def indexed(dirs):
    kb, uploads, storage = dirs
    for rel, text in {"Python/b_functions.txt": PY2, "Python/a_basics.txt": PY1,
                      "Arduino/pwm.txt": ARD}.items():
        (kb / rel).parent.mkdir(exist_ok=True)
        (kb / rel).write_text(text, encoding="utf-8")
    m = IndexManager(storage_dir=storage, embed_model=CountingEmbedding(embed_dim=8))
    m.load_or_create()
    m.sync_folder(kb, "knowledge_base")
    return m, kb, uploads


class RecordingLLMFactory:
    """make_llm replacement: records max_tokens of every model and every prompt."""

    def __init__(self, responses, error=None):
        self.responses, self.error = list(responses), error
        self.prompts, self.max_tokens = [], []

    def __call__(self, max_tokens):
        self.max_tokens.append(max_tokens)
        factory = self

        class _LLM:
            def complete(self, prompt):
                factory.prompts.append(prompt)
                if factory.error:
                    raise factory.error
                return factory.responses.pop(0)
        return _LLM()


# ------------------------------------------------------------------- planning
def test_topic_plan_uses_all_topic_files_in_order_and_no_other_topic(indexed):
    m, _, _ = indexed
    plan = sm.plan_topic_summary(m, "Python")
    assert [s.file_name for s in plan.sources] == ["a_basics.txt", "b_functions.txt"]
    assert [s.number for s in plan.sources] == [1, 2]
    assert all(s.topic == "Python" for s in plan.sources)
    assert plan.mode == "single" and plan.calls == 1
    assert plan.input_tokens > 0 and plan.max_output_tokens == sm.FINAL_MAX_OUTPUT_TOKENS
    assert plan.estimated_cost == pytest.approx(
        sm.estimate_cost(plan.input_tokens, plan.max_output_tokens))


def test_file_plan_uses_only_that_file(indexed):
    m, kb, _ = indexed
    plan = sm.plan_file_summary(m, kb / "Arduino" / "pwm.txt")
    assert [s.file_name for s in plan.sources] == ["pwm.txt"] and plan.calls == 1


def test_unknown_topic_and_unindexed_file_make_no_calls(indexed):
    m, kb, _ = indexed
    assert sm.plan_topic_summary(m, "STM32").calls == 0
    (kb / "Python" / "new.txt").write_text("חדש", encoding="utf-8")       # not indexed
    plan = sm.plan_file_summary(m, kb / "Python" / "new.txt")
    assert plan.calls == 0 and not plan.usable and "אינו נמצא במאגר הידע" in plan.skipped[0]


def test_modified_file_is_left_out_of_topic_plan(indexed):
    m, kb, _ = indexed
    (kb / "Python" / "a_basics.txt").write_text("גרסה חדשה שלא אונדקסה", encoding="utf-8")
    plan = sm.plan_topic_summary(m, "Python")
    assert [s.file_name for s in plan.sources] == ["b_functions.txt"]
    assert "השתנה" in plan.skipped[0]


def test_large_material_is_planned_in_stages(indexed, monkeypatch):
    m, _, _ = indexed
    monkeypatch.setattr(sm, "MAX_SINGLE_INPUT_TOKENS", 10)
    plan = sm.plan_topic_summary(m, "Python")
    assert plan.mode == "staged" and plan.calls == 3           # 2 files + 1 combine
    assert plan.max_output_tokens == 2 * sm.STAGE_MAX_OUTPUT_TOKENS + sm.FINAL_MAX_OUTPUT_TOKENS


# -------------------------------------------------------------------- running
def test_single_summary_one_call_with_all_material_and_rules(indexed):
    m, _, _ = indexed
    plan = sm.plan_topic_summary(m, "Python")
    llm = RecordingLLMFactory(["# Python\n## לולאות\nלולאת for [1]\n```python\nfor i in range(3):\n"
                               "    print(i)\n```\n## פונקציות\ndef [2]"])
    result = sm.run_summary(plan, make_llm=llm)
    assert result.calls_made == 1 and len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "for i in range(3):" in prompt and "def add(a, b):" in prompt     # all material
    assert "analogWrite" not in prompt                                       # no other topic
    assert "השתמש רק במידע שמופיע בחומר" in prompt and "מילה במילה" in prompt
    assert prompt.index("a_basics.txt") < prompt.index("b_functions.txt")    # original order
    assert result.found and result.cited == [1, 2] and result.warnings == []
    assert llm.max_tokens == [sm.FINAL_MAX_OUTPUT_TOKENS]


def test_staged_summary_makes_exactly_planned_calls(indexed, monkeypatch):
    m, _, _ = indexed
    monkeypatch.setattr(sm, "MAX_SINGLE_INPUT_TOKENS", 10)
    plan = sm.plan_topic_summary(m, "Python")
    llm = RecordingLLMFactory(["חלק 1", "חלק 2", "# Python\nסיכום [1] [2]"])
    result = sm.run_summary(plan, make_llm=llm)
    assert result.calls_made == plan.calls == len(llm.prompts) == 3
    assert "חלק 1" in llm.prompts[2] and "חלק 2" in llm.prompts[2]


def test_failure_stops_without_more_calls(indexed, monkeypatch):
    m, _, _ = indexed
    monkeypatch.setattr(sm, "MAX_SINGLE_INPUT_TOKENS", 10)
    plan = sm.plan_topic_summary(m, "Python")
    llm = RecordingLLMFactory([], error=AuthenticationError("401"))
    with pytest.raises(sm.SummaryError, match="קריאה 1 מתוך 3"):
        sm.run_summary(plan, make_llm=llm)
    assert len(llm.prompts) == 1                                # stopped at the first failure


def test_no_answer_summary(indexed):
    m, _, _ = indexed
    plan = sm.plan_topic_summary(m, "Python")
    result = sm.run_summary(plan, make_llm=RecordingLLMFactory([config.NO_ANSWER_MESSAGE]))
    assert not result.found and result.text == config.NO_ANSWER_MESSAGE


def test_empty_plan_makes_no_call(indexed):
    m, _, _ = indexed
    llm = RecordingLLMFactory([])
    result = sm.run_summary(sm.plan_topic_summary(m, "STM32"), make_llm=llm)
    assert llm.prompts == [] and result.calls_made == 0 and not result.found


# ----------------------------------------------------------------- verification
def test_invented_code_is_removed_and_real_code_kept():
    material = "    for i in range(3):\n        print(i)\n"
    text = ("## א\n```python\nfor i in range(3):\n    print(i)\n```\n"
            "## ב\n```python\nimport os\nos.system('x')\n```\nסוף [1]")
    clean, cited, warnings = sm.verify_summary(text, material, 1)
    assert "for i in range(3):" in clean and "import os" not in clean
    assert cited == [1] and any("דוגמת קוד" in w for w in warnings)


def test_citation_to_missing_source_is_removed():
    clean, cited, warnings = sm.verify_summary("טקסט [1] ועוד [7]", "חומר", 2)
    assert "[7]" not in clean and "[1]" in clean and cited == [1]
    assert any("[7]" in w for w in warnings)


def test_run_summary_reports_api_usage(indexed):
    m, _, _ = indexed

    class Response:
        raw = {"usage": {"prompt_tokens": 1234, "completion_tokens": 321}}

        def __str__(self):
            return "# סיכום [1]"

    plan = sm.plan_file_summary(m, m.manifest[next(iter(m.manifest))]["path"])
    result = sm.run_summary(plan, make_llm=RecordingLLMFactory([Response()]))
    assert (result.input_tokens, result.output_tokens) == (1234, 321)


def test_summary_llm_has_no_retries_and_same_model():
    llm = sm.create_summary_llm(1800)            # constructing makes no API call
    assert llm.model == config.LLM_MODEL and llm.max_retries == 0 and llm.max_tokens == 1800


def test_fake_llm_compatible(indexed):
    """The project's FakeLLM (used by UI tests) works as a summary model."""
    m, _, _ = indexed
    fake = FakeLLM(response_text="# סיכום\nתוכן [1]")
    result = sm.run_summary(sm.plan_topic_summary(m, "Arduino"), make_llm=lambda n: fake)
    assert result.found and len(fake.prompts) == 1
