"""Summaries of a whole topic folder, of one file, or of a sub-topic
("סריקות עצים") - separate from Q&A.

Regular questions keep using the hybrid Top-3 retrieval (rag/query_engine.py).
A summary needs ALL the material of the topic/file, in its original order,
so it reads the indexed documents directly - no retrieval, no embeddings.
A sub-topic summary uses only the chunks about the sub-topic, found by the
local keyword search (BM25 + the material's glossary) - no embeddings either.

Flow (cost-controlled):
1. plan_*_summary()  - local only: collects the documents, counts tokens and
                       estimates the cost and the exact number of LLM calls.
2. The UI shows the plan; the user must approve it explicitly.
3. run_summary(plan) - performs exactly plan.calls LLM calls (no retries).
4. verify_summary()  - local check: code blocks must exist in the material,
                       citations must point to real sources.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.utils import get_tokenizer

import config
from rag.document_loader import DocumentLoadError, compute_file_hash, extract_documents
from rag.errors import friendly_api_error

MAX_SINGLE_INPUT_TOKENS = 20_000     # above this the summary is made in stages
FINAL_MAX_OUTPUT_TOKENS = 1_800
STAGE_MAX_OUTPUT_TOKENS = 700
TOKEN_SAFETY_MARGIN = 1.15           # tokenizer differs slightly from gpt-4o-mini's

SUMMARY_PROMPT = """אתה עוזר לימודי. כתוב סיכום לימודי מקצועי, ברור ומאורגן בעברית של החומר שלהלן בלבד.
כללים:
1. השתמש רק במידע שמופיע בחומר. אל תוסיף ידע כללי, עובדות, דוגמאות או מקורות שאינם בחומר.
2. מבנה: כותרת ראשית אחת (#), ואחריה כותרות משנה (##) לפי נושאי החומר ובסדר שבו הם מופיעים.
3. בכל נושא: הסבר קצר וברור, ונקודות מפתח ברשימה.
4. דוגמאות קוד: העתק רק דוגמאות שמופיעות בחומר, מילה במילה, בתוך בלוק קוד עם שם השפה. אל תכתוב קוד חדש ואל תשנה קוד קיים.
5. בסוף כל נושא ציין את מספר המקור שעליו הוא מבוסס, למשל [1].
6. אל תזכיר נושאים שאינם מופיעים בחומר.
7. אם אין בחומר מידע לסיכום, השב במדויק: "{no_answer}"

{scope_line}

החומר:
{material}

הסיכום:"""

STAGE_PROMPT = """סכם בעברית את חלק החומר הבא ({part}), לצורך שילוב בסיכום כולל.
כללים: השתמש רק במידע שבחומר; כתוב נקודות מפתח קצרות לפי סדר החומר; העתק מילה במילה עד שתי דוגמאות קוד חשובות בתוך בלוק קוד; אל תוסיף ידע חיצוני.

החומר:
[{number}] (קובץ: {file_name})
{text}

סיכום החלק:"""

COMBINE_PROMPT = """אתה עוזר לימודי. לפניך סיכומים חלקיים של חומר לימוד. כתוב מהם סיכום לימודי אחד, מקצועי ומאורגן בעברית.
כללים:
1. השתמש רק במידע שבסיכומים החלקיים. אל תוסיף ידע חיצוני.
2. מבנה: כותרת ראשית (#), כותרות משנה (##) לפי סדר החומר, הסבר קצר ונקודות מפתח.
3. דוגמאות קוד: רק כאלה שמופיעות בסיכומים החלקיים, מילה במילה, בתוך בלוק קוד.
4. בסוף כל נושא ציין את מספר המקור, למשל [1].

{scope_line}

הסיכומים החלקיים:
{partials}

הסיכום:"""


class SummaryError(Exception):
    """User-facing (Hebrew) error. No further calls are made after it."""


@dataclass
class SummarySource:
    number: int
    file_name: str
    topic: str | None
    path: str
    tokens: int
    text: str = field(repr=False, default="")


@dataclass
class SummaryPlan:
    scope: str                    # "topic" | "file"
    title: str                    # topic folder name or file name
    sources: list[SummarySource]
    mode: str                     # "single" | "staged"
    calls: int                    # exact number of LLM calls run_summary will make
    input_tokens: int             # estimated (with safety margin)
    max_output_tokens: int        # upper bound (sum of output limits)
    estimated_cost: float         # upper bound in USD
    skipped: list[str] = field(default_factory=list)   # files left out and why
    topic: str | None = None      # sub-topic summaries: the topic folder of the material

    @property
    def usable(self) -> bool:
        return bool(self.sources)


@dataclass
class SummaryResult:
    text: str
    found: bool
    sources: list[SummarySource]
    cited: list[int]
    warnings: list[str]
    calls_made: int
    input_tokens: int | None = None   # actual, when the API reports usage
    output_tokens: int | None = None


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * config.LLM_PRICE_INPUT_PER_1M
            + output_tokens * config.LLM_PRICE_OUTPUT_PER_1M) / 1_000_000


def _count(text: str) -> int:
    return len(get_tokenizer()(text))


def _read(path: Path) -> str:
    docs = extract_documents(path)
    if len(docs) == 1:
        return docs[0].text
    return "\n\n".join(f"(עמוד {d.metadata.get('page')})\n{d.text}" for d in docs)


def _indexed_entries(manager, scope_filter) -> tuple[list[dict], list[str]]:
    """Manifest entries matching the filter whose file on disk is unchanged."""
    entries, skipped = [], []
    for file_hash, entry in manager.manifest.items():
        if not scope_filter(entry):
            continue
        path = Path(entry["path"])
        if not path.exists():
            skipped.append(f"{entry['file_name']}: הקובץ לא נמצא במחשב.")
        elif compute_file_hash(path) != file_hash:
            skipped.append(f"{entry['file_name']}: הקובץ השתנה, ועדיין לא עודכן במאגר הידע.")
        else:
            entries.append(entry)
    entries.sort(key=lambda e: (e.get("relative_path") or e["file_name"]).lower())
    return entries, skipped


def _make_plan(scope: str, title: str, entries: list[dict], skipped: list[str]) -> SummaryPlan:
    sources = []
    for number, entry in enumerate(entries, start=1):
        try:
            text = _read(Path(entry["path"]))
        except DocumentLoadError as error:
            skipped.append(f"{entry['file_name']}: {error}")
            continue
        sources.append(SummarySource(number, entry["file_name"], entry.get("topic"),
                                     entry["path"], _count(text), text))
    return _plan_from_sources(scope, title, sources, skipped)


def _plan_from_sources(scope: str, title: str, sources: list, skipped: list[str],
                       topic: str | None = None) -> SummaryPlan:
    # renumber after possible skips
    for i, s in enumerate(sources, start=1):
        s.number = i

    material_tokens = sum(s.tokens for s in sources)
    overhead = _count(SUMMARY_PROMPT) + 40 * len(sources)
    single_input = int((material_tokens + overhead) * TOKEN_SAFETY_MARGIN)

    if single_input <= MAX_SINGLE_INPUT_TOKENS:
        mode, calls = "single", 1 if sources else 0
        input_tokens, max_output = single_input, FINAL_MAX_OUTPUT_TOKENS
    else:
        mode = "staged"
        stage_overhead = _count(STAGE_PROMPT)
        stage_inputs = sum(int((s.tokens + stage_overhead) * TOKEN_SAFETY_MARGIN) for s in sources)
        combine_input = int((len(sources) * STAGE_MAX_OUTPUT_TOKENS + _count(COMBINE_PROMPT))
                            * TOKEN_SAFETY_MARGIN)
        calls = len(sources) + 1
        input_tokens = stage_inputs + combine_input
        max_output = len(sources) * STAGE_MAX_OUTPUT_TOKENS + FINAL_MAX_OUTPUT_TOKENS
    if not sources:
        input_tokens, max_output = 0, 0
    return SummaryPlan(scope, title, sources, mode, calls, input_tokens, max_output,
                       estimate_cost(input_tokens, max_output), skipped, topic)


def plan_topic_summary(manager, folder_key: str) -> SummaryPlan:
    """All indexed documents of a topic folder ("__uploads__" = uploaded files). Local only."""
    if folder_key == "__uploads__":
        flt = lambda e: e.get("source") == "uploads"                                   # noqa: E731
    else:
        flt = lambda e: e.get("source") == "knowledge_base" and e.get("topic") == folder_key  # noqa: E731
    entries, skipped = _indexed_entries(manager, flt)
    return _make_plan("topic", folder_key, entries, skipped)


def plan_file_summary(manager, path: str | Path) -> SummaryPlan:
    """One indexed file. Local only."""
    key = str(Path(path).resolve())
    entries, skipped = _indexed_entries(manager, lambda e: e.get("path") == key)
    if not entries and not skipped:
        skipped.append(f"{Path(path).name}: המסמך אינו נמצא במאגר הידע.")
    return _make_plan("file", Path(path).name, entries, skipped)


SUBTOPIC_MAX_CHUNKS = 6        # the sub-topic material: at most this many chunks ...
SUBTOPIC_MIN_SHARE = 0.5        # ... each scoring at least this share of the best one
SUBTOPIC_MIN_COVERAGE = 0.5     # share of the request's words that must appear in them


def _merge_chunks(texts: list[str]) -> str:
    """Join consecutive chunks of one file without repeating their overlap."""
    merged = texts[0]
    for text in texts[1:]:
        overlap = next((n for n in range(min(len(merged), len(text), 2000), 20, -1)
                        if merged.endswith(text[:n])), 0)
        merged = merged + text[overlap:] if overlap else f"{merged}\n…\n{text}"
    return merged


def plan_subtopic_summary(manager, subject: str, topic: str | None = None) -> SummaryPlan:
    """Only the material about `subject` ("סריקות עצים"), within `topic` when given,
    or else the topic that the material's own terms point to. Local only: the
    keyword search over the stored chunks + the glossary - no embeddings, no API."""
    from rag import topics as topic_rules
    from rag.hybrid import tokenize

    title = subject.strip()
    keywords = manager.keyword_index()
    topic = topic or topic_rules.infer_topic(subject, keywords.glossary)
    known_topic = topic          # named, or pointed to by the material's terms
    current = {h for h, e in manager.manifest.items()
               if Path(e["path"]).exists() and compute_file_hash(Path(e["path"])) == h}
    query = topic_rules.retrieval_query(subject, glossary=keywords.glossary)
    hits = [(node, score) for node, score in keywords.search(query, max(len(keywords), 1))
            if node.metadata.get("doc_id") in current]
    if topic is None and hits:
        # still unknown: the topic with the strongest matches
        totals: dict[str, float] = {}
        for node, score in hits[:5]:
            key = node.metadata.get("topic") or "__uploads__"
            totals[key] = totals.get(key, 0.0) + score
        topic = max(totals, key=totals.get)
    if topic:
        hits = [(n, s) for n, s in hits if (n.metadata.get("topic") or "__uploads__") == topic]
    selected = ([n for n, s in hits[:SUBTOPIC_MAX_CHUNKS] if s >= SUBTOPIC_MIN_SHARE * hits[0][1]]
                if hits else [])
    if selected:
        # A section usually continues in the next chunk (the code after a heading),
        # which on its own scores low - add the one after the best match.
        best = selected[0]
        same_file = sorted((n for n in keywords.nodes
                            if n.metadata.get("doc_id") == best.metadata.get("doc_id")),
                           key=lambda n: n.start_char_idx or 0)
        position = next(i for i, n in enumerate(same_file) if n.node_id == best.node_id)
        if position + 1 < len(same_file) and same_file[position + 1] not in selected:
            selected.insert(1, same_file[position + 1])
        selected = selected[:SUBTOPIC_MAX_CHUNKS]

    # The words of the request must really appear in the selected material
    # (not just one common word) - otherwise it is "not found", no guessing.
    found_tokens = set()
    for node in selected:
        found_tokens |= set(tokenize(node.get_content()))
    words = [w for w in re.findall(r"[א-ת]+|[A-Za-z0-9_+#]+", subject.lower()) if tokenize(w)]
    covered = [w for w in words if set(tokenize(w)) & found_tokens]
    if not selected or not words or len(covered) < SUBTOPIC_MIN_COVERAGE * len(words):
        where = f" בנושא {known_topic}" if known_topic and known_topic != "__uploads__" else ""
        return _plan_from_sources("subtopic", title, [],
                                  [f"לא נמצא במאגר הידע חומר על \"{title}\"{where}."], known_topic)

    by_file: dict[str, list] = {}
    for node in selected:
        by_file.setdefault(node.metadata["doc_id"], []).append(node)
    sources = []
    for doc_id, nodes in by_file.items():
        entry = manager.manifest[doc_id]
        nodes.sort(key=lambda n: n.start_char_idx or 0)
        text = _merge_chunks([n.get_content() for n in nodes])
        sources.append(SummarySource(0, entry["file_name"], entry.get("topic"), entry["path"],
                                     _count(text), text))
    sources.sort(key=lambda s: s.file_name.lower())
    return _plan_from_sources("subtopic", title, sources, [], topic)


def create_summary_llm(max_tokens: int):
    """OpenAI chat model for summaries: same model, no automatic retries."""
    from llama_index.llms.openai import OpenAI

    return OpenAI(model=config.LLM_MODEL, temperature=0, max_tokens=max_tokens,
                  max_retries=0, timeout=120)


def _material(sources) -> str:
    return "\n\n".join(f"[{s.number}] (קובץ: {s.file_name}"
                       f"{', נושא: ' + s.topic if s.topic else ''})\n{s.text}" for s in sources)


def _scope_line(plan: SummaryPlan) -> str:
    if plan.scope == "file":
        return f"הסיכום הוא של הקובץ {plan.title}."
    if plan.scope == "subtopic":
        where = (f" מתוך חומר הלימוד בנושא {plan.topic}"
                 if plan.topic and plan.topic != "__uploads__" else "")
        return (f"הסיכום הוא של הנושא \"{plan.title}\"{where}. החומר שלהלן הוא רק הקטעים "
                f"שעוסקים בנושא הזה - סכם רק אותם.")
    return f"הסיכום הוא של כל חומר הלימוד בנושא {plan.title} ({len(plan.sources)} קבצים)."


def _usage(response) -> tuple[int, int]:
    raw = getattr(response, "raw", None)
    usage = getattr(raw, "usage", None) if raw is not None else None
    if usage is None and isinstance(raw, dict):
        usage = raw.get("usage")
    if usage is None:
        return 0, 0
    get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
    return get("prompt_tokens"), get("completion_tokens")


def run_summary(plan: SummaryPlan, make_llm=create_summary_llm) -> SummaryResult:
    """Make exactly plan.calls LLM calls. Any failure stops immediately."""
    if not plan.usable:
        return SummaryResult(config.NO_ANSWER_MESSAGE, False, [], [], plan.skipped, 0)
    calls, tokens_in, tokens_out = 0, 0, 0

    def call(llm, prompt):
        nonlocal calls, tokens_in, tokens_out
        if calls >= plan.calls:
            raise SummaryError("נעצר: מספר הקריאות חרג מהתוכנית שאושרה.")
        calls += 1
        try:
            response = llm.complete(prompt)
        except Exception as error:
            raise SummaryError(f"יצירת הסיכום נכשלה (קריאה {calls} מתוך {plan.calls}): "
                               f"{friendly_api_error(error)}") from error
        i, o = _usage(response)
        tokens_in, tokens_out = tokens_in + i, tokens_out + o
        return str(response).strip()

    if plan.mode == "single":
        text = call(make_llm(FINAL_MAX_OUTPUT_TOKENS), SUMMARY_PROMPT.format(
            no_answer=config.NO_ANSWER_MESSAGE, scope_line=_scope_line(plan),
            material=_material(plan.sources)))
        material = "\n".join(s.text for s in plan.sources)
    else:
        stage_llm = make_llm(STAGE_MAX_OUTPUT_TOKENS)
        partials = []
        for s in plan.sources:
            partials.append(f"[{s.number}] (קובץ: {s.file_name})\n" + call(stage_llm, STAGE_PROMPT.format(
                part=f"{s.number} מתוך {len(plan.sources)}", number=s.number,
                file_name=s.file_name, text=s.text)))
        text = call(make_llm(FINAL_MAX_OUTPUT_TOKENS), COMBINE_PROMPT.format(
            scope_line=_scope_line(plan), partials="\n\n".join(partials)))
        material = "\n".join(s.text for s in plan.sources)

    if not text or config.NO_ANSWER_MESSAGE in text:
        return SummaryResult(config.NO_ANSWER_MESSAGE, False, plan.sources, [], list(plan.skipped),
                             calls, tokens_in or None, tokens_out or None)
    checked, cited, warnings = verify_summary(text, material, len(plan.sources))
    return SummaryResult(checked, True, plan.sources, cited, list(plan.skipped) + warnings,
                         calls, tokens_in or None, tokens_out or None)


# --------------------------------------------------------------- verification
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_CITATION = re.compile(r"\[(\d+)\]")


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def verify_summary(text: str, material: str, n_sources: int) -> tuple[str, list[int], list[str]]:
    """Local reliability check (no API).

    * every non-empty line of every code block must appear in the material;
      otherwise the block is removed and a warning is added
    * citations [n] outside 1..n_sources are removed with a warning
    Returns (clean_text, cited_source_numbers, warnings).
    """
    material_lines = {_norm(l) for l in material.splitlines() if _norm(l)}
    material_flat = _norm(material)
    warnings = []

    def check_block(match):
        lines = [_norm(l) for l in match.group(1).splitlines() if _norm(l)]
        bad = [l for l in lines if l not in material_lines and l not in material_flat
               and l not in ("...", "…")]
        if bad:
            warnings.append("הוסרה דוגמת קוד שלא נמצאה כפי שהיא בחומר הלימוד.")
            return ""
        return match.group(0)

    text = _FENCE.sub(check_block, text)

    cited = set()

    def check_citation(match):
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            cited.add(n)
            return match.group(0)
        warnings.append(f"הוסרה הפניה למקור [{n}] שאינו קיים.")
        return ""

    text = _CITATION.sub(check_citation, text)
    return re.sub(r"\n{3,}", "\n\n", text).strip(), sorted(cited), warnings
