"""Data layer of the study library UI (no Streamlit here - easy to test).

Builds the folder/file catalog from the knowledge_base and uploads folders,
attaches the index status of every file, searches files, and splits a study
text into display blocks (headings, paragraphs, bullet lists, code, tables).
No API calls: everything is local.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.document_loader import compute_file_hash, iter_supported_files, topic_for
from rag.topics import concept_terms, detect_topics, strip_topic_names

UPLOADS_KEY = "__uploads__"
GENERAL_KEY = "__general__"

# Display information for the known topic folders. Unknown folders still work
# (generic icon, folder name as title).
TOPICS = {
    "Python": ("Python", "🐍", "משתנים, פונקציות, מחלקות, קבצים וחריגות", "python"),
    "C": ("שפת C", "⚙️", "מצביעים, מערכים, מבנים וניהול זיכרון", "c"),
    "C++": ("C++", "🧩", "מחלקות, ירושה, פולימורפיזם ו־References", "cpp"),
    "Data-Structures-and-Algorithms": ("מבני נתונים ואלגוריתמים", "🌳",
                                       "רשימות, עצים, מיון וחיפוש", None),
    "Arduino": ("Arduino", "🔌", "קלט ופלט, חיישנים, PWM ו־Serial", "cpp"),
    "STM32": ("STM32", "🔲", "GPIO, פסיקות, טיימרים, ADC ותקשורת", "c"),
    "Electronics": ("אלקטרוניקה", "⚡", "חוק אוהם, נגדים, קבלים וטרנזיסטורים", None),
    "Embedded-Systems": ("מערכות משובצות", "📟", "Embedded C, זיכרון, פסיקות וזמן אמת", "c"),
}
TOPIC_ORDER = list(TOPICS)


@dataclass
class FileItem:
    path: Path
    name: str
    folder_key: str           # topic folder name, UPLOADS_KEY or GENERAL_KEY
    status: str               # "indexed" | "pending" | "updating" | "problem"
    status_text: str
    chunks: int = 0

    @property
    def key(self) -> str:
        return str(self.path.resolve())


@dataclass
class Folder:
    key: str
    title: str
    icon: str
    description: str
    language: str | None
    files: list = field(default_factory=list)

    @property
    def indexed_count(self) -> int:
        return sum(f.status == "indexed" for f in self.files)


def folder_info(key: str) -> tuple:
    if key == UPLOADS_KEY:
        return ("הקבצים שהעליתי", "📤", "מסמכים פרטיים שהועלו דרך האתר (לא עולים ל־GitHub)", None)
    if key == GENERAL_KEY:
        return ("כללי", "📁", "קבצים שאינם בתיקיית נושא", None)
    return TOPICS.get(key, (key, "📁", "תיקיית נושא", None))


def file_status(manager, path: Path, problem_checker) -> tuple[str, str, int]:
    """(status, text, chunks) for one file. problem_checker(path, hash) -> str | None."""
    file_hash = compute_file_hash(path)
    entry = manager.find_by_hash(file_hash)
    if entry:
        return "indexed", "במאגר הידע", entry["num_chunks"]
    problem = problem_checker(path, file_hash)
    if problem:
        return "problem", problem, 0
    previous = manager.entry_for_path(path)
    if previous:
        return ("updating",
                "הקובץ השתנה. הגרסה הקודמת זמינה במאגר הידע, והעדכון יושלם בסנכרון הבא.",
                previous["num_chunks"])
    return "pending", "תקין, עדיין לא נוסף למאגר הידע", 0


def count_he(n: int, one: str, many: str, zero: str | None = None) -> str:
    """Hebrew count: count_he(1, "מסמך אחד", "מסמכים") -> "מסמך אחד", (3, ...) -> "3 מסמכים"."""
    if n == 0 and zero:
        return zero
    return one if n == 1 else f"{n} {many}"


def build_catalog(manager, kb_dir: Path, uploads_dir: Path, problem_checker) -> list[Folder]:
    """All folders (known topics first, then other topics, general, uploads)."""
    folders: dict[str, Folder] = {}

    def add(key, path):
        if key not in folders:
            title, icon, desc, lang = folder_info(key)
            folders[key] = Folder(key, title, icon, desc, lang)
        status, text, chunks = file_status(manager, path, problem_checker)
        folders[key].files.append(FileItem(path, path.name, key, status, text, chunks))

    kb_dir = Path(kb_dir)
    for topic in TOPIC_ORDER:                     # show known topics even when empty
        if (kb_dir / topic).is_dir():
            title, icon, desc, lang = folder_info(topic)
            folders[topic] = Folder(topic, title, icon, desc, lang)
    for path in iter_supported_files(kb_dir):
        add(topic_for(path, kb_dir) or GENERAL_KEY, path)
    for path in iter_supported_files(uploads_dir):
        add(UPLOADS_KEY, path)

    def order(folder):
        if folder.key in TOPICS:
            return (0, TOPIC_ORDER.index(folder.key), "")
        if folder.key == GENERAL_KEY:
            return (2, 0, "")
        if folder.key == UPLOADS_KEY:
            return (3, 0, "")
        return (1, 0, folder.key.lower())

    for folder in folders.values():
        folder.files.sort(key=lambda f: f.name.lower())
    return sorted(folders.values(), key=order)


_SEARCH_STOP = {
    "את", "של", "על", "מה", "מהו", "מהי", "איך", "למה", "לי", "עם", "או", "גם", "כל", "זה", "זו",
    "הסבר", "תן", "תני", "בבקשה", "חומר", "החומר", "מתוך", "קובץ", "קבצים", "מסמך", "מסמכים",
    "the", "a", "an", "of", "and", "in", "to", "what", "is", "how", "about",
}
_PUNCT = ".,;:?!\"'()[]{}«»״׳"
_HEB_PREFIXES = ("וה", "וב", "ול", "ומ", "וש", "שה", "שב", "מה", "כש", "לה", "בה", "כה",
                 "ו", "ה", "ב", "ל", "מ", "ש", "כ")


def _word_variants(word: str) -> list[str]:
    """The word, the word without Hebrew prefix letters, and the material's terms for it."""
    w = word.lower().strip(_PUNCT).lstrip("־-")
    variants = [w] if w else []
    if w and "א" <= w[0] <= "ת":
        for prefix in _HEB_PREFIXES:
            if w.startswith(prefix) and len(w) - len(prefix) >= 2:
                variants.append(w[len(prefix):].lstrip("־-"))
    for v in list(variants):
        variants += [term.lower() for term in concept_terms(v)]
    return list(dict.fromkeys(v for v in variants if v))


def search_files(catalog: list[Folder], query: str, read_text, folder_key: str | None = None):
    """Library search by topic, file name and content (local, no API).

    * A topic named in Hebrew or English ("פייתון", "בפייתון", "Python", "C++")
      returns all documents of that topic.
    * Other words must all appear in the file name or content; Hebrew prefix
      letters are ignored and Hebrew study terms also match the material's
      English terms ("ירושה" -> Inheritance).
    * When the query names a topic AND other words, only that topic is searched.

    Returns [(FileItem, snippet, reason)], best matches first.
    """
    query = query.strip()
    if not query:
        return []
    topics = detect_topics(query)
    # Topic names (also multi-word ones) are handled by the topic match.
    remaining = strip_topic_names(query) if topics else query
    words = []
    for raw in remaining.split():
        w = raw.strip(_PUNCT).strip("־-")
        if not w or w.lower() in _SEARCH_STOP:
            continue
        words.append(w)

    results = []
    for folder in catalog:
        if folder_key and folder.key != folder_key:
            continue
        in_topic = folder.key in topics
        if topics and not in_topic:
            continue
        for item in folder.files:
            if not words:
                results.append((item, "", f"בנושא {folder.title}", 0, 0))
                continue
            name = item.name.lower()
            try:
                text = read_text(item.path)
            except Exception:
                text = ""
            lowered = text.lower()
            found = []
            for w in words:
                variant = next((v for v in _word_variants(w) if v in name or v in lowered), None)
                if variant is None:
                    break
                found.append(variant)
            else:
                name_hit = any(v in name for v in found)
                score = sum(lowered.count(v) for v in found)
                pos = next((lowered.find(v) for v in found if v in lowered), -1)
                snippet = ""
                if pos >= 0:
                    start = max(0, pos - 60)
                    snippet = ("…" if start else "") + " ".join(text[start:pos + 140].split()) + "…"
                reason = "בשם הקובץ" if name_hit else "בתוכן המסמך"
                results.append((item, snippet, reason, 1 if name_hit else 0, score))
    results.sort(key=lambda r: (-r[3], -r[4], r[0].name.lower()))
    return [(item, snippet, reason) for item, snippet, reason, _, _ in results]


# ------------------------------------------------------------------ bidi
# Runs of Latin letters / digits / code symbols inside Hebrew text. Without
# isolation the browser's bidi algorithm may show "C++" as "++C" or
# "7 / 2 == 3.5" as "3.5 == 2 / 7" in a right-to-left paragraph.
_LTR_RUN = re.compile(r"[A-Za-z0-9_\-+*/%=<>!&|^~()\[\]{}.,:;'\"#\\@$?` ]+")


def _bidi_segments(text: str):
    """Yield (is_ltr, piece): Latin/code runs are LTR, the rest stays as is.

    Spaces and sentence punctuation at the edges of a run are kept outside it,
    so "לדוגמה: 42" keeps the colon with the Hebrew word.
    """
    pos = 0
    for match in _LTR_RUN.finditer(text):
        run = match.group(0)
        if not re.search(r"[A-Za-z0-9]", run):
            continue
        core = run.strip(" ")
        lead = run[: len(run) - len(run.lstrip(" "))]
        # sentence punctuation only - not the dot of a name such as ".env"
        while core and core[0] in ".,;:?!" and not re.match(r"\.[A-Za-z0-9_]", core):
            lead += core[0]
            core = core[1:].lstrip(" ") if core[1:2] == " " else core[1:]
            if run.find(core) > 0:
                lead = run[: run.find(core)]
        trail_punct = ""
        while core and core[-1] in ".,;:?!":
            trail_punct = core[-1] + trail_punct
            core = core[:-1]
        if core.count("(") != core.count(")") and core.endswith(")"):
            trail_punct = ")" + trail_punct      # closing paren of the Hebrew sentence
            core = core[:-1]
        if core.count("(") > core.count(")") and core.rstrip().endswith("("):
            # opening paren of the Hebrew sentence: ".env (לפי הדוגמה..." - the "("
            # belongs to the Hebrew text that follows, not to the Latin name
            core = core.rstrip()[:-1].rstrip()
        if not core:
            continue
        core_start = match.start() + run.find(core)
        yield False, text[pos:core_start]
        yield True, core
        pos = core_start + len(core)
    yield False, text[pos:]


_DIRECTION_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮"), None)


def strip_direction_marks(text: str) -> str:
    """Remove invisible LRM/RLM/embedding marks (display only - files are not changed).
    The LTR isolation below replaces them; mixed with it they scramble the order."""
    return text.translate(_DIRECTION_MARKS)


def bidi_html(text: str) -> str:
    """HTML-escape text and wrap Latin/code runs in <bdi dir="ltr">."""
    import html

    text = strip_direction_marks(text)

    return "".join(f'<bdi dir="ltr">{html.escape(piece)}</bdi>' if is_ltr else html.escape(piece)
                   for is_ltr, piece in _bidi_segments(text))


def bidi_text(text: str) -> str:
    """Plain-text version (button labels etc.): LRI ... PDI around Latin/code runs."""
    text = strip_direction_marks(text)
    return "".join(f"⁦{piece}⁩" if is_ltr else piece
                   for is_ltr, piece in _bidi_segments(text))


_CPP_LIKE = re.compile(r"(?<![\w⁦])([A-Za-z][A-Za-z0-9]*\+\+|C#)")


def isolate_plus_names(text: str) -> str:
    """Only "C++"-like names (safe inside Markdown, e.g. chat answers)."""
    return _CPP_LIKE.sub(lambda m: f"⁦{m.group(1)}⁩", text)


# ------------------------------------------------------------------ content
_UNDERLINE = re.compile(r"^\s*([=\-])\1{2,}\s*$")


def guess_language(code: str, default: str | None) -> str | None:
    if re.search(r"^\s*(def |class \w+.*:|import |from \w+ import|print\()", code, re.M):
        return "python"
    if re.search(r"#include|std::|->|;\s*$", code, re.M):
        return default if default in ("c", "cpp") else "cpp"
    return default


_NUMBERED = re.compile(r"^(\d{1,2})[.)]\s+(\S.*)$")
# Lines that open a new paragraph, with the label shown in bold.
_LABEL = re.compile(
    r"^((?:שאלה\s*\d+|פתרון|תשובה|הערה|דוגמה|דוגמאות|כללים|שימושים|תכונות|יתרונות|חסרונות|"
    r"אזהרה|חשוב|מושגים|סיבוכיות|שיקולים מרכזיים|אמצעי זהירות|ההבדלים בין מערך למצביע)"
    r"[^:]{0,20}):\s*")
_BREAKS_PARAGRAPH = re.compile(r"^(?:שאלה\s*\d+|פתרון|תשובה)\s*:")


def line_html(line: str) -> str:
    """One display line: bold label (e.g. "פתרון:"), the rest bidi-isolated HTML."""
    match = _LABEL.match(line)
    if not match:
        return bidi_html(line)
    rest = line[match.end():]
    import html

    return f"<strong>{html.escape(match.group(1))}:</strong> {bidi_html(rest)}".rstrip()


def parse_study_text(text: str, language: str | None = None) -> list[tuple]:
    """Split a study text into display blocks (the file itself is not changed).

    ("title", text, level) | ("para", [lines]) | ("bullets", [items]) |
    ("numbered", [items]) | ("code", code, language) | ("table", [[cells]])

    Conventions of the knowledge-base files: a heading is a line underlined by
    ==== (level 1) or ---- (level 2); code is indented by 4 spaces; bullet
    lines start with "- "; numbered items start with "1." or "1)"; table rows
    start with "|". Each source line keeps its own row inside a paragraph, and
    "שאלה N:" / "פתרון:" lines open a new paragraph.
    """
    lines = strip_direction_marks(text).replace("\r\n", "\n").split("\n")
    blocks: list[tuple] = []
    para: list[str] = []
    items: list[str] = []
    kind = None                       # "bullets" | "numbered" for the open list
    i = 0

    def flush_para():
        nonlocal para
        if para:
            blocks.append(("para", para))
            para = []

    def flush_list():
        nonlocal items, kind
        if items:
            blocks.append((kind, items))
        items, kind = [], None

    def flush():
        flush_para()
        flush_list()

    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _UNDERLINE.match(nxt) and not line.startswith("    "):
            flush()
            blocks.append(("title", line.strip(), 1 if nxt.strip().startswith("=") else 2))
            i += 2
            continue
        if line.startswith("    ") and line.strip() and not line.lstrip().startswith("- "):
            flush()
            code = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                if not lines[i].strip() and not (i + 1 < len(lines) and lines[i + 1].startswith("    ")):
                    break
                code.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            code_text = "\n".join(code).rstrip()
            blocks.append(("code", code_text, guess_language(code_text, language)))
            continue
        if line.lstrip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        stripped = line.strip()
        numbered = _NUMBERED.match(stripped)
        if not stripped:
            flush()
        elif stripped.startswith("- "):
            flush_para()
            if kind != "bullets":
                flush_list()
                kind = "bullets"
            items.append(stripped[2:])
        elif numbered:
            flush_para()
            if kind != "numbered":
                flush_list()
                kind = "numbered"
            items.append(numbered.group(2))
        elif line.startswith("  ") and items:
            items[-1] += " " + stripped          # continuation of a list item
        else:
            flush_list()
            if _BREAKS_PARAGRAPH.match(stripped):
                flush_para()
            para.append(stripped)
        i += 1
    flush()
    return blocks
