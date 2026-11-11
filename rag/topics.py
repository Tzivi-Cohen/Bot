"""Local (no LLM) recognition of study topics and summary requests.

* TOPIC_ALIASES maps every topic folder to its Hebrew/English names.
* detect_topics() finds the topics mentioned in a question.
* expand_query() appends the topic's canonical search term when the question
  names it differently (e.g. "פייתון" -> adds "Python"), so the hybrid search
  can match the material, which uses the English name. This is a fixed
  lookup table - not LLM query expansion, no extra API calls.
* is_summary_request() recognizes requests for a summary in Hebrew/English.
"""

import re

# folder name -> (canonical search term, aliases). Aliases are matched
# case-insensitively; Hebrew aliases also match with a prefix letter
# (בפייתון, לארדואינו, ...).
TOPIC_ALIASES = {
    "Python": ("Python", ["python", "פייתון", "פיתון", "פייטון"]),
    "C++": ("C++", ["c++", "cpp", "cplusplus", "c plus plus", "סי פלוס פלוס", "סי++", "סי פלוספלוס"]),
    "C": ("C", ["שפת c", "שפת סי", "c language", "ansi c", "בשפת c", "שפת־c", "שפת־סי"]),
    "Data-Structures-and-Algorithms": ("מבני נתונים", [
        "מבני נתונים", "מבנה נתונים", "מבני הנתונים", "אלגוריתמים", "אלגוריתם", "data structures",
        "data structure", "algorithms", "algorithm", "dsa"]),
    "Arduino": ("Arduino", ["arduino", "ארדואינו", "ארדוינו"]),
    "STM32": ("STM32", ["stm32", "stm 32", "אס טי אם", "אס.טי.אם", "stm"]),
    "Electronics": ("אלקטרוניקה", ["אלקטרוניקה", "electronics", "electronic", "אלקטרוני"]),
    "Embedded-Systems": ("Embedded", ["מערכות משובצות", "מערכת משובצת", "משובצות", "משובצת",
                                      "embedded systems", "embedded system", "embedded", "אמבדד"]),
}

_HEB_PREFIX = "(?:[ובלמשכה]|וה|וב|ול|מה|שה|כש|לה|בה)?"
_SUMMARY_RE = re.compile(
    r"(?<![\w])(?:"
    r"ה?סכמ?(?:י|ו|ם|ת)?|ו?תסכ(?:ם|מי|מו)|לסכם|סכמי|סיכום|סיכומים|"
    r"תמצת(?:י|ו)?|לתמצת|תמצית|תקציר|סקירה|"
    r"summar(?:y|ize|ise|ies)|overview|recap|tl;?dr"
    r")(?![\w])",
    re.IGNORECASE,
)
_FILE_WORDS = re.compile(r"הקובץ הזה|את הקובץ|הקובץ הנוכחי|המסמך הזה|this file|this document|current file",
                         re.IGNORECASE)


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias.lower()).replace(r"\ ", r"[\s־\-]*")
    if re.match(r"[א-ת]", alias):
        return re.compile(rf"(?<![א-ת\w]){_HEB_PREFIX}[־\-]?{escaped}(?![א-ת\w])")
    # Latin aliases: also after a Hebrew prefix with maqaf ("ב־Python").
    return re.compile(rf"(?<![A-Za-z0-9_+#]){escaped}(?![A-Za-z0-9_+#])")


_PATTERNS = {topic: [_alias_pattern(a) for a in aliases] for topic, (_, aliases) in TOPIC_ALIASES.items()}
# A lone "C" (e.g. "מצביעים ב־C") counts as the C topic only when it is not C++/C#.
_LONE_C = re.compile(r"(?<![A-Za-z0-9_])c(?![A-Za-z0-9_+#])")


def detect_topics(text: str) -> list[str]:
    """Topic folders mentioned in the text, in TOPIC_ALIASES order."""
    lowered = text.lower()
    found = [topic for topic, patterns in _PATTERNS.items() if any(p.search(lowered) for p in patterns)]
    if "C" not in found and _LONE_C.search(lowered):
        found.append("C")
    return [t for t in TOPIC_ALIASES if t in found]


# Hebrew study terms -> the English term used in the material (and back).
# Fixed table, no LLM. Keys are base forms; prefixed forms ("בבנאי") match too.
CONCEPTS = {
    "בנאי": "Constructor", "בנאים": "Constructor", "מפרק": "Destructor", "הורס": "Destructor",
    "מחלקה": "class", "מחלקות": "class", "אובייקט": "object", "אובייקטים": "object",
    "ירושה": "Inheritance", "פולימורפיזם": "Polymorphism", "הפניה": "Reference",
    "רפרנס": "Reference", "מצביע": "Pointer", "מצביעים": "Pointer", "מערך": "array",
    "מערכים": "array", "מבנה": "struct", "מבנים": "struct", "לולאה": "Loops", "לולאות": "Loops",
    "פונקציה": "def", "פונקציות": "def", "משתנה": "Variables", "משתנים": "Variables",
    "חריגה": "Exception", "חריגות": "Exception", "קובץ": "open", "קבצים": "open",
    "רשימה מקושרת": "Linked List", "מחסנית": "Stack", "תור": "Queue", "עץ": "Tree",
    "עצים": "Tree", "מיון": "Sort", "חיפוש בינארי": "Binary Search", "רקורסיה": "Recursion",
    "פסיקה": "Interrupt", "פסיקות": "Interrupt", "טיימר": "Timer", "טיימרים": "Timer",
    "נגד": "Resistor", "נגדים": "Resistor", "קבל": "Capacitor", "קבלים": "Capacitor",
    "דיודה": "Diode", "דיודות": "Diode", "טרנזיסטור": "Transistor", "טרנזיסטורים": "Transistor",
    "מתח": "Voltage", "זרם": "Current", "התנגדות": "Resistance", "חיישן": "sensor",
    "חיישנים": "sensor", "זיכרון": "Memory", "זמן אמת": "Real-Time",
}
_CONCEPT_PATTERNS = [(_alias_pattern(k), v) for k, v in sorted(CONCEPTS.items(), key=lambda kv: -len(kv[0]))]


def concept_terms(text: str) -> list[str]:
    """English material terms for the Hebrew study terms in the text."""
    lowered = text.lower()
    terms = []
    for pattern, term in _CONCEPT_PATTERNS:
        if pattern.search(lowered) and term not in terms:
            terms.append(term)
    return terms


def expand_query(question: str, context_topic: str | None = None) -> str:
    """Append the material's own terms for what the question names differently:
    the canonical topic name ("פייתון" -> Python) and English study terms
    ("בנאי" -> Constructor). context_topic (e.g. the topic of the previous
    question) is added when the question names no topic. No LLM, no API call."""
    lowered = question.lower()
    topics = detect_topics(question) or ([context_topic] if context_topic in TOPIC_ALIASES else [])
    extra = [TOPIC_ALIASES[t][0] for t in topics if TOPIC_ALIASES[t][0].lower() not in lowered]
    extra += [t for t in concept_terms(question) if t.lower() not in lowered and t not in extra]
    return f"{question} ({', '.join(extra)})" if extra else question


# Request / conversation phrasing that describes the request, not the content:
# "אני לא מבינה את עניין ה...", "הסבירי לי את...", "יש לך מידע על...".
_REQUEST_PHRASES = re.compile(
    r"(?<![א-ת\w])(?:"
    r"(?:אני\s+)?(?:עדיין\s+)?לא\s+(?:מבינה|מבין|מבינים|הבנתי|ברור\s+לי)(?:\s+(?:טוב|את|מה|איך|למה))*"
    r"|(?:אני\s+)?(?:רוצה|צריכה|צריך|מנסה)\s+(?:להבין|לדעת|ללמוד|לשאול)(?:\s+(?:את|על|מה))?"
    r"|(?:תוכל|תוכלי|תוכלו|אפשר)\s+(?:בבקשה\s+)?(?:להסביר|לספר|לפרט)(?:\s+(?:לי|לנו))?(?:\s+(?:את|על))?"
    r"|(?:הסבר|הסבירי|הסבירו|תסביר|תסבירי|תסבירו|ספר|ספרי|תספר|תספרי|פרט|פרטי)"
    r"(?:\s+(?:לי|לנו))?(?:\s+בבקשה)?(?:\s+(?:את|על))?"
    r"|(?:יש|אין)\s+(?:לך|לכם|לבוט)\s+(?:מידע|חומר|משהו)(?:\s+(?:על|לגבי|בנושא))?"
    r"|(?:עזור|עזרי|תעזור|תעזרי)\s+לי(?:\s+(?:להבין|עם))?"
    r"|(?:את\s+)?(?:עניין|הנושא\s+של)(?=\s+[א-תA-Za-z])"
    r")(?![א-ת\w])",
)
# A follow-up continues the previous question ("ומה לגבי...", "אבל...", "תסבירי את זה").
_FOLLOW_UP = re.compile(
    r"^\s*(?:אבל|אז|גם|רגע|ומה|ואם|ואיך|ולמה|ומתי|וגם|ועוד|ועכשיו|ובנוסף|ולגבי|לגבי|מה\s+לגבי|מה\s+עם)(?![א-ת])"
    r"|(?<![א-ת])(?:את|על|ב|ל)[־\s]?(?:זה|זאת|הזה|הזאת)(?![א-ת])"
    r"|(?<![א-ת])(?:אותו|אותה|אותם|אותן|עליו|עליה|עליהם|ממנו|ממנה)(?![א-ת])"
)


def strip_request_phrases(text: str) -> str:
    """The question without its request phrasing - only for search.
    "אני לא מבינה את עניין הסריקות בעצים" -> "הסריקות בעצים"."""
    if not _REQUEST_PHRASES.search(text):
        return text
    stripped = _REQUEST_PHRASES.sub(" ", text)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,.?!:")
    return stripped or text


def has_own_content(text: str) -> bool:
    """Does the message name something to search for (beyond request words and
    topic names)? "ומה לגבי מחסנית?" -> yes; "תסבירי את זה שוב" -> no."""
    from rag.hybrid import tokenize
    return bool(tokenize(strip_topic_names(strip_request_phrases(text))))


# Searched (an example chunk says "דוגמה"), but not a subject of its own:
# "תן לי דוגמה" continues the previous question.
_GENERIC_WORDS = re.compile(r"(?<![א-ת])[ובלמשכה]?(?:דוגמה|דוגמא|דוגמאות|דוגמת|הסבר|סיכום|תרגיל|תרגילים|שאלה|שאלות)(?![א-ת])")


def _content_tokens(text: str) -> set[str]:
    from rag.hybrid import tokenize
    return set(tokenize(_GENERIC_WORDS.sub(" ", strip_topic_names(strip_request_phrases(text)))))


def continues_previous(text: str, previous: str) -> bool:
    """Is the message about the same subject as the previous question?
    No content of its own, or a shared term (in any inflection)."""
    own = _content_tokens(text)
    return not own or bool(own & _content_tokens(previous))


def is_follow_up(text: str) -> bool:
    """Does the message continue the previous question? Local rules, no LLM."""
    return bool(_FOLLOW_UP.search(text)) or not has_own_content(text)


def glossary_terms(text: str, glossary) -> list:
    """Terms of the material's own Hebrew/English glossary (rag.glossary) in the text."""
    if not glossary:
        return []
    from rag.glossary import find_terms
    return find_terms(text, glossary)


def english_glossary_terms(text: str, glossary) -> list:
    """Glossary terms named in ENGLISH in the text ("Interrupts" -> the Hebrew "פסיקות")."""
    if not glossary:
        return []
    from rag.glossary import find_english_terms
    return find_english_terms(text, glossary)


def infer_topic(text: str, glossary=None, keyword_index=None) -> str | None:
    """The topic the text is about, without the user naming it:
    1. the topic name it mentions;
    2. else the one topic that the material's terms in it belong to
       ("סריקות עצים" -> data structures);
    3. else (keyword_index given) the topic of most of the best-matching chunks
       ("איך עובדות פסיקות?" - the term is in several topics' material).
    Local only, no API call."""
    named = detect_topics(text)
    if named:
        return named[0] if len(named) == 1 else None
    counts: dict[str, int] = {}
    for term in glossary_terms(text, glossary) + english_glossary_terms(text, glossary):
        for topic in term.topics:
            if topic in TOPIC_ALIASES:
                counts[topic] = counts.get(topic, 0) + len(term.keys)
    if counts:
        best = max(counts.values())
        leaders = [topic for topic, n in counts.items() if n == best]
        if len(leaders) == 1:
            return leaders[0]
    if keyword_index is None or not _content_tokens(text):
        return None       # "תן לי דוגמה" alone says nothing about the topic
    totals: dict[str, float] = {}
    for node, score in keyword_index.search(retrieval_query(text, glossary=glossary), 5):
        topic = node.metadata.get("topic")
        if topic in TOPIC_ALIASES:
            totals[topic] = totals.get(topic, 0.0) + score
    if not totals:
        return None
    best_topic = max(totals, key=totals.get)
    # only a clear majority counts - otherwise the topic stays unknown
    return best_topic if totals[best_topic] >= 0.5 * sum(totals.values()) else None


def retrieval_query(question: str, context_topic: str | None = None, glossary=None,
                    previous: str | None = None) -> str:
    """The text used for SEARCH (the LLM still sees the original question).

    * request phrasing is removed ("אני לא מבינה את עניין...", "הסבר לי...");
    * Hebrew topic names are replaced by the material's name ("בפייתון" -> "Python"),
      so a rare Hebrew word doesn't pull unrelated chunks;
    * the material's own terms are appended: the topic, the fixed CONCEPTS table,
      and the glossary learned from the documents ("סריקות עץ" -> Tree Traversals);
    * previous: the previous question. Its content is added when this question
      continues it: it has no content of its own ("תסבירי את זה שוב", "אבל למה?"),
      or it shares a term with it ("אבל יש לך מידע על עצים?" after a question on
      "סריקות בעצים"). A question on a new subject ("ומה לגבי מחסנית?" after
      "רשימה מקושרת") only keeps the previous topic, through context_topic.
    No LLM, no API call.
    """
    text = strip_request_phrases(question)
    if previous and continues_previous(question, previous):
        text = f"{text} {strip_request_phrases(previous)}"
    glossary_english = [term.english for term in glossary_terms(text, glossary)]
    # and the other way: an English term the material also names in Hebrew
    # ("תסביר לי על Interrupts" -> also "פסיקות")
    glossary_english += [term.hebrew for term in english_glossary_terms(text, glossary)
                         if term.hebrew not in text]
    for topic, (canonical, aliases) in TOPIC_ALIASES.items():
        for alias in aliases:
            if re.match(r"[א-ת]", alias):
                text = _alias_pattern(alias).sub(f" {canonical} ", text.lower()) \
                    if _alias_pattern(alias).search(text.lower()) else text
    expanded = expand_query(re.sub(r"\s{2,}", " ", text).strip(), context_topic)
    extra = [t for t in dict.fromkeys(glossary_english) if t.lower() not in expanded.lower()]
    return f"{expanded} ({', '.join(extra)})" if extra else expanded


# Words of a summary request that are not its subject ("בצורה טובה ומקצועית").
_SUMMARY_FILLER = re.compile(
    r"(?<![א-ת\w])(?:[ובלמשכה]|וה|וב|ול|מה|שה|כש|לה|בה)?(?:"
    r"אפשר|תוכל|תוכלי|בבקשה|לי|לנו|את|של|על|לגבי|בנושא|נושא|הנושא|כל|הכל|החומר|חומר|חומרי|הלימוד|לימוד|"
    r"קצר|קצרה|קצרים|מפורט|מפורטת|ברור|ברורה|מסודר|מסודרת|טוב|טובה|מקצועי|מקצועית|מלא|מלאה|"
    r"צורה|אופן|ממש|רק|גם|עוד|שוב|קצת|מה|למדנו|הקורס|המקצוע|מקצוע|הזה|הזאת|זה|זאת|"
    r"קובץ|הקובץ|מסמך|המסמך|הנוכחי|"
    r"רוצה|צריך|צריכה|אני|תן|תני|תנו|הכן|הכיני|תכין|תכיני|כתוב|כתבי|תכתוב|תכתבי|עשה|עשי|תעשה|תעשי|"
    r"please|me|the|of|a|an|for|about|on|give|make|write|all|material|topic|short|full|detailed"
    r")(?![א-ת\w])",
    re.IGNORECASE,
)


def summary_subject(text: str) -> str:
    """What a summary request asks to summarize, beyond a topic name:
    "אפשר סיכום של סריקות עצים באלגוריתמים" -> "סריקות עצים". Empty when the
    request is for a whole topic ("הסכם לי את החומר בפייתון בצורה טובה")."""
    subject = _SUMMARY_RE.sub(" ", text)
    subject = _FILE_WORDS.sub(" ", subject)
    subject = strip_topic_names(strip_request_phrases(subject))
    subject = _SUMMARY_FILLER.sub(" ", subject)
    subject = re.sub(r"[^\w\s+#\-־]", " ", subject)
    words = [w.strip("־-") for w in subject.split()]
    return " ".join(w for w in words if len(w) > 1 or w.isdigit())


def strip_topic_names(text: str) -> str:
    """The text without the topic names it mentions (multi-word names like
    "סי פלוס פלוס" or "מערכות משובצות" are removed as a whole)."""
    lowered = text.lower()
    for patterns in _PATTERNS.values():
        for pattern in patterns:
            lowered = pattern.sub(" ", lowered)
    lowered = _LONE_C.sub(" ", lowered)
    return re.sub(r"\s{2,}", " ", lowered).strip()


def is_summary_request(text: str) -> bool:
    return bool(_SUMMARY_RE.search(text))


def mentions_current_file(text: str) -> bool:
    return bool(_FILE_WORDS.search(text))


def route(text: str, focus_folder: str | None = None, focus_file: str | None = None) -> dict:
    """Decide what a chat message is asking for - locally, before any API call.

    {"kind": "question"}                              -> regular RAG question
    {"kind": "summary", "scope": "file", "file": ...}  -> summarize the open/focused file
    {"kind": "summary", "scope": "topic", "topic": t}  -> summarize a topic folder
    {"kind": "summary", "scope": "subtopic", "subject": s, "topic": t or None}
                                                      -> summarize only the material on s
    {"kind": "choose_topic", "candidates": [...]}      -> ask the user which topic
    """
    if not is_summary_request(text):
        return {"kind": "question"}
    topics = detect_topics(text)
    if focus_file and mentions_current_file(text):
        return {"kind": "summary", "scope": "file", "file": focus_file}
    subject = summary_subject(text)
    if subject:
        # A part of a topic ("סריקות עצים באלגוריתמים"): the topic named, else the
        # focused folder; None = inferred later from the material's own terms.
        topic = topics[0] if len(topics) == 1 else (
            focus_folder if not topics and focus_folder in TOPIC_ALIASES else None)
        return {"kind": "summary", "scope": "subtopic", "subject": subject, "topic": topic}
    if focus_file and not topics:
        return {"kind": "summary", "scope": "file", "file": focus_file}
    if len(topics) == 1:
        return {"kind": "summary", "scope": "topic", "topic": topics[0]}
    return {"kind": "choose_topic", "candidates": topics}
