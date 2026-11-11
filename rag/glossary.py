"""A Hebrew -> English glossary learned from the study material itself.

The material names its terms in Hebrew followed by the English term in
parentheses: "סריקות עץ (Tree Traversals)", "מחלק מתח (Voltage Divider)",
"ערימת מינימום (Min-Heap)". build_glossary() collects these pairs from the
stored chunks, so a question in Hebrew ("אני לא מבינה את הסריקות בעצים") can
also be searched with the material's English term, and the topic folder the
term belongs to is known. Built locally from the text - no LLM, no API call,
no re-indexing. A new document automatically adds its own terms.
"""

import re
from dataclasses import dataclass, field

from rag.hybrid import _hebrew_forms, is_stopword

# Up to 3 Hebrew words right before "(English ...)".
_PAIR_RE = re.compile(
    r"((?:[א-ת][א-ת\"'־]*\s+){0,2}[א-ת][א-ת\"'־]*)\s*"
    r"\(([A-Za-z][A-Za-z0-9 +\-]{1,40}?)\s*(?:[/,)])"
)
_EN_SMALL = {"and", "of", "the", "a", "an", "in", "to", "for", "with", "or"}


@dataclass
class Term:
    hebrew: str                     # the Hebrew words, as written in the material
    english: str                    # the English term, as written in the material
    keys: list[set[str]]            # for each Hebrew word: its matching forms
    topics: set[str] = field(default_factory=set)


def _hebrew_words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s־]+", text) if w]


def _word_forms(word: str) -> set[str]:
    return {f for f in _hebrew_forms(word.lower()) if len(f) >= 2}


def _english_size(english: str) -> int:
    """How many Hebrew words name the term: like the English ("Tree Traversals" -> 2,
    "Min-Heap" -> 2); an acronym (ADC, DFS, LDR) is usually 2 Hebrew words."""
    words = [w for w in re.split(r"[\s\-]+", english) if w and w.lower() not in _EN_SMALL]
    if len(words) == 1 and words[0].isupper() and len(words[0]) >= 2:
        return 2
    return max(1, min(3, len(words)))


def _is_construct(word: str) -> bool:
    """Likely Hebrew construct form ("העמסת", "ערימת", "יציבות") - the word that
    follows belongs to the term ("העמסת פונקציות", not "פונקציות"). Words ending
    in "י" are not treated so - those are mostly adjectives ("קריטי", "מעגלי")."""
    return len(word) >= 3 and word[-1] == "ת" and not is_stopword(word)


def build_glossary(nodes) -> list[Term]:
    """Hebrew/English term pairs found in the chunks' text, with their topics."""
    terms: dict[tuple[str, str], Term] = {}
    for node in nodes:
        topic = node.metadata.get("topic")
        for hebrew, english in _PAIR_RE.findall(node.get_content()):
            english = english.strip()
            all_words = _hebrew_words(hebrew)
            # as many Hebrew words as the English term has ("סריקות עץ" <-> "Tree Traversals"),
            # one more when the first of them completes a construct form ("העמסת פונקציות")
            size = _english_size(english)
            if len(all_words) > size and _is_construct(all_words[-size - 1]):
                size += 1
            if len(all_words) < size:
                continue          # the Hebrew name is cut by punctuation - don't guess
            words = all_words[-size:]
            if any(len(w) < 2 or is_stopword(w) for w in words):
                continue
            key = (" ".join(words), english)
            if key not in terms:
                terms[key] = Term(key[0], english, [_word_forms(w) for w in words])
            if topic:
                terms[key].topics.add(topic)
    single = [t for t in terms.values() if len(t.keys) == 1]
    if single:
        from rag.hybrid import tokenize

        chunks = [(set(tokenize(n.get_content())), n.get_content().lower()) for n in nodes]
        rejected = {id(t) for t in single if not _co_occurs(t, chunks)}
    else:
        rejected = set()
    return [t for t in terms.values() if id(t) not in rejected]


MIN_CO_OCCURRENCE = 0.1


def _co_occurs(term: Term, chunks: list[tuple[set, str]]) -> bool:
    """A one-word pair is kept only if the English term really goes with the word:
    it appears in at least 10% of the chunks that contain the Hebrew word.
    "פסיקה (Interrupt)" passes; "נתונים (Attributes)" - one chance phrase among
    49 chunks with "נתונים" - does not. chunks: (tokens, lowercase text) per chunk."""
    english = term.english.lower()
    with_word = with_both = 0
    for tokens, text in chunks:
        if term.keys[0] & tokens:
            with_word += 1
            with_both += english in text
    return with_word == 0 or with_both >= MIN_CO_OCCURRENCE * with_word


def _english_key(text: str) -> str:
    """Lowercase English words, singular ("Interrupts" ~ "interrupt")."""
    words = re.findall(r"[a-z0-9+]+", text.lower())
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
                    for w in words)


def find_english_terms(text: str, glossary: list[Term]) -> list[Term]:
    """Glossary terms whose ENGLISH name is in the text ("תסביר לי על Interrupts"),
    so the material's Hebrew name ("פסיקות") can be searched too."""
    if not glossary or not re.search(r"[A-Za-z]", text):
        return []
    key = f" {_english_key(text)} "
    found = [t for t in glossary if len(t.english) >= 3 and f" {_english_key(t.english)} " in key]
    found.sort(key=lambda t: -len(t.english))
    return found


def find_terms(text: str, glossary: list[Term]) -> list[Term]:
    """Glossary terms whose Hebrew words all appear in the text (any inflection,
    with or without a prefix). Longer (more specific) terms first."""
    if not glossary:
        return []
    words = [_word_forms(w) for w in re.findall(r"[א-ת]+", text.lower()) if not is_stopword(w)]
    matches = []
    for term in glossary:
        positions = []
        for key in term.keys:
            position = next((i for i, forms in enumerate(words) if key & forms), None)
            if position is None:
                break
            positions.append(position)
        else:
            matches.append((term, set(positions)))
    # A term inside a longer matched term is dropped ("מחלק מתח" -> not also "מתח").
    matches.sort(key=lambda m: -len(m[0].keys))
    found, used = [], set()
    for term, positions in matches:
        if positions <= used and len(term.keys) < max(len(t.keys) for t in found or [term]):
            continue
        found.append(term)
        used |= positions
    return found
