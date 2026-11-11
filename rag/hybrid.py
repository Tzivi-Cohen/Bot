"""Local keyword search (BM25) and Reciprocal Rank Fusion for hybrid retrieval.

The BM25 index is built in memory from the chunk texts that are already
stored in ChromaDB - no re-indexing, no embeddings, no API calls.
"""

import math
import re
from collections import Counter

# Identifiers (analogWrite, HAL_GPIO_WritePin), numbers, Hebrew words (letters
# only - the maqaf "־" is a separator), and programming operators.
_TOKEN_RE = re.compile(
    r"[a-z_][a-z0-9_]*"
    r"|\d+(?:\.\d+)?"
    r"|[א-ת]+"
    r"|//|\*\*|==|!=|<=|>=|->|::|<<|>>|&&|\|\||\+\+|[/%&|^~]"
)

# Common one/two-letter Hebrew prefixes (ו, ה, ב, ל, מ, ש, כ and pairs like "וה").
_PREFIXES_1 = set("והבלמשכ")
_PREFIXES_2 = {"וה", "וב", "ול", "ומ", "וש", "שה", "שב", "של", "מה", "כש", "לה", "בה", "כה"}

_STOPWORDS = {
    "של", "את", "על", "זה", "זו", "הוא", "היא", "הם", "או", "גם", "כי", "עם",
    "לא", "יש", "אין", "בין", "כל", "מה", "איך", "למה", "אם", "כמו", "אבל",
    # request / question words - they describe the request, not the content
    "מהו", "מהי", "מהם", "מהן", "הסבר", "הסבירי", "הסבירו", "תסביר", "תסבירי", "תן", "תני",
    "תנו", "לי", "לנו", "אפשר", "בבקשה", "מתוך", "החומר", "חומר", "ההבדל", "רוצה", "צריך",
    # conversation words ("אני לא מבינה את עניין...", "יש לך מידע על...") - without
    # them a prefix-stripped "מבינה" -> "בינה" or "מידע" -> "ידע" matched unrelated material
    "אני", "אתה", "את", "אנחנו", "לך", "לכם", "מבין", "מבינה", "מבינים", "הבנתי", "להבין",
    "עניין", "נושא", "הנושא", "מידע", "המידע", "להסביר", "תוכל", "תוכלי", "תוכלו", "ספר",
    "ספרי", "תספר", "תספרי", "לדעת", "ללמוד", "קצת", "עוד", "בדיוק", "לגבי", "בנוגע",
    "צריכה", "רוצים", "תעזור", "תעזרי", "עזרה", "שאלה", "יודע", "יודעת", "בעצם", "ממש",
    "זאת", "אותו", "אותה", "אותם", "דבר", "משהו", "הזה", "הזאת", "האלה", "אלה", "היה", "יהיה",
    "שוב", "פירוט", "יותר", "פשוט", "פשוטה", "פשוטות", "פשוטים",
    "מילים", "במילים",
    # question-frame words ("איך עובד...", "מה התפקיד של...") - the question's
    # frame, not its subject; a rare one like "תפקיד" matched unrelated chunks
    "עובד", "עובדת", "עובדים", "עובדות", "פועל", "פועלת", "פועלים", "פועלות", "עושה", "עושים",
    "עושות", "תפקיד", "התפקיד", "תפקידו", "תפקידה", "תפקידם", "משמש", "משמשת", "משמשים",
    "משמשות", "מטרה", "המטרה", "מטרת", "כיצד", "מדוע", "האם", "אילו", "איזה", "איזו", "מתי",
    "קורה", "קורים", "נקרא", "נקראת", "אומר", "אומרת",
    "the", "a", "an", "is", "are", "of", "and", "or", "to", "in", "on", "for",
}

# Hebrew final letters are written differently at the end of a word (עץ / עצים).
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
_SUFFIXES = ("יות", "ים", "ות")


def hebrew_stem(word: str) -> str:
    """A light, rule-based Hebrew stem for matching inflections:
    final letters normalized, and plural / feminine endings removed
    (סריקות, סריקה -> סריק; עצים, עץ -> עצ; פסיקות, פסיקה -> פסיק).
    Not a full morphological analyzer - it is only used as an extra token."""
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 2 and word.endswith(suffix):
            return word[: -len(suffix)].translate(_FINALS)
    if len(word) >= 4 and word[-1] in "הת":
        word = word[:-1]
    return word.translate(_FINALS)


def _hebrew_forms(token: str) -> list[str]:
    """The word, the word without a common prefix, and their stems."""
    forms = [token]
    if len(token) > 4 and token[:2] in _PREFIXES_2:
        forms.append(token[2:])
    if len(token) > 3 and token[0] in _PREFIXES_1:
        forms.append(token[1:])
    for form in list(forms):
        stem = hebrew_stem(form)
        if stem != form and len(stem) >= 2:
            forms.append(stem)
    # Extra forms that are function words are noise ("לולאה" -> "ולאה" -> "ולא").
    extra = [f for f in forms[1:] if f not in _STOPWORDS
             and not (len(f) <= 3 and f[0] in _PREFIXES_1 and f[1:] in _STOPWORDS)]
    return list(dict.fromkeys([forms[0], *extra]))


def is_stopword(token: str) -> bool:
    """A stopword, also with a prefix ("שאני", "בעניין"). Stripped forms
    shorter than 3 letters are not checked ("כלי" is not "לי")."""
    if token in _STOPWORDS:
        return True
    if "א" <= token[:1] <= "ת":
        stripped = []
        if len(token) > 4 and token[:2] in _PREFIXES_2:
            stripped.append(token[2:])
        if len(token) > 3 and token[0] in _PREFIXES_1:
            stripped.append(token[1:])
        return any(len(form) >= 3 and form in _STOPWORDS for form in stripped)
    return False


def tokenize(text: str) -> list[str]:
    """Tokens for BM25, in Hebrew and English.

    Hebrew words with a common prefix are also indexed without it
    ("לשדות" -> also "שדות"), and with a light stem ("הסריקות" -> also
    "סריק", like "סריקה" in the material), so inflected forms still match.
    """
    return [form for forms in word_forms(text) for form in forms]


def word_forms(text: str) -> list[list[str]]:
    """For each word of the text (stopwords removed): the forms it matches."""
    words = []
    for token in _TOKEN_RE.findall(text.lower()):
        if is_stopword(token):
            continue
        if "א" <= token[0] <= "ת":
            words.append(_hebrew_forms(token))
        else:
            words.append([token])
    return words


class KeywordIndex:
    """In-memory BM25 over LlamaIndex nodes."""

    def __init__(self, nodes, k1: float = 1.5, b: float = 0.75):
        self.nodes = list(nodes)
        self.k1, self.b = k1, b
        words = [word_forms(n.get_content()) for n in self.nodes]
        self._docs = [Counter(form for forms in w for form in forms) for w in words]
        # length in words, not forms - the extra forms of Hebrew words must not
        # make Hebrew text look longer than English text
        self._lengths = [len(w) for w in words]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0
        self._df = Counter(t for d in self._docs for t in d)
        self._glossary = None

    @property
    def glossary(self):
        """Hebrew/English term pairs learned from these chunks (rag.glossary), built once."""
        if self._glossary is None:
            from rag.glossary import build_glossary
            self._glossary = build_glossary(self.nodes)
        return self._glossary

    def __len__(self) -> int:
        return len(self.nodes)

    def search(self, query: str, top_n: int) -> list[tuple]:
        """Return up to top_n (node, score) pairs with score > 0, best first.

        Each query word counts once - by the best of its forms (the word, without
        its prefix, its stem) - so inflection support adds matches, not weight.
        """
        words = word_forms(query)
        if not words or not self.nodes:
            return []
        n_docs = len(self.nodes)
        scored = []
        for node, tf, length in zip(self.nodes, self._docs, self._lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self._avgdl)
            for forms in words:
                best = 0.0
                for term in forms:
                    freq = tf.get(term)
                    if freq:
                        idf = math.log(1 + (n_docs - self._df[term] + 0.5) / (self._df[term] + 0.5))
                        best = max(best, idf * freq * (self.k1 + 1) / (freq + norm))
                score += best
            if score > 0:
                scored.append((node, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse several ranked id lists: score(id) = sum of 1 / (k + rank).

    Each id appears once in the result (no duplicates). Ties keep the order
    of first appearance, with the first ranking (vector search) first.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if item not in first_seen:
                first_seen[item] = position
                position += 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
