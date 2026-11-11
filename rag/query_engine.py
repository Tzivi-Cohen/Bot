"""RAG question answering: retrieve Top-K chunks, answer only from them."""

import re
from dataclasses import dataclass, field

from llama_index.core import PromptTemplate, VectorStoreIndex

import config
from rag.errors import friendly_api_error

QA_PROMPT = PromptTemplate(
    "אתה עוזר לימודי. ענה על השאלה אך ורק על סמך המקורות שלהלן.\n"
    "כללים:\n"
    "1. השתמש רק במידע שמופיע במקורות. אל תשתמש בידע כללי ואל תמציא מידע או מקורות.\n"
    "2. אם המקורות אינם מכילים מידע שעונה על השאלה, השב במדויק:\n"
    "\"{no_answer}\"\n"
    "   ואל תוסיף דבר.\n"
    "3. אם המקורות עונים רק על חלק מהשאלה, ענה על החלק הזה בלבד וציין מה לא נמצא במקורות.\n"
    "4. ענה בעברית, בקצרה ובבהירות. דוגמאות קוד מהמקורות יש להעתיק כפי שהן, בתוך בלוק קוד.\n"
    "5. אחרי כל טענה ציין את מספר המקור שעליו היא מבוססת, למשל [1].\n"
    "6. השאלה יכולה להיות מנוסחת בחופשיות, במילים אחרות מאלה שבמקורות (למשל \"אני לא מבינה את...\" "
    "או \"הסבר לי...\"). אם המקורות עוסקים במה ששואלים עליו, הסבר אותו בפשטות ובהדרגה, "
    "על סמך המקורות בלבד.\n"
    "\n"
    "מקורות:\n"
    "{context}\n"
    "\n"
    "{history}"
    "שאלה: {question}\n"
    "תשובה:"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def create_openai_llm():
    """OpenAI chat model. Constructing it makes no API call."""
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=config.LLM_MODEL,
        temperature=0,       # factual answers, no creativity
        max_tokens=600,      # caps the cost of a single answer
        max_retries=2,
        timeout=60,
    )


@dataclass
class Source:
    number: int          # the [n] used in the prompt
    file_name: str
    text: str
    score: float | None  # similarity score - NOT a probability of correctness
    page: int | None = None
    topic: str | None = None  # sub-folder of knowledge_base, e.g. "Arduino"
    vector_rank: int | None = None   # rank in semantic search (None = not in candidates)
    keyword_rank: int | None = None  # rank in BM25 keyword search
    node_id: str | None = None


@dataclass
class RagAnswer:
    answer: str
    found: bool                                   # False -> "no info" answer
    sources: list[Source] = field(default_factory=list)    # used in answer
    retrieved: list[Source] = field(default_factory=list)  # all Top-K results


class RagError(Exception):
    """User-facing error (Hebrew message)."""


def format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        where = f"קובץ: {s.file_name}"
        if s.topic:
            where += f", נושא: {s.topic}"
        if s.page is not None:
            where += f", עמוד {s.page}"
        blocks.append(f"[{s.number}] ({where})\n{s.text}")
    return "\n\n".join(blocks)


def _make_source(number: int, node, score=None, vector_rank=None, keyword_rank=None) -> Source:
    return Source(
        number=number,
        file_name=node.metadata.get("file_name", "לא ידוע"),
        page=node.metadata.get("page"),
        topic=node.metadata.get("topic"),
        text=node.get_content(),
        score=score,
        vector_rank=vector_rank,
        keyword_rank=keyword_rank,
        node_id=node.node_id,
    )


class RagEngine:
    def __init__(self, index: VectorStoreIndex, llm=None, top_k: int = config.TOP_K,
                 keyword_index=None):
        """keyword_index: a rag.hybrid.KeywordIndex over the same chunks.

        When given (and config.HYBRID_SEARCH is on), retrieval is hybrid:
        semantic search + BM25, fused with Reciprocal Rank Fusion.
        Without it, retrieval is semantic only.
        """
        self.index = index
        self.llm = llm or create_openai_llm()
        self.top_k = top_k
        self.keyword_index = keyword_index

    def retrieve(self, question: str) -> list[Source]:
        """Return the Top-K chunks for the question (one embedding call)."""
        hybrid = (config.HYBRID_SEARCH and self.keyword_index is not None
                  and len(self.keyword_index) > 0)
        if not hybrid:
            retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            results = retriever.retrieve(question)
            return [_make_source(i, r.node, r.score, vector_rank=i)
                    for i, r in enumerate(results, start=1)]

        from rag.hybrid import reciprocal_rank_fusion

        candidates = max(config.HYBRID_CANDIDATES, self.top_k)
        vector_results = self.index.as_retriever(
            similarity_top_k=candidates).retrieve(question)
        keyword_results = self.keyword_index.search(question, candidates)

        nodes = {}
        vector_rank, keyword_rank, vector_score = {}, {}, {}
        for rank, r in enumerate(vector_results, start=1):
            nodes.setdefault(r.node.node_id, r.node)
            vector_rank[r.node.node_id] = rank
            vector_score[r.node.node_id] = r.score
        for rank, (node, _score) in enumerate(keyword_results, start=1):
            nodes.setdefault(node.node_id, node)
            keyword_rank[node.node_id] = rank

        fused = reciprocal_rank_fusion(
            [[r.node.node_id for r in vector_results],
             [node.node_id for node, _ in keyword_results]],
            k=config.RRF_K,
        )
        return [
            _make_source(i, nodes[node_id], vector_score.get(node_id),
                         vector_rank.get(node_id), keyword_rank.get(node_id))
            for i, node_id in enumerate(fused[: self.top_k], start=1)
        ]

    def ask(self, question: str, search_query: str | None = None,
            previous_question: str | None = None) -> RagAnswer:
        """search_query: optional text for retrieval only (e.g. with Hebrew topic
        names mapped to the material's terms). The LLM sees the original question.
        previous_question: for a follow-up question - shown to the model only as
        context ("ומה לגבי ירושה?" after a question on constructors)."""
        question = (question or "").strip()
        if not question:
            raise RagError("יש להזין שאלה.")

        try:
            retrieved = self.retrieve((search_query or "").strip() or question)
        except Exception as error:
            raise RagError(friendly_api_error(error)) from error

        if not retrieved:
            return RagAnswer(config.NO_ANSWER_MESSAGE, found=False)

        prompt = QA_PROMPT.format(
            no_answer=config.NO_ANSWER_MESSAGE,
            context=format_context(retrieved),
            history=(f"השאלה הקודמת בשיחה (רק להבנת ההקשר): {previous_question.strip()}\n"
                     if previous_question and previous_question.strip() else ""),
            question=question,
        )
        try:
            answer = str(self.llm.complete(prompt)).strip()
        except Exception as error:
            raise RagError(friendly_api_error(error)) from error

        cited = {int(n) for n in _CITATION_RE.findall(answer)
                 if 1 <= int(n) <= len(retrieved)}
        if not answer or (config.NO_ANSWER_MESSAGE in answer and not cited):
            # A refusal (no source cited): don't present sources.
            # A partial answer that cites sources and also says what is missing is kept.
            return RagAnswer(config.NO_ANSWER_MESSAGE, found=False,
                             retrieved=retrieved)
        used = [s for s in retrieved if s.number in cited] or retrieved
        return RagAnswer(answer, found=True, sources=used, retrieved=retrieved)
