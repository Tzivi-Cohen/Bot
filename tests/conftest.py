"""Shared fixtures. Automated tests never call the OpenAI API."""

import pytest
from llama_index.core.embeddings import MockEmbedding
from pydantic import PrivateAttr


@pytest.fixture(autouse=True)
def no_openai_key(monkeypatch):
    """Remove the key so any accidental real API call fails instead of costing money."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class CountingEmbedding(MockEmbedding):
    """Mock embedding that counts how many texts were embedded."""

    _calls: list = PrivateAttr(default_factory=list)

    def _get_text_embedding(self, text):
        self._calls.append(text)
        return super()._get_text_embedding(text)

    def _get_text_embeddings(self, texts):
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)

    @property
    def call_count(self) -> int:
        return len(self._calls)


class FailingEmbedding(MockEmbedding):
    """Simulates an OpenAI API error (e.g. invalid key / quota exceeded)."""

    def _get_text_embedding(self, text):
        raise RuntimeError("simulated API error 401")

    def _get_text_embeddings(self, texts):
        raise RuntimeError("simulated API error 401")

    def _get_query_embedding(self, query):
        raise RuntimeError("simulated API error 401")

    async def _aget_query_embedding(self, query):
        raise RuntimeError("simulated API error 401")


class KeywordEmbedding(MockEmbedding):
    """Deterministic 'semantic' embedding: one dimension per keyword.

    Texts that share keywords get similar vectors, so retrieval can be tested
    meaningfully without calling OpenAI.
    """

    vocab: list = ["ממוצע", "חציון", "שכיח", "סטיית", "הסתברות", "עוגה"]

    def __init__(self, **kwargs):
        super().__init__(embed_dim=7, **kwargs)

    def _vector(self, text):
        # Last dimension is a small constant so no vector is all zeros.
        return [float(text.count(word)) for word in self.vocab] + [0.1]

    def _get_text_embedding(self, text):
        return self._vector(text)

    def _get_query_embedding(self, query):
        return self._vector(query)

    async def _aget_query_embedding(self, query):
        return self._vector(query)

    async def _aget_text_embedding(self, text):
        return self._vector(text)


from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata  # noqa: E402
from llama_index.core.llms.callbacks import llm_completion_callback  # noqa: E402


class FakeLLM(CustomLLM):
    """Returns a fixed answer and records every prompt it received."""

    response_text: str = "תשובה [1]"
    error: Exception | None = None
    _prompts: list = PrivateAttr(default_factory=list)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="fake")

    @property
    def prompts(self) -> list:
        return self._prompts

    @llm_completion_callback()
    def complete(self, prompt, formatted=False, **kwargs):
        self._prompts.append(prompt)
        if self.error:
            raise self.error
        return CompletionResponse(text=self.response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt, formatted=False, **kwargs):
        raise NotImplementedError


class AuthenticationError(Exception):
    """Same class name as openai.AuthenticationError (matched by name)."""


@pytest.fixture
def embed_model():
    return CountingEmbedding(embed_dim=8)


@pytest.fixture
def dirs(tmp_path):
    kb = tmp_path / "knowledge_base"
    uploads = tmp_path / "uploads"
    storage = tmp_path / "storage"
    kb.mkdir()
    return kb, uploads, storage


def make_docx(path, paragraphs):
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(str(path))
    return path


def make_blank_pdf(path):
    """A PDF with no extractable text - behaves like a scanned PDF."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path
