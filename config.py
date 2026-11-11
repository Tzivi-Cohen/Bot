"""Central configuration for StudyMate-AI.

All paths are absolute (relative to this file), so the app works no matter
which directory Streamlit is started from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Verify HTTPS certificates with the operating system's trust store instead of
# Python's bundled list. Needed when antivirus software inspects HTTPS traffic
# (e.g. ESET) - Python 3.13 rejects such certificates even though Windows
# trusts them. Certificate verification stays fully enabled.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # optional - without it, Python's default verification is used
    pass

BASE_DIR = Path(__file__).resolve().parent

# Load OPENAI_API_KEY from .env (if present). Existing environment variables win.
load_dotenv(BASE_DIR / ".env", override=False)

# --- Folders ---------------------------------------------------------------
# Permanent documents, published with the project on GitHub.
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
# Documents uploaded through the UI (private, excluded from Git).
UPLOADS_DIR = BASE_DIR / "uploads"
# Local index: ChromaDB database (storage/chroma) + manifest.json.
# Can be moved with STUDYMATE_STORAGE_DIR (e.g. to a persistent disk when
# deployed in the cloud later).
STORAGE_DIR = Path(os.getenv("STUDYMATE_STORAGE_DIR") or BASE_DIR / "storage")
CHROMA_COLLECTION = "studymate"
# Tracks which files (by SHA-256 hash) are already embedded in the index.
MANIFEST_FILENAME = "manifest.json"

# --- Models ----------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Prices used only for cost ESTIMATES shown before a paid action (USD per 1M
# tokens, gpt-4o-mini). Check https://openai.com/api/pricing - the OpenAI bill
# is the authoritative number.
LLM_PRICE_INPUT_PER_1M = 0.15
LLM_PRICE_OUTPUT_PER_1M = 0.60

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 512      # tokens per chunk
CHUNK_OVERLAP = 50    # tokens shared between consecutive chunks (~10%)

# --- Retrieval -------------------------------------------------------------
TOP_K = 3                 # chunks sent to the LLM (final, after fusion)
# Hybrid search: vector (ChromaDB) + keyword (BM25), fused with RRF.
# Uses the same single query embedding - no extra API calls.
HYBRID_SEARCH = True
HYBRID_CANDIDATES = 20    # candidates taken from each method before fusion
RRF_K = 60                # standard RRF constant

# --- Documents -------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

NO_ANSWER_MESSAGE = "לא מצאתי מידע מספיק במסמכים כדי לענות על השאלה."


def has_api_key() -> bool:
    """True if an OpenAI key is configured. Never returns or logs the key."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def api_key_status() -> str:
    """Shape check only - never returns, prints or logs the key itself.

    "missing"    - no key configured
    "incomplete" - doesn't look like a full OpenAI key (too short, wrong
                   prefix, spaces or quotes - e.g. a partial paste)
    "ok"         - looks like a complete key (validity is only known to OpenAI)
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key.strip():
        return "missing"
    if (not key.startswith("sk-") or len(key) < 40
            or any(c.isspace() or c in "\"'" for c in key)):
        return "incomplete"
    return "ok"
