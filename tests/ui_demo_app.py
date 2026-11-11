"""UI demo / manual-check mode - runs the real app.py with FAKE models.

    streamlit run tests/ui_demo_app.py

* No OpenAI calls and no cost: the embedding model and the LLM are replaced
  with local fakes, and the key in this process is overwritten with a fake
  value, so even a real key in .env cannot be used.
* knowledge_base/ is only read. The index and uploaded files go to a
  temporary folder (STUDYMATE_DEMO_DIR, default: %TEMP%/studymate_demo),
  so the real storage/ and uploads/ folders are never touched.
* Answers are NOT real - they only show that the UI, upload and sources work.
"""

import os
import runpy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (loads .env - overwritten right below)
from rag import index_manager, query_engine, summarizer  # noqa: E402
from tests.conftest import FakeLLM, KeywordEmbedding  # noqa: E402

# Fake value shaped like a key (so the UI doesn't warn "incomplete key");
# it is never used - both OpenAI factories are replaced below.
os.environ["OPENAI_API_KEY"] = "sk-demo-mode-fake-key-not-real-" + "0" * 20
os.environ["STUDYMATE_DEMO_MODE"] = "1"   # shows a "demo mode" badge in the UI

demo_dir = Path(os.environ.get("STUDYMATE_DEMO_DIR")
                or Path(tempfile.gettempdir()) / "studymate_demo")
config.STORAGE_DIR = demo_dir / "storage"
config.UPLOADS_DIR = demo_dir / "uploads"

DEMO_ANSWER = "[מצב הדגמה – תשובה מזויפת, ללא OpenAI] כאן תופיע התשובה האמיתית, והמקור שלה מוצג למטה [1]."
index_manager.create_openai_embed_model = KeywordEmbedding
query_engine.create_openai_llm = lambda: FakeLLM(response_text=DEMO_ANSWER)

# Fake summary: a fixed text whose code lines exist in the Python material, so
# the local code verification keeps them. Not a real summary.
DEMO_SUMMARY = """# סיכום Python (מצב הדגמה – טקסט מזויף, ללא OpenAI)

## רשימות
רשימה (list) היא מבנה נתונים שניתן לשנות [1].

```python
nums = [5, 2, 9]
nums.append(4)      # [5, 2, 9, 4]
```

## הערה
בסיכום אמיתי המודל כותב כאן את כל נושאי החומר [1]."""
summarizer.create_summary_llm = lambda max_tokens: FakeLLM(response_text=DEMO_SUMMARY)

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

