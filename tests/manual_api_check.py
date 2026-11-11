"""Manual end-to-end check with the REAL OpenAI API (costs money - very little).

    python tests/manual_api_check.py            # all questions that have text
    python tests/manual_api_check.py --only 5   # only question 5

Not collected by pytest. What it does:
1. Loads the real index from storage/ (no cost) and indexes only new or
   changed files from knowledge_base/ and uploads/ (embedding cost).
2. Runs at most 5 questions from tests/api_questions.json (LLM cost).
3. Prints answers, retrieved sources and the actual token usage, and saves
   a Markdown report in tests/results/ for the README.

Nothing is sent before you type "כן" (or pass --yes).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llama_index.core import Settings  # noqa: E402
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler  # noqa: E402

import config  # noqa: E402
from rag import topics  # noqa: E402
from rag.errors import friendly_api_error  # noqa: E402
from rag.index_manager import IndexManager  # noqa: E402
from rag.query_engine import RagEngine, RagError, create_openai_llm  # noqa: E402

MAX_QUESTIONS = 5
# USD per 1M tokens - check https://openai.com/api/pricing before running.
PRICE_EMBED = 0.02
PRICE_LLM_IN = 0.15
PRICE_LLM_OUT = 0.60


def main() -> int:
    status = config.api_key_status()
    if status != "ok":
        print(f"OPENAI_API_KEY is {status}. Fix it in .env (see .env.example) - nothing was sent.")
        return 1

    spec = json.loads((ROOT / "tests" / "api_questions.json").read_text(encoding="utf-8"))
    questions = [q for q in spec["questions"] if q["question"].strip()]
    # --only 5  (or --only 1,3) runs just those question ids - avoids paying again for the rest
    if "--only" in sys.argv:
        wanted = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            print(f"Question(s) {sorted(missing)} have no text in tests/api_questions.json - nothing was sent.")
            return 1
    questions = questions[:MAX_QUESTIONS]

    print(f"Questions to run: {len(questions)} (max {MAX_QUESTIONS})")
    print(f"Models: {config.EMBED_MODEL} + {config.LLM_MODEL}, Top K = {config.TOP_K}, "
          f"hybrid={config.HYBRID_SEARCH}")
    print("Worst case per question: ~2,000 input + 600 output tokens  ->  < $0.001")
    print("New documents are embedded once (~$0.02 per 1M tokens).")
    if "--yes" not in sys.argv:
        try:
            answer = input('Type "כן" to send requests to OpenAI: ').strip()
        except EOFError:
            answer = ""
        if answer not in ("כן", "yes"):
            print("Cancelled - nothing was sent.")
            return 0

    counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([counter])

    # --strict: no automatic retries, and stop at the first failure (no further calls).
    strict = "--strict" in sys.argv
    make_llm = create_openai_llm
    if strict:
        import rag.index_manager as index_manager_module
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        index_manager_module.create_openai_embed_model = lambda: OpenAIEmbedding(
            model=config.EMBED_MODEL, max_retries=0, timeout=60)

        def make_llm():
            return OpenAI(model=config.LLM_MODEL, temperature=0, max_tokens=600,
                          max_retries=0, timeout=60)
        print("Strict mode: max_retries=0, stop at first failure.")

    manager = IndexManager()
    if manager.load_or_create() is None:
        print(manager.load_error)
        return 1
    print("Index loaded from disk:", manager.loaded_from_storage)

    for folder, source in ((config.KNOWLEDGE_BASE_DIR, "knowledge_base"),
                           (config.UPLOADS_DIR, "uploads")):
        for result in manager.sync_folder(folder, source, stop_on_error=strict):
            print(f"  [{result.status}] {result.file_name}: {result.message}")
            if strict and result.status == "error":
                print("STOPPED (strict): indexing failed - no question was sent.")
                return 1
    stats = manager.stats()
    print(f"Index: {stats['num_documents']} documents, {stats['num_chunks']} chunks")
    embed_tokens_indexing = counter.total_embedding_token_count

    keyword_index = manager.keyword_index()
    engine = RagEngine(manager.index, llm=make_llm(), keyword_index=keyword_index)
    lines = [f"# API test run – {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Index: {stats['num_documents']} documents, {stats['num_chunks']} chunks. "
             f"Models: {config.EMBED_MODEL}, {config.LLM_MODEL}. Top K = {config.TOP_K}.", ""]

    for q in questions:
        print("\n" + "=" * 70)
        print(f"Q{q['id']} ({q['type']}): {q['question']}")
        lines += [f"## שאלה {q['id']} – {q['type']}", "", f"**שאלה:** {q['question']}", ""]
        # Same search text as app.py (local, no extra call): request phrasing removed,
        # topic names and the material's glossary terms added. The model still gets
        # the original question.
        search_text = topics.retrieval_query(q["question"], glossary=keyword_index.glossary)
        print(f"Search text: {search_text}")
        try:
            result = engine.ask(q["question"], search_query=search_text)
        except RagError as error:
            print("ERROR:", error)
            lines += [f"**שגיאה:** {error}", ""]
            if strict:
                print("STOPPED (strict): no further calls.")
                return 1
            continue
        except Exception as error:  # unexpected - still never print the key
            print("ERROR:", friendly_api_error(error))
            lines += [f"**שגיאה:** {friendly_api_error(error)}", ""]
            continue

        retrieved_names = [s.file_name for s in result.retrieved]
        checks = []
        if q["expect_no_answer"]:
            checks.append(("refusal", not result.found))
        elif q.get("expected_source"):
            checks.append(("expected source retrieved", q["expected_source"] in retrieved_names))
            checks.append(("answer found", result.found))

        print("ANSWER:", result.answer)
        lines += [f"**תשובה:** {result.answer}", "", "**מקורות שאוחזרו (Top K):**", ""]
        for s in result.retrieved:
            used = "✔ צוטט" if s in result.sources else ""
            page = f", עמוד {s.page}" if s.page else ""
            score = f"{s.score:.3f}" if s.score is not None else "-"
            ranks = f"vector_rank={s.vector_rank} bm25_rank={s.keyword_rank}"
            print(f"  [{s.number}] {s.file_name}{page}  score={score} {ranks} {used}")
            print("      " + s.text[:200].replace("\n", " "))
            lines.append(f"- [{s.number}] `{s.file_name}`{page} – ציון דמיון {score}, "
                         f"דירוג וקטורי {s.vector_rank}, דירוג BM25 {s.keyword_rank} {used}")
        lines += ["", f"**תשובה צפויה:** {q['expected_answer']}", ""]
        for name, ok in checks:
            print(f"  CHECK {name}: {'OK' if ok else 'FAILED'}")
            lines.append(f"- בדיקה אוטומטית – {name}: {'עבר' if ok else 'נכשל'}")
        lines.append("")

    # Free checks: reload from disk + re-sync must not embed anything, and the
    # index must contain exactly the chunks listed in the manifest.
    tokens_before = counter.total_embedding_token_count
    reloaded = IndexManager()
    reloaded.load_or_create()
    new_results = []
    for folder, source in ((config.KNOWLEDGE_BASE_DIR, "knowledge_base"),
                           (config.UPLOADS_DIR, "uploads")):
        new_results += reloaded.sync_folder(folder, source)
    reembedded = counter.total_embedding_token_count - tokens_before
    expected_chunks = sum(e["num_chunks"] for e in reloaded.manifest.values())
    stored = reloaded.all_nodes()
    unique_chunks = {(n.ref_doc_id, n.text) for n in stored}
    integrity = (f"Reload: loaded_from_storage={reloaded.loaded_from_storage}, "
                 f"new files indexed={sum(r.status == 'added' for r in new_results)}, "
                 f"re-embedded tokens={reembedded}. "
                 f"Chunks in ChromaDB={reloaded.num_vectors()}, in manifest={expected_chunks}, "
                 f"unique={len(unique_chunks)}")
    print("\n" + integrity)
    lines += ["## בדיקת טעינה מחדש וכפילויות", "", integrity, ""]

    embed = counter.total_embedding_token_count
    llm_in = counter.prompt_llm_token_count
    llm_out = counter.completion_llm_token_count
    cost = (embed * PRICE_EMBED + llm_in * PRICE_LLM_IN + llm_out * PRICE_LLM_OUT) / 1e6
    usage = (f"Tokens – embeddings: {embed} (of which indexing: {embed_tokens_indexing}), "
             f"LLM input: {llm_in}, LLM output: {llm_out}. Estimated cost: ${cost:.5f}")
    print("\n" + usage)
    lines += ["## שימוש בטוקנים", "", usage, ""]

    out_dir = ROOT / "tests" / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"api_run_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Report saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
