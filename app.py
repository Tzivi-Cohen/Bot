"""StudyMate AI - Streamlit interface.

Layout (desktop): the study library in the wide main area, the AI assistant in
a sticky panel on the right. On small screens the panels stack.
Navigation: folders -> files -> file content. The RAG engine (rag/) is used
exactly as before; this file only builds the interface.
"""

import html
import os
from pathlib import Path

import streamlit as st

import config
from rag.document_loader import DocumentLoadError, compute_file_hash, extract_documents
from rag.index_manager import IndexManager
from rag import summarizer, topics
from rag.query_engine import RagEngine, RagError, create_openai_llm
from ui import library as lib
from ui.styles import CSS

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = [
    "מה ההבדל בין / ל־// ב־Python?",
    "באילו פינים של Arduino UNO אפשר להשתמש ב־analogWrite?",
    "מה זה Member Initializer List ב־C++?",
    "סכמי לי את החומר ב־Python",
]
FILE_ICONS = {".pdf": "📕", ".docx": "📘", ".txt": "📄"}
STATUS_ICONS = {"indexed": "✅", "pending": "🕓", "updating": "🔄", "problem": "⚠️"}


# ------------------------------------------------------------------ backend
@st.cache_resource(show_spinner="טוען את מאגר הידע…")
def get_manager() -> IndexManager:
    """Load the saved index once per server process.

    Only new or changed files are embedded during sync; files that are
    already in the index (same hash) are skipped, so a restart costs nothing.
    """
    manager = IndexManager()
    manager.load_or_create()
    if manager.index is not None and config.has_api_key():
        manager.sync_folder(config.KNOWLEDGE_BASE_DIR, "knowledge_base")
        manager.sync_folder(config.UPLOADS_DIR, "uploads")
    return manager


@st.cache_resource
def get_llm():
    return create_openai_llm()


def ltr(text: str) -> str:
    """Isolate a Latin name inside Hebrew text, so "C++" isn't shown as "++C"."""
    return f"⁦{text}⁩"


@st.cache_data(show_spinner=False)
def local_file_problem(path_str: str, file_hash: str) -> str | None:
    """Free, local check (no API): can text be extracted from this file?

    Cached by content hash, so each file is parsed only once.
    """
    try:
        extract_documents(Path(path_str), file_hash=file_hash)
    except DocumentLoadError as error:
        return str(error)
    return None


@st.cache_data(show_spinner=False)
def read_document(path_str: str, file_hash: str) -> str:
    """Text of a document for viewing/searching (local extraction, cached by hash)."""
    docs = extract_documents(Path(path_str), file_hash=file_hash)
    if len(docs) == 1:
        return docs[0].text
    return "\n\n".join(f"עמוד {d.metadata.get('page')}\n{'-' * 10}\n{d.text}" for d in docs)


def read_text(path: Path) -> str:
    return read_document(str(path), compute_file_hash(path))


# ------------------------------------------------------------------ state
ss = st.session_state
for name, default in [("view", "home"), ("folder", None), ("file", None), ("messages", []),
                      ("focus", None), ("pending_q", None), ("upload_results", []),
                      ("summary_request", None), ("summary_go", False), ("force_question", False),
                      ("summary_shown", None), ("last_topic", None), ("last_question", None)]:
    ss.setdefault(name, default)


def go_home():
    ss.view, ss.folder, ss.file = "home", None, None


def open_folder(key):
    ss.view, ss.folder, ss.file = "folder", key, None


def open_file(path_key, folder_key):
    ss.view, ss.folder, ss.file = "file", folder_key, path_key
    ss.search = ""


def back():
    if ss.view == "file":
        open_folder(ss.folder)
    elif ss.view == "summary" and ss.folder:
        open_folder(ss.folder)
    else:
        go_home()


def set_focus(path_key, name, folder_key):
    ss.focus = {"path": path_key, "name": name, "folder": folder_key}


def clear_focus():
    ss.focus = None


def ask(question):
    ss.pending_q = question


def request_summary(scope, key, topic=None):
    """Only prepares a local plan (tokens + cost). Nothing is sent before approval.
    scope "subtopic": key = what to summarize ("סריקות עצים"), topic = its folder or None."""
    ss.summary_request = {"scope": scope, "key": key, **({"topic": topic} if topic else {})}
    ss.summary_go = False


def cancel_summary():
    ss.summary_request, ss.summary_go = None, False


def confirm_summary():
    ss.summary_go = True


def send_as_question(text):
    ss.pending_q, ss.force_question = text, True


def show_summary(index):
    ss.view, ss.summary_shown = "summary", index
    ss.search = ""


# ------------------------------------------------------------------ data
manager = get_manager()
api_ready = config.has_api_key()
index_ready = manager.index is not None
catalog = lib.build_catalog(manager, config.KNOWLEDGE_BASE_DIR, config.UPLOADS_DIR,
                            lambda path, h: local_file_problem(str(path), h))
folders = {f.key: f for f in catalog}
files_by_key = {item.key: item for f in catalog for item in f.files}


def find_source_file(file_name, topic):
    """The library file a chat source came from (by name and topic folder)."""
    for item in files_by_key.values():
        if item.name == file_name and (item.folder_key == topic or not topic):
            return item
    return None


# ------------------------------------------------------------------ pieces
def status_badge(item) -> str:
    return (f'<span class="sm-badge {item.status}">{STATUS_ICONS[item.status]} '
            f'{html.escape(item.status_text)}</span>')


def topic_badge(folder_key) -> str:
    title, icon, _, _ = lib.folder_info(folder_key)
    return f'<span class="sm-badge topic">{icon} {lib.bidi_html(title)}</span>'


def file_row(item, row_key, snippet=None, show_topic=False, reason=None):
    with st.container(key=f"file_row_{row_key}"):
        text_col, button_col = st.columns([5, 1.2], vertical_alignment="center")
        with text_col:
            extra = f" {topic_badge(item.folder_key)}" if show_topic else ""
            if reason:
                extra += f' <span class="sm-badge reason">🔎 נמצא {html.escape(reason)}</span>'
            snippet_html = (f'<div class="sm-snippet">{lib.bidi_html(snippet)}</div>'
                            if snippet else "")
            st.markdown(
                f'<div class="sm-file"><div class="sm-file-icon">'
                f'{FILE_ICONS.get(item.path.suffix.lower(), "📄")}</div><div>'
                f'<div class="sm-file-name">{lib.bidi_html(item.name)}</div>'
                f'<div class="sm-file-sub">{status_badge(item)}{extra}</div>{snippet_html}'
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with button_col:
            st.button("פתיחה", key=f"file_open_{row_key}", on_click=open_file,
                      args=(item.key, item.folder_key), use_container_width=True)


def breadcrumbs(parts):
    crumbs = ' <span>›</span> '.join(
        f"<b>{lib.bidi_html(p)}</b>" if i == len(parts) - 1 else lib.bidi_html(p)
        for i, p in enumerate(parts))
    back_col, crumb_col = st.columns([1, 6], vertical_alignment="center")
    with back_col:
        st.button("→ חזרה", key="nav_back", on_click=back, use_container_width=True)
    with crumb_col:
        st.markdown(f'<div class="sm-crumbs">{crumbs}</div>', unsafe_allow_html=True)


def render_document(text, language):
    """Formatted study text: headings, paragraphs, lists, tables, and LTR code blocks."""
    buffer = []

    def flush():
        if buffer:
            st.markdown(f'<div class="sm-doc">{"".join(buffer)}</div>', unsafe_allow_html=True)
            buffer.clear()

    for block in lib.parse_study_text(text, language):
        kind = block[0]
        if kind == "title":
            tag = "h1" if block[2] == 1 else "h2"
            buffer.append(f"<{tag}>{lib.bidi_html(block[1])}</{tag}>")
        elif kind == "para":
            buffer.append("<p>" + "<br>".join(lib.line_html(line) for line in block[1]) + "</p>")
        elif kind == "bullets":
            buffer.append("<ul>" + "".join(f"<li>{lib.line_html(b)}</li>" for b in block[1]) + "</ul>")
        elif kind == "numbered":
            buffer.append("<ol>" + "".join(f"<li>{lib.line_html(b)}</li>" for b in block[1]) + "</ol>")
        elif kind == "table":
            rows = block[1]
            head = "".join(f"<th>{lib.bidi_html(c)}</th>" for c in rows[0]) if rows else ""
            body = "".join("<tr>" + "".join(f"<td>{lib.bidi_html(c)}</td>" for c in r) + "</tr>"
                           for r in rows[1:])
            buffer.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif kind == "code":
            flush()
            st.code(block[1], language=block[2])
    flush()


def render_sources(sources, message_index):
    with st.expander(f"מקורות ({len(sources)})"):
        for j, source in enumerate(sources):
            topic = f" · {topic_badge(source.topic)}" if source.topic else ""
            page = f" · עמוד {source.page}" if source.page is not None else ""
            st.markdown(f'<div class="sm-source">📄 {lib.bidi_html(source.file_name)}{page}{topic}</div>'
                        f'<div class="sm-source-snippet">{lib.bidi_html(source.text[:220])}…</div>',
                        unsafe_allow_html=True)
            item = find_source_file(source.file_name, source.topic)
            if item:
                st.button("פתיחת המסמך", key=f"src_{message_index}_{j}", on_click=open_file,
                          args=(item.key, item.folder_key), type="tertiary")


def make_summary_llm(max_tokens):
    return summarizer.create_summary_llm(max_tokens)


def summary_plan(request):
    if request["scope"] == "topic":
        return summarizer.plan_topic_summary(manager, request["key"])
    if request["scope"] == "subtopic":
        return summarizer.plan_subtopic_summary(manager, request["key"], request.get("topic"))
    return summarizer.plan_file_summary(manager, request["key"])


def summary_title(plan) -> str:
    if plan.scope == "topic":
        return f"סיכום הנושא: {lib.folder_info(plan.title)[0]}"
    if plan.scope == "subtopic":
        where = f" ({lib.folder_info(plan.topic)[0]})" if plan.topic else ""
        return f"סיכום הנושא: {plan.title}{where}"
    return f"סיכום המסמך: {plan.title}"


def render_summary_sources(sources, message_index):
    with st.expander(f"מקורות הסיכום ({len(sources)})"):
        for j, source in enumerate(sources):
            topic = f" · {topic_badge(source['topic'])}" if source["topic"] else ""
            st.markdown(f'<div class="sm-source">[{source["number"]}] 📄 '
                        f'{lib.bidi_html(source["file_name"])}{topic}</div>', unsafe_allow_html=True)
            item = find_source_file(source["file_name"], source["topic"] or lib.UPLOADS_KEY)
            if item:
                st.button("פתיחת המסמך", key=f"sumsrc_{message_index}_{j}", on_click=open_file,
                          args=(item.key, item.folder_key), type="tertiary")


def render_summary_message(message, index, in_chat=True):
    st.markdown(lib.isolate_plus_names(message["content"]))
    for warning in message.get("warnings", []):
        st.caption(f"⚠️ {warning}")
    render_summary_sources(message["summary_sources"], index if in_chat else f"v{index}")
    if in_chat and message.get("found"):
        st.button("📖 הצגה מלאה בספרייה", key=f"show_summary_{index}", on_click=show_summary,
                  args=(index,), type="tertiary")


# ------------------------------------------------------------------ top bar
indexed_total = sum(f.indexed_count for f in catalog)
attention = [i for f in catalog for i in f.files if i.status != "indexed"]
key_status = config.api_key_status()
documents_in_store = len(manager.manifest) if index_ready else 0
topics_count = sum(1 for f in catalog if f.files)
files_total = sum(len(f.files) for f in catalog)
documents_text = lib.count_he(documents_in_store, "מסמך אחד", "מסמכים", zero="אין מסמכים")
topics_text = lib.count_he(topics_count, "נושא אחד", "נושאים", zero="אין נושאים")
if not index_ready:
    pill = ("bad", "שגיאה בטעינת מאגר הידע")
elif key_status == "missing":
    pill = ("warn", f"מצב צפייה בלבד • {documents_text} • {topics_text}")
elif key_status == "incomplete":
    pill = ("warn", f"כדאי לבדוק את מפתח ה־API • {documents_text} • {topics_text}")
elif attention:
    pill = ("warn", lib.count_he(len(attention), "קובץ אחד דורש טיפול", "קבצים דורשים טיפול"))
else:
    pill = ("ok", f"מאגר הידע מוכן • {documents_text} • {topics_text}")

brand_col, actions_col = st.columns([3, 2], vertical_alignment="center")
with brand_col:
    demo = (' <span class="sm-pill warn">מצב הדגמה – מודלים מזויפים</span>'
            if os.environ.get("STUDYMATE_DEMO_MODE") else "")
    st.markdown(
        '<div class="sm-brand"><div class="sm-logo">📚</div><div>'
        '<div class="sm-brand-title">StudyMate AI</div>'
        f'<div class="sm-brand-sub">מרכז הלימוד שלך – תכנות, שבבים וחומרה{demo}</div>'
        '</div></div><a class="sm-jump" href="#sm-chat">💬 שאלי את הבוט</a>',
        unsafe_allow_html=True)
with actions_col:
    pill_col, status_col, upload_col = st.columns([3.2, 1.3, 1.5], vertical_alignment="center")
    with pill_col:
        st.markdown(f'<div style="text-align:left"><span class="sm-pill {pill[0]}">'
                    f'<span class="sm-dot"></span>{pill[1]}</span></div>', unsafe_allow_html=True)
    with status_col:
        with st.popover("סטטוס", use_container_width=True):
            st.markdown("**מצב המערכת**")
            for message in manager.recovery_messages:
                st.info(f"שחזור אוטומטי: {message}")
            if not index_ready:
                st.error(manager.load_error)
            else:
                m1, m2 = st.columns(2)
                m1.metric("מסמכים במאגר", documents_in_store)
                m2.metric("נושאים", topics_count)
                if manager.loaded_from_storage:
                    st.success("מאגר הידע נטען בהצלחה.")
                else:
                    st.info("נוצר מאגר ידע חדש.")
            if key_status == "missing":
                st.warning(lib.bidi_text("לא הוגדר מפתח OPENAI_API_KEY. יש להוסיף אותו לקובץ .env "
                                         "(לפי הדוגמה בקובץ .env.example) ולהפעיל את האתר מחדש."))
            elif key_status == "incomplete":
                st.warning(lib.bidi_text("המפתח בקובץ .env נראה לא שלם (קצר מדי, או שיש בו רווחים או מרכאות). "
                                         "יש להדביק את המפתח המלא ולהפעיל את האתר מחדש."))
            if attention:
                st.markdown("**קבצים שדורשים טיפול**")
                for item in attention[:10]:
                    st.markdown(f"{STATUS_ICONS[item.status]} {lib.bidi_html(item.name)} — "
                                f"{html.escape(item.status_text)}", unsafe_allow_html=True)
                if len(attention) > 10:
                    st.caption(f"ועוד {len(attention) - 10} קבצים.")
    with upload_col:
        with st.popover("⬆ העלאת קובץ", use_container_width=True):
            st.markdown("**הוספת מסמך לספרייה**")
            st.caption("אפשר להעלות קובצי PDF עם טקסט, DOCX או TXT. "
                       "הקבצים נשמרים בתיקייה uploads ואינם עולים ל־GitHub.")
            uploaded_files = st.file_uploader("בחירת קבצים", type=["pdf", "docx", "txt"],
                                              accept_multiple_files=True, label_visibility="collapsed")
            can_index = index_ready and api_ready
            if st.button("הוספה למאגר", type="primary", disabled=not (uploaded_files and can_index),
                         use_container_width=True):
                results = []
                for uploaded_file in uploaded_files:
                    with st.spinner(f"מעבד את {uploaded_file.name}…"):
                        results.append(manager.add_uploaded_file(uploaded_file.name,
                                                                 uploaded_file.getvalue()))
                ss.upload_results = [(r.status, r.file_name, r.message) for r in results]
                st.rerun()
            if not can_index:
                st.caption("כדי להוסיף מסמכים צריך מאגר ידע תקין ומפתח API.")

st.write("")

# ------------------------------------------------------------------ layout
chat_col, library_col = st.columns([1.05, 2.15], gap="large")   # RTL: first column is on the right

# ---------------------------------------------------------------- library
with library_col:
    with st.container(key="library_panel"):
        head_col, search_col = st.columns([1.4, 1], vertical_alignment="bottom")
        with head_col:
            st.markdown('<div class="sm-section-title">ספריית הלימוד שלי</div>'
                        f'<div class="sm-section-sub">{topics_text} • '
                        f'{lib.count_he(files_total, "מסמך אחד", "מסמכים", zero="אין מסמכים")}</div>',
                        unsafe_allow_html=True)
        with search_col:
            query = st.text_input("חיפוש", key="search",
                                  placeholder="🔍 חיפוש לפי נושא, שם קובץ או מושג…",
                                  label_visibility="collapsed")

        for status, name, message in ss.upload_results:
            show = {"added": st.success, "duplicate": st.warning}.get(status, st.error)
            show(f"{name}: {message}")
        if ss.upload_results and st.button("סגירת ההודעות", key="clear_uploads", type="tertiary"):
            ss.upload_results = []
            st.rerun()

        if query.strip():
            options = ["__all__"] + [f.key for f in catalog]
            chosen = st.pills("סינון לפי תיקייה", options, default="__all__", key="search_filter",
                              label_visibility="collapsed",
                              format_func=lambda k: "הכול" if k == "__all__"
                              else lib.bidi_text(folders[k].title))
            folder_key = None if chosen in (None, "__all__") else chosen
            results = lib.search_files(catalog, query, read_text, folder_key)
            count_text = lib.count_he(len(results), "תוצאה אחת", "תוצאות", zero="אין תוצאות")
            st.markdown(f'<div class="sm-section-sub">{count_text} עבור '
                        f"„{lib.bidi_html(query.strip())}”</div>", unsafe_allow_html=True)
            for topic_key in topics.detect_topics(query):
                if topic_key in folders and (not folder_key or folder_key == topic_key):
                    st.button(lib.bidi_text(f"📁 מעבר לתיקייה {folders[topic_key].title}"),
                              key=f"goto_topic_{lib.TOPIC_ORDER.index(topic_key)}",
                              on_click=open_folder, args=(topic_key,), type="tertiary")
            if not results:
                st.info("לא נמצאו מסמכים מתאימים. אפשר לנסות מילה אחרת או לשאול את הבוט.")
            for i, (item, snippet, reason) in enumerate(results):
                file_row(item, f"search_{i}", snippet=snippet, show_topic=True, reason=reason)

        elif (ss.view == "summary" and ss.summary_shown is not None
              and ss.summary_shown < len(ss.messages)):
            message = ss.messages[ss.summary_shown]
            breadcrumbs(["ספריית הלימוד", message["title"]])
            with st.container(key="summary_view"):
                st.markdown(f'<div class="sm-doc-head"><div class="sm-folder-icon">📝</div><div>'
                            f'<div class="sm-doc-title">{lib.bidi_html(message["title"])}</div>'
                            f'<div class="sm-file-sub">סיכום שנוצר מחומרי הלימוד בלבד • '
                            f'{lib.count_he(len(message["summary_sources"]), "מקור אחד", "מקורות")}'
                            "</div></div></div>",
                            unsafe_allow_html=True)
                render_summary_message(message, ss.summary_shown, in_chat=False)

        elif ss.view == "file" and ss.file in files_by_key:
            item = files_by_key[ss.file]
            folder = folders[item.folder_key]
            breadcrumbs(["ספריית הלימוד", folder.title, item.name])
            with st.container(key="doc_view"):
                st.markdown(
                    f'<div class="sm-doc-head"><div class="sm-folder-icon">{folder.icon}</div><div>'
                    f'<div class="sm-doc-title">{lib.bidi_html(item.name)}</div>'
                    f'<div class="sm-file-sub">{topic_badge(item.folder_key)} {status_badge(item)}</div>'
                    "</div></div>", unsafe_allow_html=True)
                ask_col, sum_col, _ = st.columns([1.6, 1.4, 2], vertical_alignment="center")
                with ask_col:
                    st.button("🤖 שאלי את הבוט על המסמך", key="focus_file", on_click=set_focus,
                              args=(item.key, item.name, item.folder_key), use_container_width=True)
                with sum_col:
                    st.button("📝 סכמי את המסמך", key="summarize_file", on_click=request_summary,
                              args=("file", item.key), use_container_width=True,
                              disabled=item.status != "indexed",
                              help=None if item.status == "indexed"
                              else "אפשר לסכם רק מסמך שנמצא במאגר הידע.")
                try:
                    render_document(read_text(item.path), folder.language)
                except DocumentLoadError as error:
                    st.warning(str(error))

        elif ss.view == "folder" and ss.folder in folders:
            folder = folders[ss.folder]
            breadcrumbs(["ספריית הלימוד", folder.title])
            st.markdown(
                f'<div class="sm-doc-head"><div class="sm-folder-icon">{folder.icon}</div><div>'
                f'<div class="sm-doc-title">{lib.bidi_html(folder.title)}</div>'
                f'<div class="sm-file-sub">{lib.bidi_html(folder.description)} • '
                f'{lib.count_he(len(folder.files), "מסמך אחד", "מסמכים", zero="אין מסמכים")}'
                "</div></div></div>", unsafe_allow_html=True)
            if folder.indexed_count:
                st.button("📝 סכמי את הנושא", key="summarize_folder",
                          on_click=request_summary, args=("topic", folder.key))
            if not folder.files:
                st.info("התיקייה ריקה. אפשר להוסיף אליה קבצים במחשב, או להעלות קובץ דרך כפתור ההעלאה.")
            for i, item in enumerate(folder.files):
                file_row(item, f"folder_{i}")

        else:
            for start in range(0, len(catalog), 4):
                cols = st.columns(4, gap="medium")
                for col, folder in zip(cols, catalog[start:start + 4]):
                    with col, st.container(key=f"folder_card_{start}_{catalog.index(folder)}"):
                        meta = lib.count_he(len(folder.files), "מסמך אחד", "מסמכים", zero="תיקייה ריקה")
                        st.markdown(
                            f'<div class="sm-folder-icon">{folder.icon}</div>'
                            f'<div class="sm-folder-title">{lib.bidi_html(folder.title)}</div>'
                            f'<div class="sm-folder-desc">{lib.bidi_html(folder.description)}</div>'
                            f'<div class="sm-folder-meta">{meta}</div>', unsafe_allow_html=True)
                        st.button("פתיחת התיקייה", key=f"open_folder_{catalog.index(folder)}",
                                  on_click=open_folder, args=(folder.key,))

# ------------------------------------------------------------------ chat
with chat_col:
    with st.container(key="chat_panel"):
        st.markdown('<div id="sm-chat" class="sm-bot-head"><div class="sm-bot-avatar">🤖</div><div>'
                    '<div class="sm-bot-title">העוזר החכם</div>'
                    '<div class="sm-bot-sub">עונה מתוך חומרי הלימוד בלבד, ומציג מקורות</div>'
                    "</div></div>", unsafe_allow_html=True)

        if not index_ready:
            blocked_reason = "מאגר הידע אינו זמין כרגע."
        elif not api_ready:
            blocked_reason = lib.bidi_text("כדי לשאול שאלות צריך להגדיר מפתח OpenAI בקובץ .env.")
        elif documents_in_store == 0:
            blocked_reason = "עדיין אין מסמכים במאגר הידע. אפשר להוסיף מסמכים דרך כפתור ההעלאה."
        else:
            blocked_reason = None

        if ss.focus:
            focus_title = lib.folder_info(ss.focus["folder"])[0]
            chip_col, x_col = st.columns([6, 1], vertical_alignment="center")
            with chip_col:
                st.markdown(f'<div class="sm-focus">🎯 שאלות על: {lib.bidi_html(ss.focus["name"])}'
                            "</div>", unsafe_allow_html=True)
            with x_col:
                st.button("✕", key="clear_focus", on_click=clear_focus, help="ביטול ההתמקדות במסמך")

        messages_box = st.container(height=460, key="chat_messages")
        with messages_box:
            if not ss.messages:
                st.markdown('<div class="sm-empty-chat"><div class="big">💬</div>'
                            "אפשר לשאול כל שאלה על חומרי הלימוד או לבקש סיכום של נושא."
                            "<br>התשובות מבוססות על המסמכים בלבד.</div>",
                            unsafe_allow_html=True)
                if not blocked_reason:
                    for i, suggestion in enumerate(SUGGESTIONS):
                        st.button(lib.bidi_text(suggestion), key=f"suggest_{i}", on_click=ask,
                                  args=(suggestion,))
            for i, message in enumerate(ss.messages):
                with st.chat_message(message["role"]):
                    if message.get("error"):
                        st.error(message["error"])
                    elif message.get("summary"):
                        render_summary_message(message, i)
                    else:
                        st.markdown(lib.isolate_plus_names(message["content"]))
                    if message.get("sources"):
                        render_sources(message["sources"], i)
                    if message.get("choose") is not None and i == len(ss.messages) - 1:
                        options = message["choose"] or [f.key for f in catalog if f.indexed_count]
                        for topic in options:
                            st.button(lib.bidi_text(f"📝 סיכום: {lib.folder_info(topic)[0]}"),
                                      key=f"choose_{i}_{topic}", on_click=request_summary,
                                      args=("topic", topic))
                        st.button("💬 זו שאלה רגילה – שליחה לבוט", key=f"as_question_{i}",
                                  on_click=send_as_question, args=(message["original"],),
                                  type="tertiary")

        if ss.summary_request:
            plan = summary_plan(ss.summary_request)
            with st.container(key="summary_card"):
                st.markdown(f'<div class="sm-plan-title">📝 {lib.bidi_html(summary_title(plan))}</div>',
                            unsafe_allow_html=True)
                if plan.usable:
                    if plan.scope == "topic":
                        found_text = "נמצאו " + lib.count_he(len(plan.sources), "מסמך אחד", "מסמכים") + " בנושא."
                        if len(plan.sources) == 1:
                            found_text = "נמצא מסמך אחד בנושא."
                    elif plan.scope == "subtopic":
                        found_text = ("נמצא חומר מתאים במסמך אחד." if len(plan.sources) == 1 else
                                      f"נמצא חומר מתאים ב־{len(plan.sources)} מסמכים.")
                    else:
                        found_text = "המסמך נמצא במאגר הידע."
                    stages = ("<br>החומר ארוך, ולכן הסיכום יוכן בכמה שלבים."
                              if plan.mode == "staged" else "")
                    st.markdown(
                        f'<div class="sm-plan">{found_text}<br>'
                        f"המערכת מוכנה להכין סיכום של חומר הלימוד.{stages}</div>",
                        unsafe_allow_html=True)
                elif plan.scope == "subtopic" and plan.skipped:
                    st.warning(lib.bidi_text(plan.skipped[0]))   # "לא נמצא במאגר הידע חומר על ..."
                else:
                    st.warning("אין במאגר הידע מסמכים מוכנים לסיכום הזה.")
                for note in ([] if plan.scope == "subtopic" and not plan.usable else plan.skipped):
                    st.caption(f"⚠️ {note}")
                ok_col, cancel_col = st.columns(2)
                with ok_col:
                    st.button("יצירת סיכום", key="confirm_summary", type="primary",
                              on_click=confirm_summary, use_container_width=True,
                              disabled=not plan.usable or blocked_reason is not None)
                with cancel_col:
                    st.button("ביטול", key="cancel_summary", on_click=cancel_summary,
                              use_container_width=True)
                if not api_ready:
                    st.caption(lib.bidi_text("כדי ליצור סיכום צריך להגדיר מפתח OpenAI בקובץ .env."))
                elif blocked_reason:
                    st.caption(blocked_reason)

            if ss.summary_go and plan.usable and not blocked_reason:
                ss.summary_go = False
                ss.messages.append({"role": "user", "content": f"📝 {summary_title(plan)}"})
                with messages_box, st.chat_message("assistant"), \
                        st.spinner("מכין סיכום…"):
                    try:
                        result = summarizer.run_summary(plan, make_llm=make_summary_llm)
                    except summarizer.SummaryError as error:
                        ss.messages.append({"role": "assistant", "content": "", "error": str(error)})
                    else:
                        ss.messages.append({
                            "role": "assistant", "summary": True, "found": result.found,
                            "title": summary_title(plan), "content": result.text,
                            "warnings": result.warnings,
                            "summary_sources": [{"number": s.number, "file_name": s.file_name,
                                                 "topic": s.topic} for s in result.sources],
                        })
                ss.summary_request = None
                if plan.scope == "topic":
                    ss.folder = plan.title
                elif plan.scope == "subtopic" and plan.topic:
                    ss.folder = plan.topic
                st.rerun()

        if blocked_reason:
            st.info(blocked_reason)

        typed = st.chat_input("שאלי שאלה או בקשי סיכום…", key="chat_input",
                              disabled=blocked_reason is not None)
        question = typed or ss.pending_q
        forced = ss.force_question
        ss.pending_q, ss.force_question = None, False

        decision = {"kind": "question"}
        if question and not blocked_reason and not forced:
            focus = ss.focus or {}
            decision = topics.route(question, focus.get("folder"), focus.get("path"))

        if question and not blocked_reason and decision["kind"] == "summary":
            # A summary request only prepares a plan - the paid call waits for approval.
            ss.messages.append({"role": "user", "content": question})
            if decision["scope"] == "subtopic":
                request_summary("subtopic", decision["subject"], decision["topic"])
            else:
                request_summary(decision["scope"],
                                decision["topic"] if decision["scope"] == "topic" else decision["file"])
            ss.messages.append({"role": "assistant",
                                "content": "הבקשה לסיכום מוכנה. אפשר ליצור את הסיכום או לבטל בכרטיס שמתחת לשיחה."})
            st.rerun()

        if question and not blocked_reason and decision["kind"] == "choose_topic":
            ss.messages.append({"role": "user", "content": question})
            ss.messages.append({"role": "assistant", "original": question,
                                "choose": decision["candidates"],
                                "content": "איזה נושא לסכם? אפשר לבחור נושא מהרשימה:"})
            st.rerun()

        if question and not blocked_reason:
            ss.messages.append({"role": "user", "content": question})
            # Search text (local, no LLM call): request phrasing removed, Hebrew topic
            # names mapped to the material's names, and the material's own terms
            # added (fixed table + the glossary learned from the documents).
            # Conversation context: the focused document's topic, or the topic of
            # the previous question; and the previous question itself when this
            # one continues it ("אבל...", "תסבירי את זה שוב").
            keyword_index = manager.keyword_index()
            glossary = keyword_index.glossary
            previous = ss.last_question
            follow_up = bool(previous) and (topics.is_follow_up(question)
                                            or topics.continues_previous(question, previous))
            # The previous topic is a hint only for a question that continues it - a
            # new question ("איך עובדות פסיקות?" after a Python question) must not
            # be searched with "Python" added.
            context_topic = (ss.focus["folder"] if ss.focus and ss.focus["folder"] in lib.TOPICS
                             else ss.last_topic if follow_up else None)
            search_text = topics.retrieval_query(question, context_topic, glossary=glossary,
                                                 previous=previous)
            ss.last_topic = (topics.infer_topic(question, glossary, keyword_index)
                             or (ss.last_topic if follow_up else None))
            ss.last_question = question
            with messages_box:
                with st.chat_message("user"):
                    st.markdown(lib.isolate_plus_names(question))
                with st.chat_message("assistant"), st.spinner("מחפש במסמכים…"):
                    try:
                        result = RagEngine(manager.index, llm=get_llm(),
                                           keyword_index=keyword_index).ask(
                            question, search_query=search_text,
                            previous_question=previous if follow_up else None)
                    except RagError as error:
                        ss.messages.append({"role": "assistant", "content": "", "error": str(error)})
                    else:
                        if result.sources and result.sources[0].topic in lib.TOPICS \
                                and not topics.detect_topics(question):
                            ss.last_topic = result.sources[0].topic   # the topic actually answered from
                        ss.messages.append({"role": "assistant", "content": result.answer,
                                            "sources": result.sources})
            st.rerun()
