"""Visual design of the StudyMate AI interface (CSS injected into Streamlit).

Elements are targeted with the classes Streamlit creates for keyed
containers/widgets ("st-key-<key>") and with stable data-testid attributes.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;600;700&display=swap');

:root {
  --sm-bg: #F5F6FB;
  --sm-card: #FFFFFF;
  --sm-border: #E5E7F0;
  --sm-text: #1E2235;
  --sm-muted: #6B7186;
  --sm-primary: #5B5BD6;
  --sm-primary-soft: #EEEEFC;
  --sm-primary-dark: #4646B8;
  --sm-ok: #12805C;
  --sm-ok-soft: #E7F6EF;
  --sm-warn: #A15C07;
  --sm-warn-soft: #FDF3E1;
  --sm-bad: #B42318;
  --sm-bad-soft: #FDECEA;
  --sm-radius: 16px;
  --sm-shadow: 0 1px 2px rgba(20, 24, 50, .04), 0 4px 16px rgba(20, 24, 50, .06);
  --sm-shadow-hover: 0 2px 4px rgba(20, 24, 50, .06), 0 10px 28px rgba(70, 70, 184, .14);
}

/* ---------- page ---------- */
html, body, [data-testid="stAppViewContainer"], .stApp {
  font-family: "Rubik", "Segoe UI", "Arial Hebrew", Arial, sans-serif;
  background: var(--sm-bg);
  color: var(--sm-text);
}
[data-testid="stAppViewContainer"] { direction: rtl; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { max-width: 1480px; padding: 1.4rem 2rem 2rem 2rem !important; }
[data-testid="stMarkdownContainer"], [data-testid="stText"], [data-testid="stWidgetLabel"],
[data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stCaptionContainer"],
[data-testid="stChatMessageContent"], .stChatInput textarea { text-align: right; }
h1, h2, h3, h4 { font-family: inherit !important; color: var(--sm-text); }

/* ---------- top bar ---------- */
.sm-brand { display: flex; align-items: center; gap: .8rem; }
.sm-logo {
  width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
  background: linear-gradient(135deg, #6D6BF0, #8E5BE8); color: #fff; font-size: 1.45rem;
  box-shadow: 0 6px 18px rgba(91, 91, 214, .35);
}
.sm-brand-title { font-size: 1.45rem; font-weight: 700; line-height: 1.1; }
.sm-brand-sub { color: var(--sm-muted); font-size: .9rem; }
.sm-pill {
  display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .8rem;
  border-radius: 999px; font-size: .85rem; font-weight: 500; white-space: nowrap;
}
.sm-pill.ok { background: var(--sm-ok-soft); color: var(--sm-ok); }
.sm-pill.warn { background: var(--sm-warn-soft); color: var(--sm-warn); }
.sm-pill.bad { background: var(--sm-bad-soft); color: var(--sm-bad); }
.sm-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

/* ---------- panels ---------- */
.st-key-library_panel, .st-key-chat_panel {
  background: var(--sm-card); border: 1px solid var(--sm-border);
  border-radius: 20px; box-shadow: var(--sm-shadow); padding: 1.3rem 1.4rem 1.2rem;
}
.sm-section-title { font-size: 1.35rem; font-weight: 700; margin: 0; }
.sm-section-sub { color: var(--sm-muted); font-size: .92rem; margin-top: .15rem; }

/* chat column stays visible while the library scrolls (desktop) */
[data-testid="stColumn"]:has(.st-key-chat_panel) { position: sticky; top: 1rem; align-self: flex-start; }

/* ---------- folder cards ---------- */
div[class*="st-key-folder_card_"] {
  background: var(--sm-card); border: 1px solid var(--sm-border); border-radius: var(--sm-radius);
  padding: 1.05rem 1.05rem .7rem; box-shadow: var(--sm-shadow);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; height: 100%;
}
div[class*="st-key-folder_card_"]:hover {
  transform: translateY(-3px); box-shadow: var(--sm-shadow-hover); border-color: #CFCFF5;
}
.sm-folder-icon {
  width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
  font-size: 1.55rem; background: var(--sm-primary-soft); margin-bottom: .6rem;
}
.sm-folder-title { font-weight: 600; font-size: 1.05rem; margin-bottom: .15rem; }
.sm-folder-desc { color: var(--sm-muted); font-size: .84rem; min-height: 2.5em; line-height: 1.35; }
.sm-folder-meta { color: var(--sm-primary-dark); font-size: .82rem; font-weight: 500; margin-top: .45rem; }
div[class*="st-key-folder_card_"] button {
  width: 100%; border-radius: 10px; border: 1px solid var(--sm-border); background: #FAFAFE;
  color: var(--sm-primary-dark); font-weight: 500;
}
div[class*="st-key-folder_card_"] button:hover { background: var(--sm-primary); color: #fff; border-color: var(--sm-primary); }

/* ---------- file rows ---------- */
div[class*="st-key-file_row_"] {
  background: var(--sm-card); border: 1px solid var(--sm-border); border-radius: 14px;
  padding: .7rem .9rem; transition: border-color .15s ease, box-shadow .15s ease;
}
div[class*="st-key-file_row_"]:hover { border-color: #CFCFF5; box-shadow: var(--sm-shadow); }
.sm-file { display: flex; align-items: center; gap: .75rem; min-width: 0; }
.sm-file-icon {
  width: 40px; height: 40px; flex: 0 0 40px; border-radius: 11px; display: grid; place-items: center;
  background: var(--sm-primary-soft); font-size: 1.15rem;
}
.sm-file-name { font-weight: 600; font-size: .98rem; overflow-wrap: anywhere; }
.sm-file-sub { color: var(--sm-muted); font-size: .8rem; margin-top: .1rem; }
.sm-badge {
  display: inline-block; padding: .12rem .55rem; border-radius: 999px; font-size: .74rem; font-weight: 500;
}
.sm-badge.indexed { background: var(--sm-ok-soft); color: var(--sm-ok); }
.sm-badge.pending, .sm-badge.updating { background: var(--sm-warn-soft); color: var(--sm-warn); }
.sm-badge.problem { background: var(--sm-bad-soft); color: var(--sm-bad); }
.sm-badge.topic { background: var(--sm-primary-soft); color: var(--sm-primary-dark); }
.sm-badge.reason { background: #F1F3F8; color: var(--sm-muted); }
.sm-snippet { color: var(--sm-muted); font-size: .82rem; margin-top: .3rem; }

/* ---------- breadcrumbs ---------- */
.sm-crumbs { color: var(--sm-muted); font-size: .9rem; display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; }
.sm-crumbs b { color: var(--sm-text); font-weight: 600; }

/* ---------- document view ---------- */
.st-key-doc_view { padding-top: .2rem; }
.sm-doc-head {
  display: flex; gap: .9rem; align-items: center; padding: 1rem 1.1rem; border-radius: 16px;
  background: linear-gradient(135deg, #F1F1FE, #F7F2FE); border: 1px solid #E4E2FA; margin-bottom: 1rem;
}
.sm-doc-head .sm-folder-icon { margin: 0; background: #fff; }
.sm-doc-title { font-size: 1.25rem; font-weight: 700; overflow-wrap: anywhere; }
.sm-doc h1 { font-size: 1.45rem; margin: .4rem 0 .6rem; }
.sm-doc h2 {
  font-size: 1.12rem; margin: 1.5rem 0 .5rem; padding-bottom: .35rem; border-bottom: 1px solid var(--sm-border);
}
.sm-doc p, .sm-doc li { line-height: 1.8; font-size: 1rem; color: #2A2F45; }
.sm-doc ul, .sm-doc ol { padding-inline-start: 1.5rem; margin: .3rem 0 .9rem; }
.sm-doc li { margin-bottom: .25rem; }
.sm-doc ol li::marker { color: var(--sm-primary); font-weight: 600; }
.sm-doc p { margin: 0 0 .85rem; }
.sm-doc strong { color: var(--sm-primary-dark); font-weight: 600; }
.sm-doc bdi { unicode-bidi: isolate; }
.sm-doc li::marker { color: var(--sm-primary); }
.sm-doc table { border-collapse: collapse; width: 100%; margin: .6rem 0 1rem; font-size: .9rem; }
.sm-doc th, .sm-doc td { border: 1px solid var(--sm-border); padding: .45rem .6rem; text-align: right; }
.sm-doc th { background: #F4F4FD; font-weight: 600; }
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {
  direction: ltr !important; text-align: left !important;
}
[data-testid="stCode"] { border-radius: 12px; }

/* ---------- chat ---------- */
.sm-bot-head { display: flex; align-items: center; gap: .7rem; margin-bottom: .2rem; }
.sm-bot-avatar {
  width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; font-size: 1.35rem;
  background: linear-gradient(135deg, #6D6BF0, #8E5BE8); color: #fff;
}
.sm-bot-title { font-weight: 700; font-size: 1.1rem; }
.sm-bot-sub { color: var(--sm-muted); font-size: .82rem; }
.st-key-chat_messages { border-radius: 14px; background: #FAFAFD; border: 1px solid var(--sm-border); }
[data-testid="stChatMessage"] { background: transparent; padding: .35rem .2rem; gap: .55rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: var(--sm-primary); color: #fff; border-radius: 14px 14px 14px 4px; padding: .55rem .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p { color: #fff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
  background: #fff; border: 1px solid var(--sm-border); border-radius: 14px 14px 4px 14px; padding: .55rem .8rem;
}
.sm-empty-chat { text-align: center; color: var(--sm-muted); padding: 1.6rem .6rem .6rem; font-size: .92rem; }
.sm-empty-chat .big { font-size: 2.2rem; }
.sm-focus {
  display: flex; align-items: center; gap: .45rem; background: var(--sm-primary-soft); color: var(--sm-primary-dark);
  border-radius: 10px; padding: .4rem .7rem; font-size: .85rem; font-weight: 500;
}
.sm-source { font-size: .85rem; font-weight: 500; }
.sm-source-snippet { color: var(--sm-muted); font-size: .8rem; line-height: 1.5; margin: .1rem 0 .3rem; }
div[class*="st-key-suggest_"] button {
  width: 100%; text-align: right; border-radius: 12px; background: #fff; border: 1px solid var(--sm-border);
  color: var(--sm-text); font-size: .86rem;
}
div[class*="st-key-suggest_"] button:hover { border-color: var(--sm-primary); color: var(--sm-primary-dark); }
.stChatInput { border-radius: 14px; }

/* ---------- summaries ---------- */
.st-key-summary_card {
  background: linear-gradient(135deg, #F3F2FF, #F8F4FF); border: 1px solid #DCD8FA;
  border-radius: 14px; padding: .8rem .9rem; margin: .5rem 0;
}
.sm-plan-title { font-weight: 700; margin-bottom: .3rem; }
.sm-plan { font-size: .84rem; color: #3A3F57; line-height: 1.6; }
.sm-plan-cost { display: inline-block; margin-top: .25rem; font-weight: 600; color: var(--sm-primary-dark); }
[data-testid="stChatMessageContent"] h1 { font-size: 1.15rem; margin: .2rem 0 .4rem; }
[data-testid="stChatMessageContent"] h2 { font-size: 1rem; margin: .7rem 0 .3rem; }
[data-testid="stChatMessageContent"] h3 { font-size: .95rem; }
.st-key-summary_view h1 { font-size: 1.4rem; }
.st-key-summary_view h2 {
  font-size: 1.12rem; margin-top: 1.3rem; padding-bottom: .3rem; border-bottom: 1px solid var(--sm-border);
}
.st-key-summary_view p, .st-key-summary_view li { line-height: 1.8; }
[data-testid="stMarkdownContainer"] pre, [data-testid="stMarkdownContainer"] pre code {
  direction: ltr !important; text-align: left !important;
}

/* ---------- widgets ---------- */
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--sm-primary); border-color: var(--sm-primary); border-radius: 10px;
}
.stTextInput input { border-radius: 12px; }
[data-testid="stPopoverButton"] { border-radius: 10px; }

/* ---------- small screens: library first, chat below, full width ---------- */
@media (max-width: 900px) {
  .block-container { padding: .8rem .8rem 1.5rem !important; }
  [data-testid="stHorizontalBlock"]:has(.st-key-chat_panel) { flex-direction: column; }
  [data-testid="stColumn"]:has(.st-key-chat_panel) { position: static; order: 2; width: 100% !important; flex: 1 1 100% !important; }
  [data-testid="stColumn"]:has(.st-key-library_panel) { order: 1; width: 100% !important; flex: 1 1 100% !important; }
  .st-key-library_panel, .st-key-chat_panel { padding: 1rem .9rem; border-radius: 16px; }
  .sm-brand-title { font-size: 1.2rem; }
  /* folder cards: two per row, compact */
  [data-testid="stHorizontalBlock"]:has(div[class*="st-key-folder_card_"]) {
    flex-direction: row !important; flex-wrap: wrap !important; gap: .7rem !important;
  }
  [data-testid="stColumn"]:has(div[class*="st-key-folder_card_"]) {
    flex: 1 1 calc(50% - .4rem) !important; width: calc(50% - .4rem) !important; min-width: 0 !important;
  }
  div[class*="st-key-folder_card_"] { padding: .8rem .75rem .55rem; }
  .sm-folder-desc { display: none; }
  .sm-folder-icon { width: 40px; height: 40px; font-size: 1.3rem; margin-bottom: .4rem; }
  .sm-folder-title { font-size: .95rem; }
  /* status + upload buttons side by side */
  [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {
    flex-direction: row !important; flex-wrap: wrap !important; gap: .5rem !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) > [data-testid="stColumn"] {
    flex: 1 1 calc(50% - .3rem) !important; width: calc(50% - .3rem) !important; min-width: 0 !important;
  }
  [data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) > [data-testid="stColumn"]:first-child {
    flex: 1 1 100% !important; width: 100% !important;
  }
  .sm-jump { display: inline-flex !important; }
}
.sm-jump {
  display: none; margin: .6rem 0 .7rem; padding: .4rem .9rem; border-radius: 999px; text-decoration: none;
  background: var(--sm-primary); color: #fff !important; font-size: .88rem; font-weight: 500;
}
</style>
"""
