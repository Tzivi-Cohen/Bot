"""Data layer of the library UI (ui/library.py) - local, no Streamlit, no API."""

from rag.index_manager import IndexManager
from tests.conftest import CountingEmbedding, make_blank_pdf
from ui import library as lib


def no_problem(path, file_hash):
    return None


def test_bidi_html_isolates_latin_and_escapes():
    assert lib.bidi_html("מחלקות ב־C++ ובשפת C.") == \
        'מחלקות ב־<bdi dir="ltr">C++</bdi> ובשפת <bdi dir="ltr">C</bdi>.'
    assert '<bdi dir="ltr">7 / 2 == 3.5</bdi>' in lib.bidi_html("למשל 7 / 2 == 3.5")
    assert lib.bidi_html("<script>") == '<bdi dir="ltr">&lt;script&gt;</bdi>'
    assert lib.bidi_html("עברית בלבד") == "עברית בלבד"
    assert lib.bidi_html("הערך 0.3, כי") == 'הערך <bdi dir="ltr">0.3</bdi>, כי'


def test_parse_study_text_blocks():
    text = (
        "כותרת ראשית\n===========\n\n1. סעיף\n--------\nפסקה ראשונה\nשממשיכה.\n\n"
        "- פריט א\n- פריט ב\n\n    def f():\n        return 1\n\n    print(f())\n\n"
        "| a | b |\n| 1 | 2 |\n"
    )
    blocks = lib.parse_study_text(text, "python")
    assert blocks[0] == ("title", "כותרת ראשית", 1)
    assert blocks[1] == ("title", "1. סעיף", 2)
    assert blocks[2] == ("para", ["פסקה ראשונה", "שממשיכה."])       # each line keeps its row
    assert blocks[3] == ("bullets", ["פריט א", "פריט ב"])
    assert blocks[4] == ("code", "def f():\n    return 1\n\nprint(f())", "python")
    assert blocks[5] == ("table", [["a", "b"], ["1", "2"]])


def test_guess_language():
    assert lib.guess_language("int x = 5;\nprintf(x);", "c") == "c"
    assert lib.guess_language("std::vector<int> v;", None) == "cpp"
    assert lib.guess_language("def f():\n    pass", "c") == "python"
    assert lib.guess_language("R = V / I", None) is None


def test_build_catalog_folders_counts_and_status(dirs):
    kb, uploads, storage = dirs
    for topic, name in [("Python", "a.txt"), ("Python", "b.txt"), ("C++", "c.txt"), ("Robotics", "r.txt")]:
        (kb / topic).mkdir(exist_ok=True)
        (kb / topic / name).write_text(f"תוכן {name}", encoding="utf-8")
    (kb / "STM32").mkdir()                               # known topic, empty
    uploads.mkdir()
    (uploads / "mine.txt").write_text("קובץ שלי", encoding="utf-8")
    make_blank_pdf(kb / "Python" / "scan.pdf")

    m = IndexManager(storage_dir=storage, embed_model=CountingEmbedding(embed_dim=8))
    m.load_or_create()
    m.add_file(kb / "Python" / "a.txt", topic="Python")

    problems = {"scan.pdf": "PDF סרוק"}
    catalog = lib.build_catalog(m, kb, uploads, lambda p, h: problems.get(p.name))
    keys = [f.key for f in catalog]
    assert keys == ["Python", "C++", "STM32", "Robotics", lib.UPLOADS_KEY]   # topic order, others, uploads
    python = catalog[0]
    assert [f.name for f in python.files] == ["a.txt", "b.txt", "scan.pdf"]
    assert [f.status for f in python.files] == ["indexed", "pending", "problem"]
    assert python.indexed_count == 1
    assert catalog[2].files == []                         # empty known topic is still shown
    assert catalog[3].title == "Robotics" and catalog[3].icon == "📁"
    assert catalog[-1].title == "הקבצים שהעליתי" and catalog[-1].files[0].name == "mine.txt"


def test_search_by_name_and_content(dirs):
    kb, uploads, storage = dirs
    (kb / "Arduino").mkdir()
    (kb / "Arduino" / "pwm.txt").write_text("analogWrite מייצרת PWM", encoding="utf-8")
    (kb / "Python").mkdir()
    (kb / "Python" / "loops.txt").write_text("לולאת for ו־while", encoding="utf-8")
    m = IndexManager(storage_dir=storage, embed_model=CountingEmbedding(embed_dim=8))
    m.load_or_create()
    catalog = lib.build_catalog(m, kb, uploads, no_problem)

    def read(path):
        return path.read_text(encoding="utf-8")

    by_content = lib.search_files(catalog, "analogwrite", read)
    assert [i.name for i, _, _ in by_content] == ["pwm.txt"] and "analogWrite" in by_content[0][1]
    assert by_content[0][2] == "בתוכן המסמך"
    assert [i.name for i, _, _ in lib.search_files(catalog, "loops", read)] == ["loops.txt"]
    assert lib.search_files(catalog, "PWM", read, folder_key="Python") == []
    assert lib.search_files(catalog, "   ", read) == []


def test_bidi_keeps_file_names_whole_and_punctuation_outside():
    assert lib.bidi_html("הוסיפי את המפתח לקובץ .env.") == \
        'הוסיפי את המפתח לקובץ <bdi dir="ltr">.env</bdi>.'
    assert '<bdi dir="ltr">.env.example</bdi>' in lib.bidi_html("לפי הקובץ .env.example ולהפעיל")
    assert lib.bidi_html("לדוגמה: 42, -7.") == 'לדוגמה: <bdi dir="ltr">42, -7</bdi>.'
    assert lib.bidi_text("ב־C++?") == "ב־\u2066C++\u2069?"
    # an opening paren of the Hebrew sentence stays outside the Latin name
    # (the status window showed ".env (" isolated together)
    assert lib.bidi_text("לקובץ .env (לפי הדוגמה בקובץ .env.example) ולהפעיל") == \
        "לקובץ \u2066.env\u2069 (לפי הדוגמה בקובץ \u2066.env.example\u2069) ולהפעיל"
    assert lib.bidi_text("הפונקציה print(x) מדפיסה") == "הפונקציה \u2066print(x)\u2069 מדפיסה"


def test_line_html_bolds_labels_and_strips_marks():
    assert lib.line_html("פתרון: \u200f10 20.") == '<strong>פתרון:</strong> <bdi dir="ltr">10 20</bdi>.'
    assert lib.line_html("שאלה 3: מה יודפס?").startswith("<strong>שאלה 3:</strong>")
    assert lib.line_html("משפט רגיל: בלי תווית") == "משפט רגיל: בלי תווית"


def test_numbered_lists_and_question_paragraphs():
    text = "מקרים:\n1) ראשון\n2) שני\n\nשאלה 1: מה?\nפתרון: כך.\nשאלה 2: ולמה?\nפתרון: ככה."
    blocks = lib.parse_study_text(text)
    assert ("numbered", ["ראשון", "שני"]) in blocks
    paragraphs = [b[1] for b in blocks if b[0] == "para"]
    assert ["שאלה 1: מה?"] in paragraphs and ["שאלה 2: ולמה?"] in paragraphs
    assert ["פתרון: כך."] in paragraphs
