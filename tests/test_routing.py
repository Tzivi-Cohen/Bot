"""Local recognition of summary requests and topics (rag/topics.py) - no API."""

import pytest

from rag.topics import detect_topics, expand_query, is_summary_request, route


@pytest.mark.parametrize("text", [
    "סכם לי את החומר", "הסכם לי את החומר בפייתון", "סכמי לי ארדואינו", "תסכם בבקשה",
    "תמצת את הנושא", "תמצתי לי", "סיכום של STM32", "תקציר על אלקטרוניקה",
    "תן לי סקירה על C++", "summary of Python", "Summarize embedded systems", "overview of C++",
])
def test_summary_requests_recognized(text):
    assert is_summary_request(text)


@pytest.mark.parametrize("text", [
    "מה ההבדל בין / ל־// ב־Python?", "מה זה מצביע ב־C?", "איך עובד PWM?",
    "מה זה Member Initializer List?", "מהו חציון?",
])
def test_regular_questions_not_summaries(text):
    assert not is_summary_request(text)


@pytest.mark.parametrize("text, topic", [
    ("הסכם לי את החומר בפייתון", "Python"),
    ("python loops", "Python"),
    ("סכמי לי ארדואינו", "Arduino"),
    ("Arduino PWM", "Arduino"),
    ("תמצת את סי פלוס פלוס", "C++"),
    ("summary of C++", "C++"),
    ("ירושה ב־cpp", "C++"),
    ("מצביעים בשפת C", "C"),
    ("מה זה מצביע ב־C?", "C"),
    ("סיכום מערכות משובצות", "Embedded-Systems"),
    ("embedded systems overview", "Embedded-Systems"),
    ("טיימרים ב־STM32", "STM32"),
    ("סקירה על אלקטרוניקה", "Electronics"),
    ("מבני נתונים", "Data-Structures-and-Algorithms"),
    ("algorithms summary", "Data-Structures-and-Algorithms"),
])
def test_topics_detected_in_hebrew_and_english(text, topic):
    assert detect_topics(text) == [topic]


def test_cpp_is_not_mistaken_for_c_and_csharp_is_ignored():
    assert detect_topics("C++ classes") == ["C++"]
    assert detect_topics("מה זה C#?") == []


def test_wrong_topic_not_chosen():
    assert route("סכמי לי ארדואינו")["topic"] == "Arduino"
    assert route("הסכם לי פייתון")["topic"] == "Python"
    assert route("סכם את C++")["topic"] == "C++"


def test_route_decisions():
    assert route("מה זה רשימה בפייתון?") == {"kind": "question"}
    assert route("סכם לי בבקשה") == {"kind": "choose_topic", "candidates": []}
    assert route("סכם את פייתון וארדואינו")["kind"] == "choose_topic"
    assert set(route("סכם את פייתון וארדואינו")["candidates"]) == {"Python", "Arduino"}
    # inside a file: "summarize this file" / no topic -> that file
    assert route("סכם את הקובץ הזה", "Python", "/kb/a.txt") == \
        {"kind": "summary", "scope": "file", "file": "/kb/a.txt"}
    assert route("תסכם", "Python", "/kb/a.txt")["scope"] == "file"
    # inside a file but explicitly naming another topic -> that topic
    assert route("סכם לי ארדואינו", "Python", "/kb/a.txt") == \
        {"kind": "summary", "scope": "topic", "topic": "Arduino"}


def test_expand_query_adds_canonical_name_only_when_missing():
    assert expand_query("מה זה רשימה בפייתון?") == "מה זה רשימה בפייתון? (Python)"
    assert expand_query("מה זה list ב־Python?") == "מה זה list ב־Python?"
    assert expand_query("טיימרים בארדואינו") == "טיימרים בארדואינו (Arduino, Timer)"
    assert expand_query("מה זה חציון?") == "מה זה חציון?"


def test_concept_terms_map_hebrew_study_terms():
    from rag.topics import concept_terms
    assert concept_terms("מהו בנאי?") == ["Constructor"]
    assert concept_terms("הסבר על ירושה ופולימורפיזם") == ["Polymorphism", "Inheritance"] or \
        set(concept_terms("הסבר על ירושה ופולימורפיזם")) == {"Inheritance", "Polymorphism"}
    assert concept_terms("לולאות בפייתון") == ["Loops"]
    assert concept_terms("מה זה חציון?") == []


def test_retrieval_query_replaces_hebrew_topic_names_and_keeps_context():
    from rag.topics import retrieval_query
    # the request phrasing ("הסבר לי") is not searched for
    assert retrieval_query("הסבר לי לולאות בפייתון") == "לולאות Python (Loops)"
    assert "Arduino" in retrieval_query("מה עושה analogWrite בארדואינו?")
    assert "ארדואינו" not in retrieval_query("מה עושה analogWrite בארדואינו?")
    assert retrieval_query("תן לי דוגמה מתוך החומר", "Python") == "תן לי דוגמה מתוך החומר (Python)"
    # a topic named in the question wins over the context topic
    assert "Arduino" in retrieval_query("PWM בארדואינו", "Python")
    assert "Python" not in retrieval_query("PWM בארדואינו", "Python")
