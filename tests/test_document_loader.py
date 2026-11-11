"""Local tests for text extraction and chunking (no API calls)."""


import pytest

import config
from rag.document_loader import (
    DocumentLoadError,
    compute_file_hash,
    extract_documents,
    load_and_split_documents,
    split_documents,
)
from tests.conftest import make_blank_pdf, make_docx

HEBREW = "ממוצע חשבוני הוא סכום כל הערכים חלקי מספר הערכים."


def test_txt_utf8_hebrew(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(HEBREW, encoding="utf-8")
    docs = extract_documents(path)
    assert docs[0].text == HEBREW
    assert docs[0].metadata["file_name"] == "a.txt"
    assert docs[0].metadata["doc_id"] == compute_file_hash(path)


def test_txt_cp1255_hebrew(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes(HEBREW.encode("cp1255"))
    assert extract_documents(path)[0].text == HEBREW


def test_docx_paragraphs_and_tables(tmp_path):
    import docx

    path = tmp_path / "d.docx"
    document = docx.Document()
    document.add_paragraph("פסקה ראשונה")
    document.add_paragraph("Second paragraph in English")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "חציון"
    table.rows[0].cells[1].text = "הערך האמצעי"
    document.save(str(path))

    text = extract_documents(path)[0].text
    assert "פסקה ראשונה" in text
    assert "Second paragraph in English" in text
    assert "חציון | הערך האמצעי" in text


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(DocumentLoadError, match="ריק"):
        extract_documents(path)


def test_whitespace_only_rejected(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        extract_documents(path)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(DocumentLoadError, match="אינו נתמך"):
        extract_documents(path)


def test_corrupt_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not really a pdf")
    with pytest.raises(DocumentLoadError, match="פגום"):
        extract_documents(path)


def test_corrupt_docx(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(DocumentLoadError, match="פגום"):
        extract_documents(path)


def test_scanned_pdf_message(tmp_path):
    path = make_blank_pdf(tmp_path / "scan.pdf")
    with pytest.raises(DocumentLoadError, match="סרוק"):
        extract_documents(path)


def test_real_scanned_pdf_in_knowledge_base_is_reported():
    """The existing 'פרשת תולדות.pdf' has no text layer (read-only check)."""
    pdfs = list(config.KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no PDF in knowledge_base")
    scanned = [p for p in pdfs if p.name == "פרשת תולדות.pdf"]
    if not scanned:
        pytest.skip("scanned PDF not present")
    with pytest.raises(DocumentLoadError, match="סרוק"):
        extract_documents(scanned[0])


def test_hebrew_chunking_respects_size_and_overlap():
    from llama_index.core import Document
    from llama_index.core.utils import get_tokenizer

    sentences = [f"משפט מספר {i} עוסק בסטטיסטיקה ובניתוח נתונים." for i in range(400)]
    doc = Document(text=" ".join(sentences), metadata={"file_name": "long.txt"})
    nodes = split_documents([doc])
    tokenizer = get_tokenizer()

    assert len(nodes) > 2
    for node in nodes:
        assert len(tokenizer(node.get_content())) <= config.CHUNK_SIZE
        assert node.metadata["file_name"] == "long.txt"
    # Overlap: the end of each chunk reappears at the start of the next one.
    for first, second in zip(nodes, nodes[1:]):
        tail = first.text[-30:]
        assert tail in second.text


def test_internal_metadata_not_embedded_or_sent_to_llm(tmp_path):
    """file_name is kept for context; the hash id is excluded (saves tokens)."""
    from llama_index.core.schema import MetadataMode

    path = tmp_path / "x.txt"
    path.write_text(HEBREW, encoding="utf-8")
    node = split_documents(extract_documents(path, source="uploads"))[0]
    file_hash = compute_file_hash(path)

    assert node.metadata["doc_id"] == file_hash
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        content = node.get_content(metadata_mode=mode)
        assert "x.txt" in content
        assert file_hash not in content
        assert "uploads" not in content


def test_load_and_split_skips_bad_files(tmp_path):
    (tmp_path / "good.txt").write_text(HEBREW, encoding="utf-8")
    (tmp_path / "empty.txt").write_bytes(b"")
    make_blank_pdf(tmp_path / "scan.pdf")
    make_docx(tmp_path / "doc.docx", ["שכיח הוא הערך הנפוץ ביותר."])

    documents, nodes = load_and_split_documents(tmp_path)
    names = {d.metadata["file_name"] for d in documents}
    assert names == {"good.txt", "doc.docx"}
    assert len(nodes) == 2
