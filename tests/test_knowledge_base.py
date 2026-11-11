"""Checks on the REAL knowledge_base content - read-only, no embeddings, no API."""

import re
from collections import Counter

import pytest
from llama_index.core.utils import get_tokenizer

import config
from rag.document_loader import (
    compute_file_hash,
    extract_documents,
    iter_supported_files,
    relative_path_for,
    split_documents,
    topic_for,
)

KB = config.KNOWLEDGE_BASE_DIR
TOPICS = ["Python", "C", "C++", "Data-Structures-and-Algorithms",
          "Arduino", "STM32", "Electronics", "Embedded-Systems"]
TOPIC_FILES = [p for p in iter_supported_files(KB) if topic_for(p, KB)]


def test_every_topic_folder_has_at_least_two_documents():
    counts = Counter(topic_for(p, KB) for p in TOPIC_FILES)
    for topic in TOPICS:
        assert (KB / topic).is_dir(), topic
        assert counts[topic] >= 2, f"{topic}: {counts[topic]} files"


def test_no_duplicate_documents():
    hashes = Counter(compute_file_hash(p) for p in iter_supported_files(KB))
    assert all(n == 1 for n in hashes.values())


@pytest.mark.parametrize("path", TOPIC_FILES, ids=lambda p: relative_path_for(p, KB))
def test_document_loads_with_hebrew_and_metadata(path):
    docs = extract_documents(path, topic=topic_for(path, KB),
                             relative_path=relative_path_for(path, KB))
    text = "".join(d.text for d in docs)
    assert len(text) > 3000, "study files should be substantial"
    assert "�" not in text and "­" not in text
    assert re.search("[֐-׿]{3,}", text), "Hebrew text expected"

    nodes = split_documents(docs)
    assert len(nodes) >= 3
    tokenizer = get_tokenizer()
    for node in nodes:
        assert node.metadata["topic"] == topic_for(path, KB)
        assert node.metadata["file_name"] == path.name
        assert len(tokenizer(node.get_content())) <= config.CHUNK_SIZE


def test_enough_chunks_for_meaningful_retrieval():
    nodes = split_documents([d for p in TOPIC_FILES for d in extract_documents(p)])
    assert len(nodes) >= 150
