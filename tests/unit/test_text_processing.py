from __future__ import annotations

from core.text_processing import (
    expand_query_tokens,
    preprocess_text,
    split_sections,
    tokenize_for_search,
)


def test_split_sections_finds_numbered_headings() -> None:
    text = (
        "1 Introduction\n"
        "Long introduction body with enough characters to pass the minimum section size.\n"
        "2 Purpose\n"
        "Purpose section body also long enough for the section splitter threshold.\n"
    )
    sections = split_sections(text)
    assert len(sections) >= 2
    assert "Introduction" in sections[0][0]


def test_tokenize_preserves_ip() -> None:
    tokens = tokenize_for_search("sftp compacs@5.32.101.214 из WAN", lemmatize=False)
    assert "5.32.101.214" in tokens
    assert "compacs" in tokens


def test_expand_query_includes_lemma_variants() -> None:
    tokens = expand_query_tokens("загрузка данных")
    assert tokens
