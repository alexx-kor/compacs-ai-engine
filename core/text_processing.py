"""Text normalization, tokenization, lemmatization, and structural markup."""

from __future__ import annotations

import logging
import re
from enum import Enum
from functools import lru_cache
from typing import Iterable

log = logging.getLogger(__name__)

# Technical tokens preserved during lemmatization / BM25 tokenization.
PROTECTED_TOKEN_RE = re.compile(
    r"(?:"
    r"\d{1,3}(?:\.\d{1,3}){3}"
    r"|[A-Za-z0-9_]+@[\d.]+"
    r"|WT_[A-Z0-9_]+|DAGSTER_HOME"
    r"|raw_<[^>]+>"
    r"|manually_labeled_<[^>]+>"
    r"|warehouse/[a-z_/]+"
    r"|sftp\s+\S+"
    r"|ssh\s+-L\s+\S+"
    r"|sshfs\s+\S+"
    r"|\./run_dagster\.sh"
    r"|INSTALL iceberg;|LOAD iceberg;"
    r"|iceberg_scan\([^)]+\)"
    r")",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"\b[\wа-яёА-ЯЁ]{3,}\b", re.UNICODE)
SECTION_HEADER_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.{3,120})$",
    re.MULTILINE,
)
TABLE_HEADER_RE = re.compile(r"^Таблица\s+(\d+)\s*[–\-—]\s*(.+)$", re.MULTILINE | re.IGNORECASE)
DEFINITION_LINE_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s+([^:\n]{3,80})\s*:\s*(.+)$",
    re.MULTILINE,
)


class PreprocessProfile(str, Enum):
    BASELINE = "baseline"
    NORMALIZED = "normalized"
    TECHNICAL = "technical"
    MARKDOWN = "markdown"
    LEMMA_HINT = "lemma_hint"
    EXPERIMENT = "experiment"


def preprocess_text(text: str, profile: str | PreprocessProfile) -> str:
    """Apply a preprocessing profile to raw document text."""
    name = profile.value if isinstance(profile, PreprocessProfile) else profile
    if name == PreprocessProfile.BASELINE.value:
        return _baseline_clean(text)
    if name == PreprocessProfile.NORMALIZED.value:
        return _normalized_clean(text)
    if name == PreprocessProfile.TECHNICAL.value:
        return _technical_clean(text)
    if name == PreprocessProfile.MARKDOWN.value:
        return _markdown_markup(text)
    if name == PreprocessProfile.LEMMA_HINT.value:
        return _lemma_hint_markup(text)
    if name == PreprocessProfile.EXPERIMENT.value:
        return _experiment_clean(text)
    raise ValueError(f"Unknown preprocessing profile: {profile}")


def _baseline_clean(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _experiment_clean(text: str) -> str:
    cleaned = _baseline_clean(text)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\.,:;!?/\-()]", " ", cleaned)
    cleaned = re.sub(r"\b\d{16,}\b", " ", cleaned)
    cleaned = re.sub(r"([!?.,:;])\1{1,}", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _normalized_clean(text: str) -> str:
    cleaned = _baseline_clean(text)
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\.{5,}", " ", cleaned)
    return cleaned


def _technical_clean(text: str) -> str:
    """Normalize whitespace but keep commands, paths, and identifiers intact."""
    return _normalized_clean(text)


def _markdown_markup(text: str) -> str:
    """Add lightweight structure markers for embedding-friendly chunks."""
    cleaned = _normalized_clean(text)
    blocks: list[str] = []
    for title, body in split_sections(cleaned):
        blocks.append(f"## {title}\n{body.strip()}")
    return "\n\n".join(blocks) if blocks else cleaned


def _lemma_hint_markup(text: str) -> str:
    """Append normalized token hints without replacing original words."""
    cleaned = _normalized_clean(text)
    tokens = tokenize_for_search(cleaned, lemmatize=True)
    if not tokens:
        return cleaned
    hint = " ".join(sorted(set(tokens))[:80])
    return f"{cleaned}\n\n[SEARCH_HINTS] {hint}"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split document into (section_title, section_body) pairs."""
    markers: list[tuple[int, str]] = []
    for match in SECTION_HEADER_RE.finditer(text):
        title = match.group(2).strip()
        if re.search(r"\.{8,}", title):
            continue
        number = match.group(1)
        markers.append((match.start(), f"{number} {title}"))
    for match in TABLE_HEADER_RE.finditer(text):
        markers.append((match.start(), f"Таблица {match.group(1)} — {match.group(2).strip()}"))

    if not markers:
        return [("document", text)]

    markers.sort(key=lambda item: item[0])
    sections: list[tuple[str, str]] = []
    for index, (start, title) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        body = text[start:end].strip()
        if len(body) >= 30:
            sections.append((title, body))
    return sections or [("document", text)]


def extract_definition_lines(text: str, source: str) -> list[dict[str, str]]:
    """Extract glossary-style lines as small definition records."""
    definitions: list[dict[str, str]] = []
    for match in DEFINITION_LINE_RE.finditer(text):
        term = match.group(2).strip()
        definition = match.group(3).strip()
        if len(definition) < 20:
            continue
        definitions.append({"term": term, "definition": definition, "source": source})
    return definitions


def format_chunk_markup(
    body: str,
    *,
    chunk_type: str,
    source: str,
    section: str = "",
    profile: str = "baseline",
) -> str:
    """Wrap chunk text with explicit type/section markers for retrieval."""
    header = f"[{chunk_type.upper()} | source={source}"
    if section:
        header += f" | section={section}"
    if profile != "baseline":
        header += f" | preprocess={profile}"
    header += "]"
    return f"{header}\n{body.strip()}"


def tokenize_for_search(text: str, *, lemmatize: bool = False) -> list[str]:
    """Tokenize text for BM25 / lexical overlap; optionally lemmatize Russian words."""
    protected: list[tuple[int, int, str]] = []
    for match in PROTECTED_TOKEN_RE.finditer(text):
        protected.append((match.start(), match.end(), match.group(0).lower()))

    tokens: list[str] = []
    for match in WORD_RE.finditer(text.lower()):
        start, end = match.span()
        if any(start >= p_start and end <= p_end for p_start, p_end, _ in protected):
            continue
        word = match.group(0)
        if lemmatize:
            word = lemmatize_word(word)
        if len(word) >= 3:
            tokens.append(word)

    for _, _, value in protected:
        tokens.extend(re.findall(r"[\w.]+", value.lower()))
    return tokens


def lemmatize_word(word: str) -> str:
    """Lemmatize a single word; uses pymorphy2 when installed."""
    morph = _get_morphAnalyzer()
    if morph is None:
        return word.lower()
    parsed = morph.parse(word)
    if not parsed:
        return word.lower()
    return parsed[0].normal_form


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    return [lemmatize_word(token) for token in tokens]


@lru_cache(maxsize=1)
def _get_morphAnalyzer() -> object | None:
    try:
        import pymorphy3  # type: ignore[import-untyped]

        return pymorphy3.MorphAnalyzer()
    except ImportError:
        try:
            import pymorphy2  # type: ignore[import-untyped]

            return pymorphy2.MorphAnalyzer()
        except ImportError:
            log.debug("pymorphy2/3 not installed; lemmatization disabled")
            return None


def expand_query_tokens(question: str) -> list[str]:
    """Build BM25 query tokens with optional lemmas and protected literals."""
    base = tokenize_for_search(question, lemmatize=False)
    lemma = tokenize_for_search(question, lemmatize=True)
    merged = list(dict.fromkeys(base + lemma))
    return merged
