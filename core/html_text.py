"""Extract plain text from Qt/HTML help pages."""

from __future__ import annotations

import html as html_lib
import re
from pathlib import Path


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Strip tags and normalize whitespace from HTML."""
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    cleaned = _TAG_RE.sub("\n", cleaned)
    cleaned = html_lib.unescape(cleaned)
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def extract_html_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    languages: tuple[str, ...] = ("ru",),
) -> list[Path]:
    """Convert ``*.html`` under ``source_dir`` into ``.txt`` files under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for html_path in sorted(source_dir.rglob("*.html")):
        rel = html_path.relative_to(source_dir)
        parts = list(rel.parts)
        if parts and "(eng)" in parts[0].lower():
            if "en" not in languages and "eng" not in languages:
                continue
        text = html_to_text(html_path.read_text(encoding="utf-8", errors="replace"))
        if len(text) < 200:
            continue
        out_name = "_".join(parts).replace(".html", ".txt")
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        written.append(out_path)
    return written
