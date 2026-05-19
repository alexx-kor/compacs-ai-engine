"""Dataset discovery for unified instructions layout."""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, enum.Enum):
        """Python 3.10 compatibility for enum.StrEnum (3.11+)."""

log = logging.getLogger(__name__)


class DatasetKind(StrEnum):
    """Supported dataset categories under ``instructions/``."""

    RAW = "raw"
    GRAPH_QA = "graph_qa"
    GOLDEN = "golden"


@dataclass(frozen=True)
class GraphQAPair:
    """Question and answer files for one graph topic directory."""

    topic_dir: Path
    questions_path: Path
    answer_path: Path


@dataclass(frozen=True)
class GoldenItem:
    """Single golden evaluation record with reference answer."""

    id: int
    question: str
    expected_answer: str


@dataclass(frozen=True)
class DatasetScanResult:
    """Summary of files discovered in instructions."""

    raw_files: tuple[Path, ...]
    graph_pairs: tuple[GraphQAPair, ...]
    golden_files: tuple[Path, ...]


class DatasetScanner:
    """Scan the unified ``instructions/`` directory tree."""

    RAW_EXTENSIONS = (".pdf", ".rst", ".md", ".txt")

    def __init__(self, instructions_dir: Path) -> None:
        self._instructions_dir = instructions_dir

    def scan(self) -> DatasetScanResult:
        """Discover raw, graph, and golden datasets."""
        raw_dir = self._instructions_dir / "raw"
        graph_dir = self._instructions_dir / "graph"
        golden_dir = self._instructions_dir / "golden"
        raw_files = tuple(self.scan_raw_files(raw_dir))
        graph_pairs = tuple(self.scan_graph_qa_pairs(graph_dir))
        golden_files = tuple(self._scan_golden_files(golden_dir))
        return DatasetScanResult(
            raw_files=raw_files,
            graph_pairs=graph_pairs,
            golden_files=golden_files,
        )

    def scan_raw_files(self, raw_dir: Path | None = None) -> list[Path]:
        """Find supported document files under ``instructions/raw/``."""
        root = raw_dir or (self._instructions_dir / "raw")
        if not root.exists():
            return []
        files: list[Path] = []
        for extension in self.RAW_EXTENSIONS:
            files.extend(sorted(root.rglob(f"*{extension}")))
        return files

    def scan_graph_qa_pairs(self, graph_dir: Path | None = None) -> list[GraphQAPair]:
        """Find ``questions.txt`` + ``answer.txt`` pairs under ``instructions/graph/``."""
        root = graph_dir or (self._instructions_dir / "graph")
        if not root.exists():
            return []
        pairs: list[GraphQAPair] = []
        for topic_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            questions_path = topic_dir / "questions.txt"
            answer_path = topic_dir / "answer.txt"
            if questions_path.exists() and answer_path.exists():
                pairs.append(
                    GraphQAPair(
                        topic_dir=topic_dir,
                        questions_path=questions_path,
                        answer_path=answer_path,
                    )
                )
        return pairs

    def _resolve_golden_json_path(self, golden_path: Path | None = None) -> Path:
        if golden_path is None:
            return self._instructions_dir / "golden" / "golden_set.json"
        if golden_path.is_dir():
            return golden_path / "golden_set.json"
        return golden_path

    def _load_golden_question_lines(self, golden_dir: Path) -> list[str]:
        for name in ("questions.txt", "questions"):
            questions_path = golden_dir / name
            if questions_path.is_file():
                return [
                    line.strip()
                    for line in questions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        return []

    def load_golden_set(self, golden_path: Path | None = None) -> list[GoldenItem]:
        """Load golden Q&A items from ``golden_set.json``."""
        json_path = self._resolve_golden_json_path(golden_path)
        if not json_path.exists():
            log.warning("Golden set not found path=%s", json_path)
            return []
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            log.warning("Invalid golden_set.json path=%s error=%s", json_path, error)
            return []
        if not isinstance(payload, list):
            return []

        items: list[GoldenItem] = []
        for index, entry in enumerate(payload, start=1):
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question", "")).strip()
            if not question:
                continue
            raw_id = entry.get("id", index)
            try:
                item_id = int(raw_id)
            except (TypeError, ValueError):
                item_id = index
            items.append(
                GoldenItem(
                    id=item_id,
                    question=question,
                    expected_answer=str(entry.get("expected_answer", "")).strip(),
                )
            )
        return items

    def load_golden_questions(self, golden_path: Path | None = None) -> list[str]:
        """Load questions from ``golden_set.json`` or plain-text question lists."""
        items = self.load_golden_set(golden_path)
        if items:
            return [item.question for item in items]

        root = (
            golden_path
            if golden_path is not None and golden_path.is_dir()
            else self._instructions_dir / "golden"
        )
        return self._load_golden_question_lines(root)

    def _scan_golden_files(self, golden_dir: Path) -> list[Path]:
        if not golden_dir.exists():
            return []
        return sorted(path for path in golden_dir.rglob("*") if path.is_file())
