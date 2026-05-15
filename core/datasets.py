"""Dataset discovery for unified instructions layout."""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class DatasetKind(enum.StrEnum):
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

    def load_golden_questions(self, golden_dir: Path | None = None) -> list[str]:
        """Load questions from ``golden/questions.txt`` or ``golden_set.json``."""
        root = golden_dir or (self._instructions_dir / "golden")
        questions_path = root / "questions.txt"
        if questions_path.exists():
            return [
                line.strip()
                for line in questions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        golden_json = root / "golden_set.json"
        if not golden_json.exists():
            return []
        try:
            payload = json.loads(golden_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            log.warning("Invalid golden_set.json path=%s error=%s", golden_json, error)
            return []
        if isinstance(payload, list):
            return [str(item.get("question", item)) for item in payload if item]
        return []

    def _scan_golden_files(self, golden_dir: Path) -> list[Path]:
        if not golden_dir.exists():
            return []
        return sorted(path for path in golden_dir.rglob("*") if path.is_file())
