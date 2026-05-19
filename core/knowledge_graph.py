"""Extract RDF-style triples from instruction text and build RAG-friendly graph chunks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

GRAPH_NS = "http://compacs.local/kg#"


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source: str
    page: int = 1

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)

    def to_turtle(self) -> str:
        subj = _turtle_id(self.subject)
        obj = _turtle_literal(self.object)
        return f"  {subj} <{GRAPH_NS}{self.predicate}> {obj} ."

    def to_chunk_line(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"


def _turtle_id(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "Entity"
    return f"<{GRAPH_NS}{slug}>"


def _turtle_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class KnowledgeGraphExtractor:
    _LAN_WAN = re.compile(
        r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s*[–\-—]\s*.*?\((?P<net>LAN|WAN)\)", re.I
    )
    _SFTP = re.compile(r"sftp\s+(?P<user>\w+)@(?P<host>[\d.]+)", re.I)
    _SSH = re.compile(r"ssh\s+(?P<user>\w+)@(?P<host>[\d.]+)", re.I)
    _TUNNEL = re.compile(
        r"ssh\s+-L\s+(?P<local>\d+):localhost:(?P<remote>\d+)\s+(?P<user>\w+)@(?P<host>[\d.]+)", re.I
    )
    _DIR = re.compile(r"^(?P<dir>[\w./-]+/)\s+(?P<desc>.+)$")
    _ENV = re.compile(r"^(WT_[A-Z0-9_]+|DAGSTER_HOME)\s+(?P<desc>.+)$")
    _PATTERN = re.compile(
        r"(raw_<источник>_<ГГГГММДД>\.zip|manually_labeled_<ГГГГММДД>\.zip|raw_<[^>]+>_<[^>]+>\.zip)"
    )

    def extract_from_text(self, text: str, source: str) -> list[Triple]:
        out: list[Triple] = []
        for line in text.splitlines():
            for m in self._LAN_WAN.finditer(line):
                pred = "hasLanAddress" if m.group("net").upper() == "LAN" else "hasWanAddress"
                out.append(Triple("AI Server", pred, m.group("ip"), source))
            for m in self._SFTP.finditer(line):
                out.append(Triple("AI Server", "hasSftpEndpoint", f"{m['user']}@{m['host']}", source))
            if "-L" not in line:
                for m in self._SSH.finditer(line):
                    out.append(Triple("AI Server", "hasSshEndpoint", f"{m['user']}@{m['host']}", source))
            for m in self._TUNNEL.finditer(line):
                cmd = f"ssh -L {m['local']}:localhost:{m['remote']} {m['user']}@{m['host']}"
                out.append(Triple("Dagster UI", "hasSshTunnel", cmd, source))
            dm = self._DIR.match(line.strip())
            if dm:
                out.append(Triple("AI Server HDD", "hasDirectory", dm.group("dir"), source))
                out.append(Triple(dm.group("dir"), "directoryPurpose", dm.group("desc").strip(), source))
            em = self._ENV.match(line.strip())
            if em:
                var = em.group(1)
                out.append(Triple("Dagster Pipeline", "hasEnvironmentVariable", var, source))
                dv = re.search(r"\((\d+)\)", em.group("desc"))
                if dv:
                    out.append(Triple(var, "environmentDefault", dv.group(1), source))
            for m in self._PATTERN.finditer(line):
                out.append(Triple(Path(source).stem, "hasFileNamingPattern", m.group(0), source))
            if "warehouse/catalog.db" in line:
                out.append(Triple("Iceberg Warehouse", "hasCatalogFile", "warehouse/catalog.db", source))
            if "INSTALL iceberg;" in line:
                out.append(Triple("DuckDB", "requiresSqlStatement", "INSTALL iceberg;", source))
            if "LOAD iceberg;" in line:
                out.append(Triple("DuckDB", "requiresSqlStatement", "LOAD iceberg;", source))
        if re.search(r"порт(?:у)?\s+`?3000`?", text, re.I):
            out.append(Triple("Dagster UI", "hasDefaultPort", "3000", source))
        return _dedupe(out)

    def extract_from_file(self, path: Path) -> list[Triple]:
        return self.extract_from_text(path.read_text(encoding="utf-8"), path.name)


def _dedupe(triples: Iterable[Triple]) -> list[Triple]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Triple] = []
    for t in triples:
        k = (t.subject, t.predicate, t.object, t.source)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def triples_to_ttl(triples: list[Triple], source: str) -> str:
    lines = ["@prefix kg: <http://compacs.local/kg#> .", f"# {source}", ""]
    lines.extend(t.to_turtle() for t in triples)
    return "\n".join(lines) + "\n"


def triples_to_graph_chunks(triples: list[Triple], *, start_id: int, batch: int = 12) -> list[dict]:
    if not triples:
        return []
    chunks: list[dict] = []
    cid = start_id
    for i in range(0, len(triples), batch):
        part = triples[i : i + batch]
        body = "[KNOWLEDGE GRAPH]\n" + "\n".join(t.to_chunk_line() for t in part)
        chunks.append(
            {
                "id": cid,
                "source": f"graph/{part[0].source}",
                "page": 1,
                "chunk": body,
                "chunk_hash": hashlib.md5(body.encode()).hexdigest(),
                "char_count": len(body),
                "chunk_type": "graph",
            }
        )
        cid += 1
    return chunks


def save_graph_artifacts(triples: list[Triple], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jp = output_dir / "triples.json"
    tp = output_dir / "knowledge.ttl"
    jp.write_text(json.dumps([t.as_dict() for t in triples], ensure_ascii=False, indent=2), encoding="utf-8")
    ttl = "\n".join(
        triples_to_ttl([t for t in triples if t.source == s], s)
        for s in sorted({t.source for t in triples})
    )
    tp.write_text(ttl, encoding="utf-8")
    return jp, tp


def extract_graph_from_instruction_files(files: Sequence[Path]) -> list[Triple]:
    ex = KnowledgeGraphExtractor()
    merged: list[Triple] = []
    for f in files:
        merged.extend(ex.extract_from_file(f))
    return _dedupe(merged)
