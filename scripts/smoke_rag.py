#!/usr/bin/env python3
"""Quick smoke test for local RAG (UI extension index)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_QUESTIONS = [
    "Что делает кнопка «Новый документ»?",
    "Как создать новый документ без CDPL-процедур?",
    "Какой WAN-адрес используется для загрузки ZIP на AI-сервер?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test RAG answers")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        dest="questions",
        help="Question (repeatable); defaults to built-in demo set",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "baseline" / "golden_set.json",
        help="Optional golden JSON to pick first N questions",
    )
    parser.add_argument("-n", type=int, default=3, help="Questions from golden file")
    parser.add_argument("--stream", action="store_true", help="Use ask_stream")
    parser.add_argument("--json", action="store_true", help="Print JSON lines")
    return parser.parse_args()


def _load_questions(args: argparse.Namespace) -> list[str]:
    if args.questions:
        return args.questions
    if args.golden.is_file():
        items = json.loads(args.golden.read_text(encoding="utf-8"))
        if isinstance(items, list):
            return [str(item.get("question", "")).strip() for item in items[: args.n] if item.get("question")]
    return DEFAULT_QUESTIONS


def main() -> int:
    args = parse_args()
    from core.embedding_alignment import configure_embeddings_for_index
    from rag_service import rag_service

    store = ROOT / "data" / "vectors"
    if store.joinpath("chunks.json").is_file():
        configure_embeddings_for_index(store)

    questions = _load_questions(args)
    if not (ROOT / "data" / "vectors" / "chunks.json").is_file():
        print("Vector index missing — run: python scripts/ui_extension_pipeline.py extract && index", file=sys.stderr)
        return 1
    exit_code = 0
    for question in questions:
        if args.stream:
            answer_parts: list[str] = []
            payload: dict = {}
            for event in rag_service.ask_stream(question):
                if event.get("event") == "token":
                    answer_parts.append(str(event.get("data", {}).get("text", "")))
                if event.get("event") == "done":
                    payload = event.get("data", {})
            answer = payload.get("answer") or "".join(answer_parts)
            sources = payload.get("sources", [])
            provider = payload.get("provider_used", "?")
        else:
            result = rag_service.ask(question)
            answer = str(result.get("answer", ""))
            sources = result.get("sources", [])
            provider = result.get("provider_used", "?")

        not_found = "NOT FOUND" in answer.upper()
        if not_found:
            exit_code = 1

        if args.json:
            print(
                json.dumps(
                    {
                        "question": question,
                        "answer_preview": answer[:400],
                        "source_count": len(sources),
                        "provider": provider,
                        "not_found": not_found,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"Q: {question}")
            print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
            print(f"   sources={len(sources)} provider={provider}")
            if not_found:
                print("   [NOT FOUND]")
            print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
