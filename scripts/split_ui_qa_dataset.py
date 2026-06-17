#!/usr/bin/env python3
"""Split UI QA dataset into train/val/test with fixed seed."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _to_chat_jsonl(items: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты эксперт по ПО КОМПАКС. Отвечай точно по документации, структурированно, на русском.",
                    },
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["expected_answer"]},
                ]
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split QA dataset into train/val/test")
    parser.add_argument("--input", default="instructions/golden/ui_extension_qa_150.json")
    parser.add_argument("--output-dir", default="instructions/golden/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))

    rng = random.Random(args.seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_n = int(n * args.train_ratio)
    val_n = int(n * args.val_ratio)

    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n :]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ui_extension_qa_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ui_extension_qa_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ui_extension_qa_test.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _to_chat_jsonl(train, output_dir / "ui_extension_qa_train.jsonl")
    _to_chat_jsonl(val, output_dir / "ui_extension_qa_val.jsonl")
    _to_chat_jsonl(test, output_dir / "ui_extension_qa_test.jsonl")

    meta = {
        "input": str(input_path),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {"total": n, "train": len(train), "val": len(val), "test": len(test)},
    }
    (output_dir / "ui_extension_qa_split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Split complete: total={n}, train={len(train)}, val={len(val)}, test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
