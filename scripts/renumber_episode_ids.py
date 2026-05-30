from __future__ import annotations

import argparse
import json
from pathlib import Path


def renumber_episode_ids(input_path: Path, output_path: Path | None = None) -> None:
    output_path = output_path or input_path
    rows = []

    with input_path.open("r", encoding="utf-8") as f:
        for episode_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            print(f"Renumbering episode_id {sample['episode_id']} to {episode_id}")
            sample["episode_id"] = episode_id
            rows.append(sample)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in rows:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renumber episode_id in a JSONL file from 0 by line order.")
    parser.add_argument("input", type=Path, help="Input JSONL file, such as hf_dataset_export/train.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL file. If omitted, the input file is overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renumber_episode_ids(args.input, args.output)


if __name__ == "__main__":
    main()

# python3 scripts/renumber_episode_ids.py hf_dataset_export/train.jsonl