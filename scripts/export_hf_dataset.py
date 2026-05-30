from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            sample_copy = sample.copy()
            sample_copy.pop("params", None)  
            f.write(json.dumps(sample_copy) + "\n")


def write_dataset_card(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_types = sorted({sample["trajectory_type"] for sample in samples})
    sample_rate = 1.0 / float(samples[0]["dt"]) if samples else 10.0
    text = f"""# ProjectAirSim UAV Kinematic Trajectories
This dataset contains UAV trajectory episodes collected from ProjectAirSim. Each row is one episode. The `states` field is a variable-length sequence sampled at approximately {sample_rate:.2f} Hz.

State vector:
`[t, x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az]`

Fields:
- `episode_id`: integer episode index.
- `trajectory_type`: trajectory family used to generate waypoints.
- `start_position`: NED start position `[x, y, z]` in meters.
- `goal_position`: NED final goal position `[x, y, z]` in meters.
- `waypoints`: planned intermediate NED waypoints.
- `obstacle_position`: synthetic obstacle center for avoidance episodes, otherwise null.
- `dt`: target sampling interval in seconds.
- `states`: sampled kinematic history.

Trajectory types:
{chr(10).join(f"- `{name}`" for name in trajectory_types)}

Source:
Generated with the ProjectAirSim Python client using asynchronous UAV velocity commands and ground-truth or estimated kinematics.
"""
    
    path.write_text(text, encoding="utf-8")


def export_dataset(input_path: Path, output: Path) -> None:
    samples = load_jsonl(input_path)

    write_jsonl(samples, output / "trajectories.jsonl")
    write_dataset_card(samples, output / "README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw UAV JSONL episodes to Hugging Face Datasets format.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/episodes.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/hf_dataset_export"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
