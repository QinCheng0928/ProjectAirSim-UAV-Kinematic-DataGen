from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataGenConfig:
    num_episodes: int = 50
    sample_rate_hz: float = 10.0
    output_path: Path = Path("data/raw/episodes.jsonl")
    scene_config: str = "sim_config/scene_config.jsonc"
    drone_name: str = "Drone1"

    workspace_x: tuple[float, float] = (-10.0, 50.0)
    workspace_y: tuple[float, float] = (-15.0, -5.0)
    workspace_z: tuple[float, float] = (-10.0, -1.0)
    path_length: tuple[float, float] = (10.0, 55.0)
    speed_range: tuple[float, float] = (2.0, 8.0)
    duration_range: tuple[float, float] = (5.0, 18.0)
    waypoint_count_range: tuple[int, int] = (8, 28)
    hover_time_range: tuple[float, float] = (1.5, 5.0)
    noise_std_range: tuple[float, float] = (0.0, 0.35)
    move_step_sec: float = 0.18
    waypoint_tolerance_m: float = 1.0

    trajectory_types: list[str] = field(
        default_factory=lambda: [
            "line",
            "turn",
            "avoidance",
            "altitude_change",
            "hover_arrival",
            "out_and_back",
            "sharp_change",
            "periodic",
            "spiral",
            "smooth_random",
            "combined",
        ]
    )

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz
