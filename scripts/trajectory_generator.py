from __future__ import annotations

import math
from dataclasses import asdict
import numpy as np
from .config import DataGenConfig


class TrajectoryGenerator:
    def __init__(self, config: DataGenConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample_episode(self) -> dict:
        trajectory_type = self.rng.choice(self.config.trajectory_types).item()
        start = self._sample_position()
        speed = float(self.rng.uniform(*self.config.speed_range))
        duration = float(self.rng.uniform(*self.config.duration_range))
        noise_std = float(self.rng.uniform(*self.config.noise_std_range))

        builder = getattr(self, f"_build_{trajectory_type}")
        episode = builder(start)
        waypoints = self._add_light_noise(episode["waypoints"], noise_std)
        waypoints[0] = start
        goal = waypoints[-1]

        return {
            "trajectory_type": trajectory_type,
            "start_position": self._round_vec(start),
            "goal_position": self._round_vec(goal),
            "waypoints": [self._round_vec(point) for point in waypoints],
            "obstacle_position": self._round_vec(episode.get("obstacle_position")) if episode.get("obstacle_position") is not None else None,
            "dt": self.config.dt,
            "params": {
                "speed": speed,
                "duration": duration,
                "noise_std": noise_std,
                **episode.get("params", {}),
            },
        }

    def _build_line(self, start: np.ndarray) -> dict:
        goal = self._goal_from_heading(start, forward_bias=True)
        count = self._waypoint_count()
        curve = self._interpolate(start, goal, count)
        profile = self.rng.choice(["constant", "accelerate", "decelerate"]).item()
        return {"waypoints": curve, "params": {"profile": profile}}

    def _build_turn(self, start: np.ndarray) -> dict:
        length = self._length()
        heading = self._heading()
        side = self.rng.choice([-1.0, 1.0])
        radius = float(self.rng.uniform(6.0, 22.0))
        angle = float(self.rng.uniform(math.pi / 3, math.pi * 1.25))
        count = self._waypoint_count()
        tangent = np.array([math.cos(heading), math.sin(heading), 0.0])
        normal = side * np.array([-tangent[1], tangent[0], 0.0])
        center = start + normal * radius
        theta0 = math.atan2((start - center)[1], (start - center)[0])
        points = []
        for alpha in np.linspace(0.0, angle, count):
            theta = theta0 - side * alpha
            point = center + np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0])
            point[2] = start[2] + self.rng.uniform(-4.0, 4.0) * alpha / angle
            points.append(self._clip(point))
        if self.rng.random() < 0.45:
            points = self._s_bend(np.array(points), length * 0.12)
        return {"waypoints": np.array(points), "params": {"radius": radius, "angle_rad": angle}}

    def _build_avoidance(self, start: np.ndarray) -> dict:
        goal = self._goal_from_heading(start, forward_bias=True)
        mid = (start + goal) / 2.0
        direction = self._unit(goal - start)
        side = self.rng.choice([-1.0, 1.0])
        lateral = side * np.array([-direction[1], direction[0], 0.0])
        offset = float(self.rng.uniform(7.0, 18.0))
        obstacle = self._clip(mid + lateral * self.rng.uniform(-2.0, 2.0))
        control1 = start + direction * np.linalg.norm(goal - start) * 0.35 + lateral * offset
        control2 = start + direction * np.linalg.norm(goal - start) * 0.65 + lateral * offset
        curve = self._bezier(start, control1, control2, goal, self._waypoint_count())
        return {
            "waypoints": curve,
            "obstacle_position": obstacle,
            "params": {"avoidance_side": float(side), "offset": offset},
        }

    def _build_altitude_change(self, start: np.ndarray) -> dict:
        goal = self._goal_from_heading(start, forward_bias=False)
        goal[2] = self._sample_z_far_from(start[2], min_delta=6.0)
        return {"waypoints": self._interpolate(start, goal, self._waypoint_count())}

    def _build_hover_arrival(self, start: np.ndarray) -> dict:
        goal = self._goal_from_heading(start, forward_bias=True)
        approach = self._interpolate(start, goal, max(5, self._waypoint_count() - 5))
        hover_points = np.repeat(goal[None, :], 5, axis=0)
        return {
            "waypoints": np.vstack([approach, hover_points]),
            "params": {"hover_time": float(self.rng.uniform(*self.config.hover_time_range))},
        }

    def _build_out_and_back(self, start: np.ndarray) -> dict:
        mid = self._goal_from_heading(start, forward_bias=True)
        end = self._clip(start + self.rng.normal(0.0, [8.0, 8.0, 3.0]))
        bend = self._clip(mid + self.rng.normal(0.0, [8.0, 8.0, 2.0]))
        return {"waypoints": self._polyline([start, mid, bend, end], self._waypoint_count())}

    def _build_sharp_change(self, start: np.ndarray) -> dict:
        heading = self._heading()
        segment = self._length() / 4.0
        points = [start]
        z = start[2]
        for idx in range(1, 5):
            turn = heading + idx * self.rng.choice([math.pi / 2, -math.pi / 2, math.pi * 0.75])
            z = float(np.clip(z + self.rng.uniform(-4.0, 4.0), *self.config.workspace_z))
            points.append(self._clip(points[-1] + np.array([math.cos(turn) * segment, math.sin(turn) * segment, z - points[-1][2]])))
        return {"waypoints": self._polyline(points, self._waypoint_count())}

    def _build_periodic(self, start: np.ndarray) -> dict:
        length = self._length()
        heading = self._heading()
        count = self._waypoint_count()
        amp = float(self.rng.uniform(4.0, 14.0))
        cycles = float(self.rng.uniform(1.0, 2.5))
        direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        lateral = np.array([-direction[1], direction[0], 0.0])
        points = []
        for s in np.linspace(0.0, 1.0, count):
            wave = math.sin(2.0 * math.pi * cycles * s)
            figure = math.sin(4.0 * math.pi * cycles * s) * 0.35
            point = start + direction * length * s + lateral * amp * wave
            point[2] = start[2] + amp * 0.25 * figure
            points.append(self._clip(point))
        return {"waypoints": np.array(points), "params": {"amplitude": amp, "cycles": cycles}}

    def _build_spiral(self, start: np.ndarray) -> dict:
        radius = float(self.rng.uniform(4.0, 14.0))
        height = float(self.rng.uniform(-10.0, 10.0))
        turns = float(self.rng.uniform(1.0, 3.0))
        center = start + np.array([radius, 0.0, 0.0])
        points = []
        for s in np.linspace(0.0, 1.0, self._waypoint_count()):
            theta = 2.0 * math.pi * turns * s
            point = center + np.array([radius * math.cos(theta), radius * math.sin(theta), height * s])
            points.append(self._clip(point))
        points[0] = start
        return {"waypoints": np.array(points), "params": {"radius": radius, "turns": turns}}

    def _build_smooth_random(self, start: np.ndarray) -> dict:
        control_points = [start]
        for _ in range(int(self.rng.integers(3, 6))):
            control_points.append(self._goal_from_heading(control_points[-1], forward_bias=False))
        return {"waypoints": self._catmull_rom(control_points, self._waypoint_count())}

    def _build_combined(self, start: np.ndarray) -> dict:
        p1 = self._goal_from_heading(start, forward_bias=True)
        p2 = self._goal_from_heading(p1, forward_bias=False)
        p3 = self._clip(p2 + np.array([0.0, 0.0, self.rng.uniform(-6.0, 6.0)]))
        hover = np.repeat(p3[None, :], 4, axis=0)
        path = self._polyline([start, p1, p2, p3], max(8, self._waypoint_count() - 4))
        return {"waypoints": np.vstack([path, hover]), "params": {"stages": "line_turn_hover"}}

    def _sample_position(self) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(*self.config.workspace_x),
                self.rng.uniform(*self.config.workspace_y),
                self.rng.uniform(*self.config.workspace_z),
            ],
            dtype=float,
        )

    def _goal_from_heading(self, start: np.ndarray, forward_bias: bool) -> np.ndarray:
        heading = self._heading() if not forward_bias else self.rng.uniform(-math.pi / 3, math.pi / 3)
        length = self._length()
        dz = self.rng.uniform(-7.0, 7.0)
        return self._clip(start + np.array([math.cos(heading) * length, math.sin(heading) * length, dz]))

    def _sample_z_far_from(self, z: float, min_delta: float) -> float:
        for _ in range(20):
            candidate = float(self.rng.uniform(*self.config.workspace_z))
            if abs(candidate - z) >= min_delta:
                return candidate
        return float(np.clip(z + np.sign(self.rng.normal()) * min_delta, *self.config.workspace_z))

    def _interpolate(self, start: np.ndarray, goal: np.ndarray, count: int) -> np.ndarray:
        return np.array([self._clip(start * (1.0 - s) + goal * s) for s in np.linspace(0.0, 1.0, count)])

    def _polyline(self, points: list[np.ndarray], count: int) -> np.ndarray:
        points = [np.array(point, dtype=float) for point in points]
        distances = [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
        total = max(sum(distances), 1e-6)
        samples = []
        for distance, a, b in zip(distances, points[:-1], points[1:]):
            local_count = max(2, int(round(count * distance / total)))
            samples.extend(self._interpolate(a, b, local_count)[:-1])
        samples.append(points[-1])
        return np.array(samples)

    def _bezier(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, count: int) -> np.ndarray:
        points = []
        for t in np.linspace(0.0, 1.0, count):
            point = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            points.append(self._clip(point))
        return np.array(points)

    def _catmull_rom(self, points: list[np.ndarray], count: int) -> np.ndarray:
        padded = [points[0], *points, points[-1]]
        samples = []
        per_segment = max(4, count // (len(points) - 1))
        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
            for t in np.linspace(0.0, 1.0, per_segment, endpoint=False):
                t2, t3 = t * t, t * t * t
                point = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
                samples.append(self._clip(point))
        samples.append(points[-1])
        return np.array(samples)

    def _s_bend(self, points: np.ndarray, amplitude: float) -> np.ndarray:
        start, goal = points[0], points[-1]
        direction = self._unit(goal - start)
        lateral = np.array([-direction[1], direction[0], 0.0])
        for idx, s in enumerate(np.linspace(0.0, 1.0, len(points))):
            points[idx] = self._clip(points[idx] + lateral * amplitude * math.sin(2.0 * math.pi * s))
        return points

    def _add_light_noise(self, points: np.ndarray, noise_std: float) -> np.ndarray:
        if noise_std <= 1e-6 or len(points) <= 2:
            return np.array(points, dtype=float)
        noisy = np.array(points, dtype=float)
        noisy[1:-1] += self.rng.normal(0.0, noise_std, size=noisy[1:-1].shape)
        return np.array([self._clip(point) for point in noisy])

    def _clip(self, point: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(point[0], *self.config.workspace_x),
                np.clip(point[1], *self.config.workspace_y),
                np.clip(point[2], *self.config.workspace_z),
            ],
            dtype=float,
        )

    def _unit(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec[:2])
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return np.array([vec[0] / norm, vec[1] / norm, 0.0])

    def _heading(self) -> float:
        return float(self.rng.uniform(-math.pi, math.pi))

    def _length(self) -> float:
        return float(self.rng.uniform(*self.config.path_length))

    def _waypoint_count(self) -> int:
        low, high = self.config.waypoint_count_range
        return int(self.rng.integers(low, high + 1))

    def _round_vec(self, value: np.ndarray | list[float] | None) -> list[float] | None:
        if value is None:
            return None
        return [round(float(x), 6) for x in value]


def config_as_dict(config: DataGenConfig) -> dict:
    values = asdict(config)
    values["output_path"] = str(config.output_path)
    return values
