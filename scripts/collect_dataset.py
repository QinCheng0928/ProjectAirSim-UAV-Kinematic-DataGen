from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import DataGenConfig
from .trajectory_generator import TrajectoryGenerator


def get_nested(data: Any, *names: str, default: Any = None) -> Any:
    current = data
    for name in names:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(name, default)
        else:
            current = getattr(current, name, default)
    return current


def xyz_from(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, dict):
        return [
            float(value.get("x", value.get("north", default[0]))),
            float(value.get("y", value.get("east", default[1]))),
            float(value.get("z", value.get("down", default[2]))),
        ]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(value[0]), float(value[1]), float(value[2])]
    return [
        float(getattr(value, "x", getattr(value, "north", default[0]))),
        float(getattr(value, "y", getattr(value, "east", default[1]))),
        float(getattr(value, "z", getattr(value, "down", default[2]))),
    ]


def quat_from(value: Any) -> list[float]:
    if value is None:
        return [1.0, 0.0, 0.0, 0.0]
    if isinstance(value, dict):
        return [
            float(value.get("w", value.get("qw", 1.0))),
            float(value.get("x", value.get("qx", 0.0))),
            float(value.get("y", value.get("qy", 0.0))),
            float(value.get("z", value.get("qz", 0.0))),
        ]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    return [
        float(getattr(value, "w", getattr(value, "qw", 1.0))),
        float(getattr(value, "x", getattr(value, "qx", 0.0))),
        float(getattr(value, "y", getattr(value, "qy", 0.0))),
        float(getattr(value, "z", getattr(value, "qz", 0.0))),
    ]


def extract_kinematic_state(
    kinematics: Any,
    t: float,
    previous_velocity: list[float] | None = None,
    dt: float = 0.1,
) -> tuple[list[float], list[float]]:
    pose = get_nested(kinematics, "pose", default=kinematics)
    position = (
        get_nested(kinematics, "position")
        or get_nested(pose, "position")
        or get_nested(pose, "translation")
        or get_nested(kinematics, "translation")
    )
    orientation = (
        get_nested(kinematics, "orientation")
        or get_nested(pose, "orientation")
        or get_nested(pose, "rotation")
        or get_nested(kinematics, "rotation")
    )
    linear_velocity = (
        get_nested(kinematics, "linear_velocity")
        or get_nested(kinematics, "twist", "linear")
        or get_nested(kinematics, "velocity")
    )
    angular_velocity = (
        get_nested(kinematics, "angular_velocity")
        or get_nested(kinematics, "twist", "angular")
        or get_nested(kinematics, "angular")
    )
    linear_acceleration = (
        get_nested(kinematics, "linear_acceleration")
        or get_nested(kinematics, "acceleration")
        or get_nested(kinematics, "accel")
    )

    pos = xyz_from(position)
    quat = quat_from(orientation)
    vel = xyz_from(linear_velocity)
    omega = xyz_from(angular_velocity)
    if linear_acceleration is None and previous_velocity is not None:
        acc = [(vel[i] - previous_velocity[i]) / dt for i in range(3)]
    else:
        acc = xyz_from(linear_acceleration)

    state = [float(t), *pos, *quat, *vel, *omega, *acc]
    return [round(float(x), 6) for x in state], vel


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def await_projectairsim_task(task_or_value: Any) -> Any:
    task = await maybe_await(task_or_value)
    if inspect.isawaitable(task):
        return await task
    return task


async def move_by_velocity(drone: Any, velocity: np.ndarray, duration: float) -> None:
    yaw = math.atan2(float(velocity[1]), float(velocity[0]))
    kwargs = {
        "v_north": float(velocity[0]),
        "v_east": float(velocity[1]),
        "v_down": float(velocity[2]),
        "duration": float(duration),
        "yaw_is_rate": False,
        "yaw": float(yaw),
    }
    try:
        await await_projectairsim_task(drone.move_by_velocity_async(**kwargs))
    except TypeError:
        await await_projectairsim_task(
            drone.move_by_velocity_async(
                v_north=kwargs["v_north"],
                v_east=kwargs["v_east"],
                v_down=kwargs["v_down"],
                duration=kwargs["duration"],
            )
        )


async def get_kinematics(drone: Any) -> Any:
    for name in (
        "get_ground_truth_kinematics",
        "get_estimated_kinematics",
        "get_kinematics",
        "get_ground_truth_state",
        "get_state",
    ):
        method = getattr(drone, name, None)
        if method is not None:
            return await maybe_await(method())
    raise AttributeError("No supported kinematics method found on ProjectAirSim Drone.")


async def sample_state(
    drone: Any,
    start_time: float,
    previous_velocity: list[float] | None,
    dt: float,
) -> tuple[list[float], list[float]]:
    kinematics = await get_kinematics(drone)
    elapsed = time.monotonic() - start_time
    return extract_kinematic_state(kinematics, elapsed, previous_velocity, dt)


async def move_to_position_with_sampling(
    drone: Any,
    target: list[float],
    speed: float,
    sample_rate: float,
    start_time: float,
    states: list[list[float]],
    previous_velocity: list[float] | None,
    move_step_sec: float,
    tolerance_m: float,
) -> list[float] | None:
    dt = 1.0 / sample_rate
    target_np = np.array(target, dtype=float)
    while True:
        state, previous_velocity = await sample_state(drone, start_time, previous_velocity, dt)
        states.append(state)
        current = np.array(state[1:4], dtype=float)
        delta = target_np - current
        distance = float(np.linalg.norm(delta))
        if distance <= tolerance_m:
            return previous_velocity

        direction = delta / max(distance, 1e-6)
        command_speed = min(speed, distance / max(move_step_sec, 1e-3))
        velocity = direction * command_speed
        await move_by_velocity(drone, velocity, move_step_sec)
        await asyncio.sleep(dt)


async def hover_with_sampling(
    drone: Any,
    seconds: float,
    sample_rate: float,
    start_time: float,
    states: list[list[float]],
    previous_velocity: list[float] | None,
) -> list[float] | None:
    await await_projectairsim_task(drone.hover_async())
    dt = 1.0 / sample_rate
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        state, previous_velocity = await sample_state(drone, start_time, previous_velocity, dt)
        states.append(state)
        await asyncio.sleep(dt)
    return previous_velocity


async def fly_episode(drone: Any, episode: dict, config: DataGenConfig) -> dict:
    states: list[list[float]] = []
    previous_velocity = None
    start_time = time.monotonic()
    speed = float(episode["params"]["speed"])

    for waypoint in episode["waypoints"]:
        previous_velocity = await move_to_position_with_sampling(
            drone=drone,
            target=waypoint,
            speed=speed,
            sample_rate=config.sample_rate_hz,
            start_time=start_time,
            states=states,
            previous_velocity=previous_velocity,
            move_step_sec=config.move_step_sec,
            tolerance_m=config.waypoint_tolerance_m,
        )

    if episode["trajectory_type"] in {"hover_arrival", "combined"}:
        hover_time = float(episode["params"].get("hover_time", 2.0))
        previous_velocity = await hover_with_sampling(
            drone, hover_time, config.sample_rate_hz, start_time, states, previous_velocity
        )

    episode["states"] = states
    return episode


async def move_to_start(drone: Any, start_position: list[float], config: DataGenConfig, speed: float) -> None:
    target = np.array(start_position, dtype=float)
    if hasattr(drone, "move_to_position_async"):
        timeout = max(20.0, float(np.linalg.norm(target)) / max(speed, 1e-3) + 10.0)
        try:
            await await_projectairsim_task(
                drone.move_to_position_async(
                    north=float(target[0]),
                    east=float(target[1]),
                    down=float(target[2]),
                    velocity=float(speed),
                    timeout_sec=timeout,
                )
            )
            return
        except TypeError:
            pass

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        kinematics = await get_kinematics(drone)
        state, _ = extract_kinematic_state(kinematics, 0.0, None, config.dt)
        current = np.array(state[1:4], dtype=float)
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= config.waypoint_tolerance_m:
            return
        velocity = delta / max(distance, 1e-6) * min(speed, distance / config.move_step_sec)
        await move_by_velocity(drone, velocity, config.move_step_sec)


async def prepare_drone(drone: Any) -> None:
    if hasattr(drone, "enable_api_control"):
        drone.enable_api_control()
    if hasattr(drone, "arm"):
        drone.arm()
    if hasattr(drone, "takeoff_async"):
        await await_projectairsim_task(drone.takeoff_async(timeout_sec=20))


async def shutdown_drone(drone: Any) -> None:
    if hasattr(drone, "hover_async"):
        await await_projectairsim_task(drone.hover_async())
    if hasattr(drone, "land_async"):
        await await_projectairsim_task(drone.land_async(timeout_sec=60))
    if hasattr(drone, "disarm"):
        drone.disarm()
    if hasattr(drone, "disable_api_control"):
        drone.disable_api_control()


async def collect_dataset(config: DataGenConfig, seed: int | None = None) -> None:
    from projectairsim import Drone, ProjectAirSimClient, World

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = TrajectoryGenerator(config, seed=seed)

    client = ProjectAirSimClient()
    client.connect()
    try:
        world = World(client, config.scene_config, delay_after_load_sec=2)
        drone = Drone(client, world, config.drone_name)
        await prepare_drone(drone)

        with config.output_path.open("w", encoding="utf-8") as f:
            for episode_id in tqdm(range(config.num_episodes), desc="Collecting episodes"):
                episode = generator.sample_episode()
                episode["episode_id"] = episode_id
                await move_to_start(
                    drone,
                    episode["start_position"],
                    config,
                    speed=float(episode["params"]["speed"]),
                )
                episode = await fly_episode(drone, episode, config)
                f.write(json.dumps(episode) + "\n")
                f.flush()

        await shutdown_drone(drone)
    finally:
        client.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect UAV kinematic trajectories from ProjectAirSim.")
    parser.add_argument("--num-episodes", type=int, default=DataGenConfig.num_episodes)
    parser.add_argument("--sample-rate", type=float, default=DataGenConfig.sample_rate_hz)
    parser.add_argument("--output", type=Path, default=DataGenConfig.output_path)
    parser.add_argument("--scene-config", type=str, default=DataGenConfig.scene_config)
    parser.add_argument("--drone-name", type=str, default=DataGenConfig.drone_name)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataGenConfig(
        num_episodes=args.num_episodes,
        sample_rate_hz=args.sample_rate,
        output_path=args.output,
        scene_config=args.scene_config,
        drone_name=args.drone_name,
    )
    asyncio.run(collect_dataset(config, seed=args.seed))


if __name__ == "__main__":
    main()
