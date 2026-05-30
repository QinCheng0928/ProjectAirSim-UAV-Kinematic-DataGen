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

from config import DataGenConfig
from trajectory_generator import TrajectoryGenerator

from projectairsim import Drone, ProjectAirSimClient, World

def xyz_from(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    return [
        float(value.get("x", default[0])),
        float(value.get("y", default[1])),
        float(value.get("z", default[2])),
    ]

def quat_from(value: Any) -> list[float]:
    return [
        float(value.get("w", 1.0)),
        float(value.get("x", 0.0)),
        float(value.get("y", 0.0)),
        float(value.get("z", 0.0)),
    ]

def extract_kinematic_state(
    kinematics: Any,
    t: float,
) -> tuple[list[float], list[float]]:
    position = kinematics["pose"]["position"]
    orientation = kinematics["pose"]["orientation"]
    linear_velocity = kinematics["twist"]["linear"]
    angular_velocity = kinematics["twist"]["angular"]
    linear_acceleration = kinematics["accels"]["linear"]

    pos = xyz_from(position)
    quat = quat_from(orientation)
    vel = xyz_from(linear_velocity)
    omega = xyz_from(angular_velocity)
    acc = xyz_from(linear_acceleration)

    state = [float(t), *pos, *quat, *vel, *omega, *acc]
    return [round(float(x), 6) for x in state]

async def move_to_position_with_sampling(
    drone: Any,
    target: list[float],
    speed: float,
    sample_rate: float,
    start_time: float,
    states: list[list[float]],
    move_step_sec: float,
    tolerance_m: float,
) -> list[float] | None:
    dt = 1.0 / sample_rate
    target_np = np.array(target, dtype=float)
    while True:
        kinematics = drone.get_ground_truth_kinematics()
        elapsed = time.monotonic() - start_time
        state = extract_kinematic_state(kinematics, elapsed)
        states.append(state)
        current = np.array(state[1:4], dtype=float)
        delta = target_np - current
        distance = float(np.linalg.norm(delta))
        if distance <= tolerance_m:
            return

        direction = delta / max(distance, 1e-6)
        command_speed = min(speed, distance / max(move_step_sec, 1e-3))
        velocity = direction * command_speed

        yaw = math.atan2(float(velocity[1]), float(velocity[0]))
        kwargs = {
            "v_north": float(velocity[0]),
            "v_east": float(velocity[1]),
            "v_down": float(velocity[2]),
            "duration": float(dt),
            "yaw_is_rate": False,
            "yaw": float(yaw),
        }
        move_task = await drone.move_by_velocity_async(**kwargs)
        await move_task


async def hover_with_sampling(
    drone: Any,
    seconds: float,
    sample_rate: float,
    start_time: float,
    states: list[list[float]],
) -> list[float] | None:
    hover_task = await drone.hover_async()
    await hover_task
    dt = 1.0 / sample_rate
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        kinematics = drone.get_ground_truth_kinematics()
        elapsed = time.monotonic() - start_time
        state = extract_kinematic_state(kinematics, elapsed)
        states.append(state)
        await asyncio.sleep(dt)


async def fly_episode(drone: Any, episode: dict, config: DataGenConfig) -> dict:
    states: list[list[float]] = []
    start_time = time.monotonic()
    speed = float(episode["params"]["speed"])

    for waypoint in episode["waypoints"]:
        await move_to_position_with_sampling(
            drone=drone,
            target=waypoint,
            speed=speed,
            sample_rate=config.sample_rate_hz,
            start_time=start_time,
            states=states,
            move_step_sec=config.move_step_sec,
            tolerance_m=config.waypoint_tolerance_m,
        )

    if episode["trajectory_type"] in {"hover_arrival", "combined"}:
        hover_time = float(episode["params"].get("hover_time", 2.0))
        await hover_with_sampling(drone, hover_time, config.sample_rate_hz, start_time, states)

    episode["states"] = states
    return episode


async def move_to_start(drone: Any, start_position: list[float], config: DataGenConfig, speed: float) -> None:
    target = np.array(start_position, dtype=float)
    timeout = max(20.0, float(np.linalg.norm(target)) / max(speed, 1e-3) + 10.0)

    init_move_task = await drone.move_to_position_async(
                north=float(target[0]),
                east=float(target[1]),
                down=float(target[2]),
                velocity=float(speed),
                timeout_sec=timeout,
            )
    await init_move_task

async def prepare_drone(drone: Any) -> None:
    drone.enable_api_control()
    drone.arm()
    takeoff_task = await drone.takeoff_async()
    await takeoff_task


async def shutdown_drone(drone: Any) -> None:
    land_task = await drone.land_async()
    await land_task
    drone.disarm()
    drone.disable_api_control()

async def collect_dataset(config: DataGenConfig, seed: int | None = None) -> None:
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
                await move_to_start(drone, episode["start_position"], config, speed=float(episode["params"]["speed"]))
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

# python scripts/collect_dataset.py --num-episodes 500 --sample-rate 10 --output data/raw/episodes.jsonl --scene-config scene_drone_classic.jsonc --drone-name Drone1 --seed 42