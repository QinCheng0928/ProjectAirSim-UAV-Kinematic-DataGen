# ProjectAirSim UAV Kinematic DataGen

一个简洁的 Python 数据生成项目，用于基于 ProjectAirSim 仿真环境采集无人机运动学轨迹，并导出为 Hugging Face Datasets 友好的格式。

项目默认使用 NED 坐标系：`x/north`、`y/east`、`z/down`，因此飞行高度通常表现为负 `z`。ProjectAirSim API 输入和输出默认采用 SI 单位，角度相关参数采用弧度。

## 数据字段

每一行 raw JSONL 表示一个 episode：

```json
{
  "episode_id": 0,
  "trajectory_type": "periodic",
  "start_position": [12.3, -8.1, -9.5],
  "goal_position": [34.0, 11.2, -8.8],
  "waypoints": [[12.3, -8.1, -9.5], [15.0, -6.0, -9.4]],
  "obstacle_position": null,
  "dt": 0.1,
  "params": {"speed": 4.5, "duration": 8.0, "noise_std": 0.12},
  "states": [[0.0, 12.3, -8.1, -9.5, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
}
```

`states` 是主要训练数据，每个时间步为：

```text
[t, x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az]
```

字段含义：

- `t`: episode 内时间戳，单位秒，默认约 10Hz。
- `x, y, z`: NED 位置，单位米。
- `qw, qx, qy, qz`: 姿态四元数。
- `vx, vy, vz`: 线速度，单位 m/s。
- `wx, wy, wz`: 角速度，单位 rad/s。
- `ax, ay, az`: 线加速度，优先读取仿真器字段，不存在时由速度差分估计。

## 轨迹类型

`scripts/trajectory_generator.py` 会为每个 episode 随机选择轨迹模式，并随机化起点、目标点、高度、路径尺度、速度、持续时间、yaw 朝向和轻微扰动。

已实现类型：

- `line`: 直线到达，包含匀速、加速或减速 profile metadata。
- `turn`: 圆弧或 S 形转弯到达。
- `avoidance`: 带障碍物 metadata 的 Bezier 绕行轨迹。
- `altitude_change`: 上升、下降或斜向高度变化。
- `hover_arrival`: 到达目标点后悬停采样。
- `out_and_back`: 往返、折线和多段 waypoint 路径。
- `sharp_change`: 方形、锯齿形和急转弯路径。
- `periodic`: 正弦或近似 8 字形轨迹后到达。
- `spiral`: 螺旋上升或下降。
- `smooth_random`: 随机平滑曲线。
- `combined`: 直线、转弯、高度变化和悬停组合。

默认 workspace：

```text
x in [-50, 50]
y in [-50, 50]
z in [-30, -3]
```

可在 `scripts/config.py` 中调整采样频率、episode 数量、速度范围、轨迹长度和 workspace。

## ProjectAirSim 环境准备

先准备并启动 ProjectAirSim 仿真环境，确保 Python client 能连接到正在运行的 simulator。

ProjectAirSim 仓库：

```bash
git clone https://github.com/iamaisim/ProjectAirSim.git
```

安装本项目依赖：

```bash
pip install -r requirements.txt
```

安装 ProjectAirSim Python client。具体路径以你的本地 ProjectAirSim 仓库为准，通常类似：

```bash
pip install -e path/to/ProjectAirSim/client/python
```

ProjectAirSim 官方文档说明其 Python API 使用：

```python
from projectairsim import ProjectAirSimClient, World, Drone
```

并通过 `World(client, "scene_basic_drone.jsonc", delay_after_load_sec=2)` 加载场景，通过 `Drone(client, world, "Drone1")` 控制无人机。

## Scene Config

仓库中的 `sim_config/scene_config.jsonc` 只是占位文件。运行采集前，请替换为真实 ProjectAirSim scene config，或直接通过命令行传入官方示例 scene config：

```bash
python -m scripts.collect_dataset \
  --num-episodes 200 \
  --sample-rate 10 \
  --output data/raw/episodes.jsonl \
  --scene-config path/to/scene_basic_drone.jsonc \
  --drone-name Drone1
```

## 数据采集

启动 ProjectAirSim simulator 后运行：

```bash
python -m scripts.collect_dataset \
  --num-episodes 200 \
  --sample-rate 10 \
  --output data/raw/episodes.jsonl \
  --scene-config sim_config/scene_config.jsonc \
  --drone-name Drone1 \
  --seed 42
```

采集流程：

1. 连接 ProjectAirSim。
2. 加载 scene config。
3. 初始化 `Drone1`。
4. enable API control、arm、takeoff。
5. 每个 episode 随机生成轨迹。
6. 先飞到该 episode 的随机起点。
7. 使用 `move_by_velocity_async` 沿 waypoints 飞行。
8. 以约 10Hz 调用 kinematics API 采样。
9. 每条 episode 写入 `data/raw/episodes.jsonl`。

采集脚本优先尝试 `get_ground_truth_kinematics()`，然后兼容 `get_estimated_kinematics()`、`get_kinematics()` 等常见命名。

## 导出 Hugging Face Dataset

```bash
python -m scripts.export_hf_dataset \
  --input data/raw/episodes.jsonl \
  --output data/hf_dataset
```

该命令会生成：

- `data/hf_dataset/`: `datasets.Dataset.save_to_disk()` 输出。
- `data/hf_dataset_export/train.jsonl`: 可直接上传的 JSONL。
- `data/hf_dataset_export/README.md`: 数据集卡片草稿。

## 手动上传到 Hugging Face

安装并登录：

```bash
pip install huggingface_hub
huggingface-cli login
```

创建数据集仓库后，可以上传导出的 JSONL 和 README：

```bash
huggingface-cli upload your-username/projectairsim-uav-kinematics \
  data/hf_dataset_export \
  --repo-type dataset
```

也可以在 Python 中加载本地磁盘格式：

```python
from datasets import load_from_disk

dataset = load_from_disk("data/hf_dataset")
print(dataset[0])
```

## 代码结构

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── trajectory_generator.py
│   ├── collect_dataset.py
│   └── export_hf_dataset.py
├── sim_config/
│   └── scene_config.jsonc
└── data/
    ├── raw/
    ├── hf_dataset/
    └── hf_dataset_export/
```

## 参考

- ProjectAirSim GitHub: https://github.com/iamaisim/ProjectAirSim
- ProjectAirSim API 文档: https://github.com/iamaisim/ProjectAirSim/blob/main/docs/api.md
