# DishBot: Robotic Dishwashing with Vision and 3D Reconstruction

A comprehensive robotic dishwashing system combining computer vision, 3D geometry reconstruction, and manipulation planning.

## Overview

DishBot is a complete pipeline for autonomous dishwashing robots that includes:

- **Vision Module**: Semantic dish detection using Qwen2-VL vision-language model
- **3D Reconstruction**: RGBD to point cloud conversion with Open3D
- **Grasp Planning**: Multiple grasp strategies (top-down, side, rim, pinch)
- **Simulation**: NVIDIA Isaac Sim integration with domain randomization
- **Robot Control**: Franka Panda arm control with inverse kinematics
- **Training Pipeline**: ML-based grasp success prediction

## Architecture

```
Pipeline Flow:
Vision (Qwen2-VL) → 3D Reconstruction → Grasp Planning → Sim-to-Real → Robot Control
```

### Components

```
dishbot/
├── src/dishbot/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── vision_module.py         # Qwen2-VL + 3D reconstruction
│   ├── grasp_planning.py        # Grasp pose generation
│   ├── isaac_sim_env.py         # Isaac Sim environment
│   ├── robot_controller.py      # Robot arm control
│   ├── training_pipeline.py     # ML training
│   └── main.py                  # Entry point
├── configs/                      # Configuration files
├── data/                         # Training data
├── checkpoints/                  # Model checkpoints
├── tests/                        # Unit tests
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Installation

### Prerequisites

- **CPython 3.10+** (PyPy is NOT supported - PyTorch requires CPython)
- CUDA 11.8+ (for GPU acceleration on Linux/Windows)
- NVIDIA Isaac Sim (optional, for real simulation)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/dishbot/dishbot.git
cd dishbot

# Create virtual environment with CPython (not PyPy!)
# If you have multiple Python versions, specify the path explicitly:
#   /usr/local/bin/python3.10 -m venv .venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify you're using CPython (should NOT say PyPy)
python --version

# Install PyTorch FIRST (required before installing dishbot)
# See https://pytorch.org/get-started/locally/ for other configurations

# macOS (CPU only):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Linux/Windows with CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Linux/Windows with CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the package
pip install -e .
```

### Install with Development Tools

```bash
# PyTorch must be installed first (see above)
pip install -e ".[dev]"
```

### Install with Visualization

```bash
# PyTorch must be installed first (see above)
pip install -e ".[visualization]"
```

### Install All Optional Dependencies

```bash
# PyTorch must be installed first (see above)
pip install -e ".[all]"
```

### Isaac Sim Installation

Isaac Sim requires Linux with an NVIDIA GPU. There are two options:

#### Option 1: Docker (Recommended for macOS/Windows or Linux without Omniverse)

Use our Docker setup to run Isaac Sim in a container. This is the recommended approach for:
- macOS users (Isaac Sim doesn't support macOS natively)
- Windows users without WSL2 GPU support
- Clean isolation of the simulation environment
- CI/CD pipelines

See [Docker Setup](#docker-setup) below for detailed instructions.

#### Option 2: Native Installation (Linux only)

For native Linux installation with NVIDIA GPU:

1. Download and install [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
2. Install Isaac Sim from the Omniverse Launcher
3. Follow the [Isaac Sim Python setup guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html)

## Docker Setup

The Docker setup allows running Isaac Sim on any machine with access to an NVIDIA GPU, including remote Linux servers.

### Prerequisites

- **NVIDIA GPU**: A CUDA-capable GPU (RTX 2070 or better recommended)
- **NVIDIA Driver**: Version 525.60 or higher
- **Docker**: Version 19.03 or higher
- **NVIDIA Container Toolkit**: For GPU access in containers

#### Installing NVIDIA Container Toolkit (Linux)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Building the Docker Image

```bash
# Build the image
docker compose build

# Or build directly with docker
docker build -t dishbot:latest .
```

### Running with Docker Compose

```bash
# Show available commands
docker compose run dishbot --help

# Run demo with Isaac Sim
docker compose run dishbot demo

# Generate training data
docker compose run dishbot train --generate-data --num-samples 1000

# Train the model
docker compose run training

# Interactive development shell
docker compose run dev

# Start with WebRTC livestream (view at http://localhost:8211)
docker compose up livestream
```

### Running with Docker Directly

```bash
# Run demo
docker run --gpus all -v $(pwd)/data:/workspace/data dishbot:latest demo

# Interactive shell
docker run --gpus all -it dishbot:latest shell

# Run custom Python script
docker run --gpus all -v $(pwd):/workspace dishbot:latest python my_script.py

# With livestream enabled
docker run --gpus all -p 8211:8211 dishbot:latest livestream
```

### Remote Development (macOS → Linux Server)

If you're on macOS and have access to a Linux server with an NVIDIA GPU:

1. **On the Linux server**, clone the repository and build the Docker image:
   ```bash
   git clone https://github.com/dishbot/dishbot.git
   cd dishbot
   docker compose build
   ```

2. **SSH tunnel** for livestream access:
   ```bash
   # On your Mac
   ssh -L 8211:localhost:8211 user@linux-server
   ```

3. **Run with livestream** on the server:
   ```bash
   docker compose up livestream
   ```

4. **View in browser** at `http://localhost:8211` on your Mac

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DISHBOT_HEADLESS` | Run without display | `1` |
| `DISHBOT_LIVESTREAM` | Enable WebRTC streaming | `0` |
| `DISHBOT_CONFIG` | Path to config file | - |

### Volumes

The following directories are mounted by default:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./src` | `/workspace/src` | Source code |
| `./configs` | `/workspace/configs` | Configuration files |
| `./data` | `/workspace/data` | Training data |
| `./checkpoints` | `/workspace/checkpoints` | Model weights |
| `./outputs` | `/workspace/outputs` | Generated outputs |

### Troubleshooting Docker

**GPU not detected:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**Permission issues:**
```bash
# Ensure your user can run Docker
sudo usermod -aG docker $USER
# Log out and back in

# Fix volume permissions
export UID=$(id -u)
export GID=$(id -g)
docker compose build
```

**Out of memory:**
```bash
# Increase shared memory for training
docker compose run --shm-size=16gb training
```

## Quick Start

### Run Demo (Mock Simulation)

```bash
dishbot demo
```

### Run Demo (Real Isaac Sim)

```bash
dishbot demo --real-sim
```

### Generate Training Data

```bash
dishbot train --generate-data --num-samples 10000
```

### Train Grasp Predictor

```bash
dishbot train --num-epochs 100
```

### Evaluate Model

```bash
dishbot evaluate --checkpoint checkpoints/checkpoint_best.pt
```

### Run Full Pipeline

```bash
dishbot run
```

## Configuration

DishBot uses a hierarchical configuration system. You can customize settings via:

1. **YAML files**: Pass `--config path/to/config.yaml`
2. **Environment variables**: Use `DISHBOT_` prefix (e.g., `DISHBOT_VISION__MODEL_NAME`)
3. **Command line arguments**: Override specific settings

### Example Configuration

```yaml
# configs/default.yaml
vision:
  model_name: "Qwen/Qwen2-VL-7B-Instruct"
  device: "auto"
  torch_dtype: "float16"
  voxel_size: 0.005
  dbscan_eps: 0.02

grasp:
  gripper_width: 0.08
  approach_distance: 0.1
  stability_weight: 0.4

simulation:
  headless: false
  physics_dt: 0.00833
  enable_domain_randomization: true

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
```

## Usage Examples

### Python API

```python
from dishbot import (
    DishBotConfig,
    DishVisionSystem,
    GraspPlanner,
    IsaacSimDishwashingEnv,
    DishwashingRobotController,
)

# Initialize configuration
config = DishBotConfig()

# Create vision system
vision = DishVisionSystem(
    vision_config=config.vision,
    camera_config=config.camera,
)

# Reconstruct 3D from RGBD
point_cloud = vision.reconstruct_3d_geometry(rgb_image, depth_image)

# Segment dishes
dishes = vision.segment_individual_dishes(point_cloud)

# Plan grasps
planner = GraspPlanner(config=config.grasp)
for dish in dishes:
    grasps = planner.compute_grasp_candidates(dish)
    best_grasp = planner.select_best_grasp(grasps)
```

### Using the Trained Predictor

```python
from dishbot import GraspTrainingPipeline

# Load trained model
pipeline = GraspTrainingPipeline()
pipeline.load_model("checkpoints/checkpoint_best.pt")

# Predict grasp success
success_prob = pipeline.predict(
    grasp_pose,
    object_center,
    object_extent,
    object_type_id,
)
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
ruff check src/ --fix
```

### Type Checking

```bash
mypy src/dishbot/
```

## Project Structure Details

### Vision Module

The `DishVisionSystem` class provides:
- Semantic dish detection using Qwen2-VL
- RGBD to point cloud conversion
- DBSCAN clustering for dish segmentation
- Dish type classification from geometry

### Grasp Planning

The `GraspPlanner` supports multiple strategies:
- **Top-down**: For flat objects (plates)
- **Side grasp**: For tall objects (cups, glasses)
- **Rim grasp**: For containers (bowls)
- **Pinch grasp**: For thin objects (utensils)

### Simulation

The `IsaacSimDishwashingEnv` provides:
- Configurable sink scene
- Random dish spawning
- RGBD camera observations
- Domain randomization
- Mock environment for development

### Robot Control

The `DishwashingRobotController` offers:
- Forward/inverse kinematics
- Trajectory generation
- Grasp execution
- Pick and place operations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DishBot in your research, please cite:

```bibtex
@software{dishbot2026,
  title={DishBot: Robotic Dishwashing with Vision and 3D Reconstruction},
  year={2026},
  url={https://github.com/dishbot/dishbot}
}
```

## Acknowledgments

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) for vision-language understanding
- [Open3D](http://www.open3d.org/) for 3D geometry processing
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) for robot simulation
- [Franka Emika](https://www.franka.de/) for the Panda robot model
