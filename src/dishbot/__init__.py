"""
DishBot: Robotic Dishwashing System with Vision and 3D Reconstruction.

This package provides a complete pipeline for robotic dishwashing including:
- Vision-based dish detection using Qwen2-VL
- 3D geometry reconstruction from RGBD images
- Grasp planning algorithms
- NVIDIA Isaac Sim integration for simulation
- Robot control interfaces
"""

__version__ = "0.1.0"
__author__ = "DishBot Team"

from dishbot.config import DishBotConfig
from dishbot.vision_module import DishVisionSystem
from dishbot.grasp_planning import GraspPlanner
from dishbot.isaac_sim_env import IsaacSimDishwashingEnv
from dishbot.robot_controller import DishwashingRobotController
from dishbot.training_pipeline import GraspTrainingPipeline

__all__ = [
    "DishBotConfig",
    "DishVisionSystem",
    "GraspPlanner",
    "IsaacSimDishwashingEnv",
    "DishwashingRobotController",
    "GraspTrainingPipeline",
]
