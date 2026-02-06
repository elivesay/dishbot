"""
Configuration module for DishBot.

This module provides configuration classes using Pydantic for type-safe
configuration management across all system components.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CameraConfig(BaseModel):
    """Camera intrinsic and extrinsic parameters."""

    fx: float = Field(default=615.0, description="Focal length x")
    fy: float = Field(default=615.0, description="Focal length y")
    cx: float = Field(default=320.0, description="Principal point x")
    cy: float = Field(default=240.0, description="Principal point y")
    width: int = Field(default=640, description="Image width in pixels")
    height: int = Field(default=480, description="Image height in pixels")
    depth_scale: float = Field(default=1000.0, description="Depth scale factor")
    min_depth: float = Field(default=0.1, description="Minimum depth in meters")
    max_depth: float = Field(default=2.0, description="Maximum depth in meters")

    def to_intrinsic_dict(self) -> dict[str, float]:
        """Convert to dictionary format for 3D reconstruction.

        Returns:
            Dictionary containing camera intrinsic parameters.
        """
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


class VisionConfig(BaseModel):
    """Vision system configuration."""

    model_name: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        description="Qwen2-VL model name or path",
    )
    device: str = Field(default="auto", description="Device for model inference")
    torch_dtype: str = Field(default="float16", description="Torch dtype for inference")
    max_new_tokens: int = Field(default=512, description="Maximum tokens for generation")

    # 3D reconstruction parameters
    voxel_size: float = Field(default=0.005, description="Voxel size for downsampling")
    dbscan_eps: float = Field(default=0.02, description="DBSCAN epsilon for clustering")
    dbscan_min_points: int = Field(default=100, description="DBSCAN minimum points")

    # Detection prompts
    dish_detection_prompt: str = Field(
        default="Identify all dishes in this sink and describe their positions, types, and orientations. Include plates, bowls, cups, utensils, and any other dishware.",
        description="Prompt for dish detection",
    )
    grasp_analysis_prompt: str = Field(
        default="Analyze the graspable features of this dish. Identify the best grasp points considering the dish type, orientation, and any obstacles.",
        description="Prompt for grasp analysis",
    )


class GraspConfig(BaseModel):
    """Grasp planning configuration."""

    gripper_width: float = Field(default=0.08, description="Maximum gripper width in meters")
    gripper_depth: float = Field(default=0.04, description="Gripper finger depth in meters")
    approach_distance: float = Field(default=0.1, description="Pre-grasp approach distance")
    flat_object_threshold: float = Field(
        default=0.05, description="Height threshold for flat objects"
    )

    # Grasp scoring weights
    stability_weight: float = Field(default=0.4, description="Weight for grasp stability")
    reachability_weight: float = Field(default=0.3, description="Weight for reachability")
    clearance_weight: float = Field(default=0.3, description="Weight for collision clearance")

    # Rim grasp parameters
    rim_percentile: float = Field(default=95.0, description="Percentile for rim detection")
    num_rim_candidates: int = Field(default=3, description="Number of rim grasp candidates")


class RobotConfig(BaseModel):
    """Robot arm configuration."""

    robot_type: str = Field(default="franka", description="Robot type (franka, ur5, etc.)")
    robot_usd_path: str = Field(
        default="/Isaac/Robots/Franka/franka_alt_fingers.usd",
        description="USD path for robot model",
    )
    end_effector_link: str = Field(
        default="panda_hand", description="End effector link name"
    )

    # Motion planning
    max_velocity: float = Field(default=1.0, description="Maximum joint velocity")
    max_acceleration: float = Field(default=0.5, description="Maximum joint acceleration")
    planning_time: float = Field(default=5.0, description="Maximum planning time in seconds")

    # Control gains
    position_gain: float = Field(default=100.0, description="Position control gain")
    velocity_gain: float = Field(default=10.0, description="Velocity control gain")


class SimulationConfig(BaseModel):
    """Isaac Sim simulation configuration."""

    # Scene setup
    headless: bool = Field(default=False, description="Run simulation headless")
    physics_dt: float = Field(default=1 / 120.0, description="Physics timestep")
    rendering_dt: float = Field(default=1 / 60.0, description="Rendering timestep")

    # Environment
    sink_position: tuple[float, float, float] = Field(
        default=(0.5, 0.0, 0.8), description="Sink center position"
    )
    sink_dimensions: tuple[float, float, float] = Field(
        default=(0.6, 0.4, 0.2), description="Sink dimensions (l, w, h)"
    )
    camera_position: tuple[float, float, float] = Field(
        default=(0.5, 0.0, 1.5), description="Camera position above sink"
    )

    # Domain randomization
    enable_domain_randomization: bool = Field(
        default=True, description="Enable domain randomization"
    )
    lighting_variation: float = Field(default=0.3, description="Lighting intensity variation")
    texture_variation: bool = Field(default=True, description="Enable texture randomization")

    # Dish spawning
    min_dishes: int = Field(default=3, description="Minimum dishes per scene")
    max_dishes: int = Field(default=10, description="Maximum dishes per scene")
    dish_types: list[str] = Field(
        default=["plate", "bowl", "cup", "mug", "utensil"],
        description="Available dish types",
    )


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    # Data
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    num_training_samples: int = Field(default=10000, description="Number of training samples")
    validation_split: float = Field(default=0.1, description="Validation split ratio")

    # Model
    hidden_dim: int = Field(default=256, description="Hidden layer dimension")
    num_layers: int = Field(default=3, description="Number of hidden layers")
    dropout: float = Field(default=0.1, description="Dropout probability")

    # Training
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    num_epochs: int = Field(default=100, description="Number of training epochs")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")

    # Checkpointing
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Checkpoint directory")
    save_every: int = Field(default=10, description="Save checkpoint every N epochs")


class DishBotConfig(BaseSettings):
    """Main configuration for the DishBot system.

    This class aggregates all component configurations and supports
    loading from environment variables with the DISHBOT_ prefix.
    """

    # Component configurations
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    grasp: GraspConfig = Field(default_factory=GraspConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")

    # General
    seed: int = Field(default=42, description="Random seed for reproducibility")
    debug: bool = Field(default=False, description="Enable debug mode")

    class Config:
        """Pydantic configuration."""

        env_prefix = "DISHBOT_"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DishBotConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            DishBotConfig instance with loaded values.
        """
        from omegaconf import OmegaConf

        yaml_config = OmegaConf.load(path)
        dict_config = OmegaConf.to_container(yaml_config, resolve=True)
        return cls(**dict_config)  # type: ignore[arg-type]

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        from omegaconf import OmegaConf

        config_dict = self.model_dump()
        # Convert Path objects to strings for YAML serialization
        config_dict = self._convert_paths(config_dict)
        OmegaConf.save(OmegaConf.create(config_dict), path)

    def _convert_paths(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings.

        Args:
            obj: Object to convert.

        Returns:
            Object with Path instances converted to strings.
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths(item) for item in obj]
        return obj
