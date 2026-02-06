"""
NVIDIA Isaac Sim environment for DishBot.

This module provides the simulation environment for training and testing
the robotic dishwashing system using NVIDIA Isaac Sim.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from dishbot.config import CameraConfig, RobotConfig, SimulationConfig


@dataclass
class SimulationObservation:
    """Container for simulation observations."""

    rgb_image: np.ndarray
    depth_image: np.ndarray
    robot_joint_positions: np.ndarray
    robot_joint_velocities: np.ndarray
    end_effector_pose: np.ndarray  # 4x4 transformation matrix
    gripper_state: float  # 0=closed, 1=open
    timestamp: float


@dataclass
class DishState:
    """State of a dish in the simulation."""

    dish_id: str
    dish_type: str
    position: np.ndarray
    orientation: np.ndarray  # Quaternion (x, y, z, w)
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    is_grasped: bool = False


class IsaacSimDishwashingEnv:
    """Isaac Sim environment for dishwashing robot simulation.

    This class provides:
    - Sink scene setup with configurable dishes
    - RGBD camera observations
    - Robot control interface (Franka Panda)
    - Domain randomization for sim-to-real transfer
    - Synthetic data generation for training
    """

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        robot_config: RobotConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Initialize the Isaac Sim environment.

        Args:
            sim_config: Simulation configuration. Uses defaults if None.
            robot_config: Robot configuration. Uses defaults if None.
            camera_config: Camera configuration. Uses defaults if None.

        Note:
            Actual Isaac Sim initialization requires the isaacsim package
            which must be installed via NVIDIA Omniverse.
        """
        self.sim_config = sim_config or SimulationConfig()
        self.robot_config = robot_config or RobotConfig()
        self.camera_config = camera_config or CameraConfig()

        self._simulation_app = None
        self._world = None
        self._robot = None
        self._camera = None
        self._dishes: dict[str, Any] = {}
        self._is_initialized = False

        logger.info("IsaacSimDishwashingEnv created (not yet initialized)")

    def initialize(self) -> None:
        """Initialize the Isaac Sim environment.

        This method should be called after the Isaac Sim app is running.

        Raises:
            ImportError: If Isaac Sim is not available.
            RuntimeError: If initialization fails.
        """
        if self._is_initialized:
            logger.warning("Environment already initialized")
            return

        try:
            # Import Isaac Sim modules
            # These imports require Isaac Sim to be installed via Omniverse
            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import add_reference_to_stage

            logger.info("Initializing Isaac Sim world...")

            # Create the simulation world
            self._world = World(
                physics_dt=self.sim_config.physics_dt,
                rendering_dt=self.sim_config.rendering_dt,
            )

            # Set up the scene
            self._setup_sink_scene()
            self._add_robot()
            self._add_camera()

            # Reset and step once to initialize physics
            self._world.reset()

            self._is_initialized = True
            logger.info("Isaac Sim environment initialized successfully")

        except ImportError as e:
            logger.error(
                "Isaac Sim is not available. Install via NVIDIA Omniverse or "
                "use the mock environment for development."
            )
            raise ImportError(
                "Isaac Sim not found. Please install via NVIDIA Omniverse."
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Isaac Sim: {e}")
            raise RuntimeError(f"Isaac Sim initialization failed: {e}") from e

    def _setup_sink_scene(self) -> None:
        """Set up the sink environment scene.

        Creates the sink, countertop, and surrounding environment.
        """
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.core.prims import XFormPrim
        from omni.isaac.core.utils.prims import create_prim

        logger.info("Setting up sink scene...")

        # Create ground plane
        create_prim(
            prim_path="/World/Ground",
            prim_type="Xform",
        )

        # Create sink (simplified as a box for now)
        sink_pos = self.sim_config.sink_position
        sink_dims = self.sim_config.sink_dimensions

        # Sink basin
        self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Sink/Basin",
                name="sink_basin",
                position=np.array(sink_pos),
                scale=np.array(sink_dims),
                color=np.array([0.8, 0.8, 0.8]),
            )
        )

        # Countertop
        counter_height = sink_pos[2] - sink_dims[2] / 2
        self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Countertop",
                name="countertop",
                position=np.array([sink_pos[0], sink_pos[1], counter_height - 0.02]),
                scale=np.array([1.2, 0.8, 0.04]),
                color=np.array([0.6, 0.5, 0.4]),
            )
        )

        logger.info("Sink scene setup complete")

    def _add_robot(self) -> None:
        """Add the robot arm to the scene."""
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage

        logger.info(f"Adding robot: {self.robot_config.robot_type}")

        # Add Franka robot from USD
        robot_prim_path = "/World/Robot"
        add_reference_to_stage(
            usd_path=self.robot_config.robot_usd_path,
            prim_path=robot_prim_path,
        )

        self._robot = self._world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name="franka_robot",
            )
        )

        # Position robot at the edge of the countertop
        sink_pos = self.sim_config.sink_position
        robot_position = np.array([sink_pos[0] - 0.5, sink_pos[1], 0.0])
        self._robot.set_world_pose(position=robot_position)

        logger.info("Robot added to scene")

    def _add_camera(self) -> None:
        """Add an RGBD camera above the sink."""
        from omni.isaac.sensor import Camera

        logger.info("Adding RGBD camera...")

        camera_position = self.sim_config.camera_position
        camera_prim_path = "/World/Camera"

        self._camera = Camera(
            prim_path=camera_prim_path,
            frequency=30,
            resolution=(self.camera_config.width, self.camera_config.height),
            position=np.array(camera_position),
            orientation=np.array([0, 0.7071, 0, 0.7071]),  # Looking down
        )

        self._camera.initialize()

        # Enable depth output
        self._camera.add_depth_to_frame()

        logger.info("Camera added to scene")

    def spawn_dishes(
        self,
        num_dishes: int | None = None,
        dish_types: list[str] | None = None,
    ) -> list[str]:
        """Spawn random dishes in the sink.

        Args:
            num_dishes: Number of dishes to spawn. Uses random in config range if None.
            dish_types: Types of dishes to spawn. Uses config defaults if None.

        Returns:
            List of spawned dish IDs.
        """
        from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
        from omni.isaac.core.utils.prims import create_prim

        if num_dishes is None:
            num_dishes = np.random.randint(
                self.sim_config.min_dishes,
                self.sim_config.max_dishes + 1,
            )

        dish_types = dish_types or self.sim_config.dish_types

        spawned_ids: list[str] = []
        sink_pos = np.array(self.sim_config.sink_position)
        sink_dims = np.array(self.sim_config.sink_dimensions)

        logger.info(f"Spawning {num_dishes} dishes...")

        for i in range(num_dishes):
            dish_type = np.random.choice(dish_types)
            dish_id = f"dish_{i}_{dish_type}"

            # Random position within sink
            pos_offset = np.array([
                np.random.uniform(-sink_dims[0] / 3, sink_dims[0] / 3),
                np.random.uniform(-sink_dims[1] / 3, sink_dims[1] / 3),
                np.random.uniform(0.05, 0.15),
            ])
            position = sink_pos + pos_offset

            # Create dish based on type (simplified geometry)
            prim_path = f"/World/Dishes/{dish_id}"

            if dish_type == "plate":
                # Flat cylinder approximation
                self._world.scene.add(
                    DynamicCuboid(
                        prim_path=prim_path,
                        name=dish_id,
                        position=position,
                        scale=np.array([0.25, 0.25, 0.02]),
                        color=np.array([1.0, 1.0, 1.0]),
                    )
                )
            elif dish_type == "bowl":
                # Hemisphere approximation
                self._world.scene.add(
                    DynamicSphere(
                        prim_path=prim_path,
                        name=dish_id,
                        position=position,
                        radius=0.08,
                        color=np.array([0.9, 0.9, 0.95]),
                    )
                )
            elif dish_type in ["cup", "mug"]:
                # Cylinder approximation
                self._world.scene.add(
                    DynamicCuboid(
                        prim_path=prim_path,
                        name=dish_id,
                        position=position,
                        scale=np.array([0.08, 0.08, 0.12]),
                        color=np.array([0.8, 0.6, 0.4]),
                    )
                )
            else:  # utensil
                # Thin box
                self._world.scene.add(
                    DynamicCuboid(
                        prim_path=prim_path,
                        name=dish_id,
                        position=position,
                        scale=np.array([0.02, 0.15, 0.01]),
                        color=np.array([0.7, 0.7, 0.7]),
                    )
                )

            self._dishes[dish_id] = {
                "type": dish_type,
                "prim_path": prim_path,
            }
            spawned_ids.append(dish_id)

        logger.info(f"Spawned {len(spawned_ids)} dishes")
        return spawned_ids

    def get_camera_observation(self) -> SimulationObservation:
        """Capture RGBD observation from the camera.

        Returns:
            SimulationObservation containing images and robot state.
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized")

        # Get camera images
        rgb = self._camera.get_rgba()[:, :, :3]  # Remove alpha channel
        depth = self._camera.get_depth()

        # Get robot state
        joint_positions = self._robot.get_joint_positions()
        joint_velocities = self._robot.get_joint_velocities()

        # Get end effector pose
        ee_pose = self._robot.end_effector.get_world_pose()
        ee_transform = np.eye(4)
        ee_transform[:3, 3] = ee_pose[0]  # Position

        # Get gripper state
        gripper_state = self._robot.gripper.get_joint_positions()[0]
        gripper_normalized = gripper_state / 0.04  # Normalize to 0-1

        return SimulationObservation(
            rgb_image=rgb,
            depth_image=depth,
            robot_joint_positions=joint_positions,
            robot_joint_velocities=joint_velocities,
            end_effector_pose=ee_transform,
            gripper_state=gripper_normalized,
            timestamp=self._world.current_time,
        )

    def get_dish_states(self) -> list[DishState]:
        """Get current states of all dishes.

        Returns:
            List of DishState objects.
        """
        states: list[DishState] = []

        for dish_id, dish_info in self._dishes.items():
            prim = self._world.scene.get_object(dish_id)
            if prim is None:
                continue

            pose = prim.get_world_pose()
            velocities = prim.get_velocities()

            state = DishState(
                dish_id=dish_id,
                dish_type=dish_info["type"],
                position=pose[0],
                orientation=pose[1],
                linear_velocity=velocities[0],
                angular_velocity=velocities[1],
            )
            states.append(state)

        return states

    def step(self, render: bool = True) -> None:
        """Advance the simulation by one step.

        Args:
            render: Whether to render the scene.
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized")

        self._world.step(render=render)

    def reset(self) -> SimulationObservation:
        """Reset the environment to initial state.

        Returns:
            Initial observation after reset.
        """
        if not self._is_initialized:
            raise RuntimeError("Environment not initialized")

        # Clear existing dishes
        for dish_id in list(self._dishes.keys()):
            prim = self._world.scene.get_object(dish_id)
            if prim is not None:
                self._world.scene.remove_object(dish_id)
        self._dishes.clear()

        # Reset world
        self._world.reset()

        # Apply domain randomization if enabled
        if self.sim_config.enable_domain_randomization:
            self._apply_domain_randomization()

        # Spawn new dishes
        self.spawn_dishes()

        # Step to settle physics
        for _ in range(10):
            self.step(render=False)

        return self.get_camera_observation()

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization for sim-to-real transfer."""
        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        logger.debug("Applying domain randomization...")

        # Randomize lighting
        if self.sim_config.lighting_variation > 0:
            intensity_scale = 1.0 + np.random.uniform(
                -self.sim_config.lighting_variation,
                self.sim_config.lighting_variation,
            )
            # Apply to scene lights (implementation depends on scene setup)

        # Randomize camera pose slightly
        camera_noise = np.random.normal(0, 0.01, 3)
        new_camera_pos = np.array(self.sim_config.camera_position) + camera_noise
        self._camera.set_world_pose(position=new_camera_pos)

    def apply_domain_randomization(self) -> None:
        """Public method to apply domain randomization.

        Can be called between episodes for additional variation.
        """
        if self.sim_config.enable_domain_randomization:
            self._apply_domain_randomization()

    def close(self) -> None:
        """Clean up the simulation environment."""
        logger.info("Closing Isaac Sim environment...")

        if self._world is not None:
            self._world.clear()
            self._world = None

        self._is_initialized = False
        logger.info("Environment closed")


class MockIsaacSimEnv(IsaacSimDishwashingEnv):
    """Mock Isaac Sim environment for development without Isaac Sim.

    This class provides the same interface as IsaacSimDishwashingEnv but
    generates synthetic data without requiring Isaac Sim to be installed.
    """

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        robot_config: RobotConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Initialize the mock environment.

        Args:
            sim_config: Simulation configuration.
            robot_config: Robot configuration.
            camera_config: Camera configuration.
        """
        super().__init__(sim_config, robot_config, camera_config)
        self._mock_dishes: list[DishState] = []
        self._mock_time = 0.0

    def initialize(self) -> None:
        """Initialize the mock environment."""
        logger.info("Initializing mock Isaac Sim environment...")
        self._is_initialized = True
        logger.info("Mock environment ready")

    def spawn_dishes(
        self,
        num_dishes: int | None = None,
        dish_types: list[str] | None = None,
    ) -> list[str]:
        """Spawn mock dishes.

        Args:
            num_dishes: Number of dishes.
            dish_types: Types of dishes.

        Returns:
            List of dish IDs.
        """
        if num_dishes is None:
            num_dishes = np.random.randint(
                self.sim_config.min_dishes,
                self.sim_config.max_dishes + 1,
            )

        dish_types = dish_types or self.sim_config.dish_types
        spawned_ids: list[str] = []

        sink_pos = np.array(self.sim_config.sink_position)
        sink_dims = np.array(self.sim_config.sink_dimensions)

        self._mock_dishes.clear()

        for i in range(num_dishes):
            dish_type = np.random.choice(dish_types)
            dish_id = f"dish_{i}_{dish_type}"

            pos_offset = np.array([
                np.random.uniform(-sink_dims[0] / 3, sink_dims[0] / 3),
                np.random.uniform(-sink_dims[1] / 3, sink_dims[1] / 3),
                np.random.uniform(0.05, 0.15),
            ])

            state = DishState(
                dish_id=dish_id,
                dish_type=dish_type,
                position=sink_pos + pos_offset,
                orientation=np.array([0, 0, 0, 1]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )

            self._mock_dishes.append(state)
            self._dishes[dish_id] = {"type": dish_type}
            spawned_ids.append(dish_id)

        return spawned_ids

    def get_camera_observation(self) -> SimulationObservation:
        """Get a mock camera observation.

        Returns:
            Mock observation with synthetic data.
        """
        # Generate synthetic RGB image
        rgb = np.random.randint(
            100, 200,
            (self.camera_config.height, self.camera_config.width, 3),
            dtype=np.uint8,
        )

        # Generate synthetic depth image
        depth = np.random.uniform(
            0.5, 1.5,
            (self.camera_config.height, self.camera_config.width),
        ).astype(np.float32)

        # Mock robot state
        joint_positions = np.zeros(7)
        joint_velocities = np.zeros(7)
        ee_pose = np.eye(4)
        ee_pose[:3, 3] = [0.5, 0, 0.5]

        return SimulationObservation(
            rgb_image=rgb,
            depth_image=depth,
            robot_joint_positions=joint_positions,
            robot_joint_velocities=joint_velocities,
            end_effector_pose=ee_pose,
            gripper_state=1.0,
            timestamp=self._mock_time,
        )

    def get_dish_states(self) -> list[DishState]:
        """Get mock dish states.

        Returns:
            List of mock dish states.
        """
        return self._mock_dishes.copy()

    def step(self, render: bool = True) -> None:
        """Advance mock simulation time.

        Args:
            render: Ignored in mock environment.
        """
        self._mock_time += self.sim_config.physics_dt

    def reset(self) -> SimulationObservation:
        """Reset the mock environment.

        Returns:
            Initial mock observation.
        """
        self._mock_time = 0.0
        self._mock_dishes.clear()
        self._dishes.clear()
        self.spawn_dishes()
        return self.get_camera_observation()

    def close(self) -> None:
        """Close the mock environment."""
        self._is_initialized = False
        logger.info("Mock environment closed")


def create_environment(
    use_mock: bool = False,
    **kwargs: Any,
) -> IsaacSimDishwashingEnv:
    """Factory function to create the appropriate environment.

    Args:
        use_mock: If True, creates a mock environment for development.
        **kwargs: Configuration arguments passed to the environment.

    Returns:
        Environment instance (real or mock).
    """
    if use_mock:
        logger.info("Creating mock Isaac Sim environment")
        return MockIsaacSimEnv(**kwargs)

    try:
        import omni.isaac.core  # noqa: F401
        logger.info("Isaac Sim available, creating real environment")
        return IsaacSimDishwashingEnv(**kwargs)
    except ImportError:
        logger.warning(
            "Isaac Sim not available, falling back to mock environment. "
            "Install Isaac Sim via NVIDIA Omniverse for full functionality."
        )
        return MockIsaacSimEnv(**kwargs)
