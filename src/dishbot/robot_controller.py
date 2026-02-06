"""
Robot controller module for DishBot.

This module provides control interfaces for the robot arm to execute
grasp and manipulation tasks in the dishwashing environment.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from dishbot.config import RobotConfig
from dishbot.grasp_planning import GraspPose


class ControllerState(Enum):
    """State of the robot controller."""

    IDLE = "idle"
    MOVING = "moving"
    GRASPING = "grasping"
    PLACING = "placing"
    ERROR = "error"


@dataclass
class TrajectoryPoint:
    """A point in a robot trajectory."""

    joint_positions: np.ndarray
    joint_velocities: np.ndarray | None = None
    time_from_start: float = 0.0


@dataclass
class CartesianPose:
    """Cartesian pose for end effector."""

    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # Quaternion [x, y, z, w]

    @classmethod
    def from_transformation_matrix(cls, T: np.ndarray) -> "CartesianPose":
        """Create CartesianPose from 4x4 transformation matrix.

        Args:
            T: 4x4 homogeneous transformation matrix.

        Returns:
            CartesianPose instance.
        """
        position = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3])
        orientation = rotation.as_quat()
        return cls(position=position, orientation=orientation)

    def to_transformation_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        T = np.eye(4)
        T[:3, 3] = self.position
        rotation = Rotation.from_quat(self.orientation)
        T[:3, :3] = rotation.as_matrix()
        return T


class DishwashingRobotController:
    """Controller for robot arm in dishwashing tasks.

    This class provides:
    - Motion planning and trajectory execution
    - Grasp pose execution
    - Pick and place operations
    - Gripper control
    - Safety checks and error handling
    """

    # Franka Panda joint limits (radians)
    JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # Default home configuration
    HOME_POSITION = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def __init__(
        self,
        config: RobotConfig | None = None,
        simulation_env: Any | None = None,
    ) -> None:
        """Initialize the robot controller.

        Args:
            config: Robot configuration. Uses defaults if None.
            simulation_env: Isaac Sim environment instance.
        """
        self.config = config or RobotConfig()
        self.simulation_env = simulation_env

        self._state = ControllerState.IDLE
        self._current_joints = self.HOME_POSITION.copy()
        self._gripper_open = True
        self._robot_articulation = None
        self._gripper_articulation = None

        logger.info(f"Initialized {self.config.robot_type} robot controller")

    def initialize_robot(self) -> bool:
        """Initialize connection to robot in simulation.

        Returns:
            True if initialization successful.
        """
        if self.simulation_env is None:
            logger.warning("No simulation environment provided, using mock mode")
            return True

        try:
            # Get robot articulation from simulation
            self._robot_articulation = self.simulation_env._robot
            if self._robot_articulation is not None:
                self._current_joints = self._robot_articulation.get_joint_positions()
                logger.info("Robot articulation initialized")
                return True

            logger.error("Failed to get robot articulation")
            return False

        except Exception as e:
            logger.error(f"Robot initialization failed: {e}")
            self._state = ControllerState.ERROR
            return False

    @property
    def state(self) -> ControllerState:
        """Get current controller state.

        Returns:
            Current ControllerState.
        """
        return self._state

    @property
    def joint_positions(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            Array of joint positions in radians.
        """
        if self._robot_articulation is not None:
            self._current_joints = self._robot_articulation.get_joint_positions()
        return self._current_joints.copy()

    @property
    def gripper_open(self) -> bool:
        """Check if gripper is open.

        Returns:
            True if gripper is open.
        """
        return self._gripper_open

    def move_to_joint_positions(
        self,
        target_positions: np.ndarray,
        duration: float = 2.0,
    ) -> bool:
        """Move robot to target joint positions.

        Args:
            target_positions: Target joint positions in radians.
            duration: Desired motion duration in seconds.

        Returns:
            True if motion completed successfully.
        """
        if not self._validate_joint_positions(target_positions):
            logger.error("Target positions outside joint limits")
            return False

        self._state = ControllerState.MOVING

        try:
            if self._robot_articulation is not None:
                # Generate trajectory
                trajectory = self._generate_trajectory(
                    self._current_joints,
                    target_positions,
                    duration,
                )

                # Execute trajectory
                for point in trajectory:
                    self._robot_articulation.set_joint_positions(point.joint_positions)
                    if self.simulation_env is not None:
                        self.simulation_env.step()

            # Update internal state
            self._current_joints = target_positions.copy()
            self._state = ControllerState.IDLE
            logger.debug(f"Moved to joint positions: {target_positions}")
            return True

        except Exception as e:
            logger.error(f"Motion failed: {e}")
            self._state = ControllerState.ERROR
            return False

    def move_to_cartesian_pose(
        self,
        pose: CartesianPose,
        duration: float = 2.0,
    ) -> bool:
        """Move end effector to Cartesian pose.

        Args:
            pose: Target Cartesian pose.
            duration: Desired motion duration in seconds.

        Returns:
            True if motion completed successfully.
        """
        # Compute inverse kinematics
        target_joints = self._inverse_kinematics(pose)
        if target_joints is None:
            logger.error("IK solution not found")
            return False

        return self.move_to_joint_positions(target_joints, duration)

    def execute_grasp(
        self,
        grasp_pose: GraspPose,
        pre_grasp_distance: float = 0.1,
    ) -> bool:
        """Execute a complete grasp sequence.

        Args:
            grasp_pose: Target grasp pose.
            pre_grasp_distance: Distance for pre-grasp approach.

        Returns:
            True if grasp executed successfully.
        """
        logger.info(f"Executing {grasp_pose.strategy.value} grasp")
        self._state = ControllerState.GRASPING

        try:
            # Ensure gripper is open
            if not self._gripper_open:
                self.open_gripper()

            # Move to pre-grasp position
            pre_grasp_position = (
                grasp_pose.position - grasp_pose.approach_vector * pre_grasp_distance
            )
            pre_grasp_pose = CartesianPose(
                position=pre_grasp_position,
                orientation=Rotation.from_matrix(grasp_pose.orientation).as_quat(),
            )

            logger.debug("Moving to pre-grasp position")
            if not self.move_to_cartesian_pose(pre_grasp_pose, duration=2.0):
                raise RuntimeError("Failed to reach pre-grasp position")

            # Approach grasp position
            grasp_cartesian = CartesianPose(
                position=grasp_pose.position,
                orientation=Rotation.from_matrix(grasp_pose.orientation).as_quat(),
            )

            logger.debug("Approaching grasp position")
            if not self.move_to_cartesian_pose(grasp_cartesian, duration=1.5):
                raise RuntimeError("Failed to reach grasp position")

            # Close gripper
            logger.debug("Closing gripper")
            if not self.close_gripper(width=grasp_pose.gripper_width):
                raise RuntimeError("Failed to close gripper")

            # Small delay to ensure stable grasp
            if self.simulation_env is not None:
                for _ in range(30):
                    self.simulation_env.step()

            # Lift object
            lift_pose = CartesianPose(
                position=grasp_pose.position + np.array([0, 0, 0.1]),
                orientation=grasp_cartesian.orientation,
            )

            logger.debug("Lifting object")
            if not self.move_to_cartesian_pose(lift_pose, duration=1.0):
                raise RuntimeError("Failed to lift object")

            self._state = ControllerState.IDLE
            logger.info("Grasp executed successfully")
            return True

        except Exception as e:
            logger.error(f"Grasp execution failed: {e}")
            self._state = ControllerState.ERROR
            return False

    def execute_place(
        self,
        place_position: np.ndarray,
        place_orientation: np.ndarray | None = None,
    ) -> bool:
        """Execute a place operation.

        Args:
            place_position: Target place position [x, y, z].
            place_orientation: Target orientation quaternion. Uses current if None.

        Returns:
            True if place executed successfully.
        """
        logger.info("Executing place operation")
        self._state = ControllerState.PLACING

        try:
            # Get current orientation if not specified
            if place_orientation is None:
                current_pose = self.get_end_effector_pose()
                place_orientation = current_pose.orientation

            # Move above place position
            above_place = CartesianPose(
                position=place_position + np.array([0, 0, 0.15]),
                orientation=place_orientation,
            )

            if not self.move_to_cartesian_pose(above_place, duration=1.5):
                raise RuntimeError("Failed to reach above place position")

            # Lower to place position
            place_pose = CartesianPose(
                position=place_position + np.array([0, 0, 0.05]),
                orientation=place_orientation,
            )

            if not self.move_to_cartesian_pose(place_pose, duration=1.0):
                raise RuntimeError("Failed to reach place position")

            # Open gripper to release
            self.open_gripper()

            # Retreat
            retreat_pose = CartesianPose(
                position=place_position + np.array([0, 0, 0.2]),
                orientation=place_orientation,
            )
            self.move_to_cartesian_pose(retreat_pose, duration=1.0)

            self._state = ControllerState.IDLE
            logger.info("Place operation completed")
            return True

        except Exception as e:
            logger.error(f"Place operation failed: {e}")
            self._state = ControllerState.ERROR
            return False

    def pick_and_place(
        self,
        grasp_pose: GraspPose,
        place_position: np.ndarray,
    ) -> bool:
        """Execute a complete pick and place operation.

        Args:
            grasp_pose: Grasp pose for picking.
            place_position: Position to place the object.

        Returns:
            True if operation completed successfully.
        """
        logger.info("Starting pick and place operation")

        # Execute grasp
        if not self.execute_grasp(grasp_pose):
            logger.error("Pick failed")
            return False

        # Execute place
        if not self.execute_place(place_position):
            logger.error("Place failed")
            return False

        logger.info("Pick and place completed successfully")
        return True

    def open_gripper(self, width: float | None = None) -> bool:
        """Open the gripper.

        Args:
            width: Target opening width. Uses max width if None.

        Returns:
            True if operation successful.
        """
        target_width = width or self.config.gripper_width

        try:
            if self._gripper_articulation is not None:
                # Set gripper joint positions (Franka has 2 finger joints)
                finger_position = target_width / 2
                self._gripper_articulation.set_joint_positions(
                    np.array([finger_position, finger_position])
                )

            self._gripper_open = True
            logger.debug(f"Gripper opened to width: {target_width}")
            return True

        except Exception as e:
            logger.error(f"Failed to open gripper: {e}")
            return False

    def close_gripper(self, width: float = 0.0, force: float = 40.0) -> bool:
        """Close the gripper.

        Args:
            width: Target closing width (0 = fully closed).
            force: Grasp force in Newtons.

        Returns:
            True if operation successful.
        """
        try:
            if self._gripper_articulation is not None:
                finger_position = width / 2
                self._gripper_articulation.set_joint_positions(
                    np.array([finger_position, finger_position])
                )

            self._gripper_open = False
            logger.debug(f"Gripper closed to width: {width}")
            return True

        except Exception as e:
            logger.error(f"Failed to close gripper: {e}")
            return False

    def move_to_home(self) -> bool:
        """Move robot to home configuration.

        Returns:
            True if motion successful.
        """
        logger.info("Moving to home position")
        return self.move_to_joint_positions(self.HOME_POSITION, duration=3.0)

    def get_end_effector_pose(self) -> CartesianPose:
        """Get current end effector pose.

        Returns:
            Current CartesianPose of end effector.
        """
        # Use forward kinematics
        T = self._forward_kinematics(self._current_joints)
        return CartesianPose.from_transformation_matrix(T)

    def _validate_joint_positions(self, positions: np.ndarray) -> bool:
        """Validate that joint positions are within limits.

        Args:
            positions: Joint positions to validate.

        Returns:
            True if positions are valid.
        """
        if len(positions) != 7:
            return False

        within_limits = np.all(positions >= self.JOINT_LIMITS_LOWER - 0.01) and np.all(
            positions <= self.JOINT_LIMITS_UPPER + 0.01
        )

        return within_limits

    def _generate_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        num_points: int = 50,
    ) -> list[TrajectoryPoint]:
        """Generate a smooth trajectory between two configurations.

        Args:
            start: Starting joint positions.
            end: Ending joint positions.
            duration: Total trajectory duration.
            num_points: Number of trajectory points.

        Returns:
            List of TrajectoryPoint objects.
        """
        trajectory: list[TrajectoryPoint] = []

        for i in range(num_points):
            # Cubic interpolation for smooth motion
            t = i / (num_points - 1)
            s = 3 * t**2 - 2 * t**3  # Smooth step function

            positions = start + s * (end - start)

            # Compute velocities (finite difference approximation)
            if i > 0:
                dt = duration / (num_points - 1)
                velocities = (positions - trajectory[-1].joint_positions) / dt
            else:
                velocities = np.zeros(7)

            point = TrajectoryPoint(
                joint_positions=positions,
                joint_velocities=velocities,
                time_from_start=t * duration,
            )
            trajectory.append(point)

        return trajectory

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for Franka Panda.

        Args:
            joint_positions: Joint positions in radians.

        Returns:
            4x4 transformation matrix of end effector.
        """
        # Simplified FK using DH parameters (approximate)
        # In production, use robot's actual FK implementation

        # Franka Panda DH parameters (a, d, alpha)
        dh_params = [
            (0, 0.333, 0),
            (0, 0, -np.pi / 2),
            (0, 0.316, np.pi / 2),
            (0.0825, 0, np.pi / 2),
            (-0.0825, 0.384, -np.pi / 2),
            (0, 0, np.pi / 2),
            (0.088, 0, np.pi / 2),
        ]

        T = np.eye(4)

        for i, (a, d, alpha) in enumerate(dh_params):
            theta = joint_positions[i]

            # DH transformation matrix
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            T_i = np.array([
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ])

            T = T @ T_i

        # Add flange to gripper transform
        T_flange = np.eye(4)
        T_flange[2, 3] = 0.1034  # Gripper offset

        return T @ T_flange

    def _inverse_kinematics(
        self,
        target_pose: CartesianPose,
        initial_guess: np.ndarray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> np.ndarray | None:
        """Compute inverse kinematics using iterative method.

        Args:
            target_pose: Target Cartesian pose.
            initial_guess: Initial joint positions. Uses current if None.
            max_iterations: Maximum IK iterations.
            tolerance: Position/orientation tolerance.

        Returns:
            Joint positions or None if no solution found.
        """
        q = initial_guess if initial_guess is not None else self._current_joints.copy()
        target_T = target_pose.to_transformation_matrix()

        for iteration in range(max_iterations):
            # Current pose
            current_T = self._forward_kinematics(q)

            # Position error
            position_error = target_T[:3, 3] - current_T[:3, 3]

            # Orientation error (using rotation matrix difference)
            R_error = target_T[:3, :3] @ current_T[:3, :3].T
            orientation_error = Rotation.from_matrix(R_error).as_rotvec()

            # Combined error
            error = np.concatenate([position_error, orientation_error])

            if np.linalg.norm(error) < tolerance:
                logger.debug(f"IK converged in {iteration + 1} iterations")
                return q

            # Compute Jacobian numerically
            J = self._compute_jacobian(q)

            # Damped least squares solution
            damping = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))

            # Update joint positions
            dq = J_pinv @ error
            q = q + 0.5 * dq  # Step size

            # Clamp to joint limits
            q = np.clip(q, self.JOINT_LIMITS_LOWER, self.JOINT_LIMITS_UPPER)

        logger.warning("IK did not converge")
        return None

    def _compute_jacobian(
        self,
        q: np.ndarray,
        delta: float = 1e-6,
    ) -> np.ndarray:
        """Compute numerical Jacobian.

        Args:
            q: Current joint positions.
            delta: Finite difference step size.

        Returns:
            6x7 Jacobian matrix.
        """
        J = np.zeros((6, 7))
        T_current = self._forward_kinematics(q)

        for i in range(7):
            q_plus = q.copy()
            q_plus[i] += delta

            T_plus = self._forward_kinematics(q_plus)

            # Position derivative
            J[:3, i] = (T_plus[:3, 3] - T_current[:3, 3]) / delta

            # Orientation derivative
            R_diff = T_plus[:3, :3] @ T_current[:3, :3].T
            angle_axis = Rotation.from_matrix(R_diff).as_rotvec()
            J[3:, i] = angle_axis / delta

        return J

    def emergency_stop(self) -> None:
        """Execute emergency stop."""
        logger.warning("Emergency stop activated!")
        self._state = ControllerState.ERROR

        if self._robot_articulation is not None:
            # Set all velocities to zero
            self._robot_articulation.set_joint_velocities(np.zeros(7))

    def recover_from_error(self) -> bool:
        """Attempt to recover from error state.

        Returns:
            True if recovery successful.
        """
        if self._state != ControllerState.ERROR:
            return True

        logger.info("Attempting error recovery...")

        try:
            # Try to move to home
            self._state = ControllerState.IDLE  # Temporarily reset
            if self.move_to_home():
                self.open_gripper()
                logger.info("Recovery successful")
                return True
            else:
                self._state = ControllerState.ERROR
                return False

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self._state = ControllerState.ERROR
            return False
