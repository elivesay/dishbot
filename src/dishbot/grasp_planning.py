"""
Grasp planning module for DishBot.

This module provides algorithms for generating and evaluating grasp poses
for various dish types based on their 3D geometry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import open3d as o3d
from loguru import logger
from scipy.spatial.transform import Rotation

from dishbot.config import GraspConfig


class GraspStrategy(Enum):
    """Available grasp strategies for different dish types."""

    TOP_DOWN = "top_down"  # For flat objects like plates
    SIDE_GRASP = "side_grasp"  # For tall objects like cups
    RIM_GRASP = "rim_grasp"  # For bowls and cups
    PINCH_GRASP = "pinch_grasp"  # For thin objects like utensils


@dataclass
class GraspPose:
    """Represents a candidate grasp pose."""

    position: np.ndarray
    orientation: np.ndarray  # Rotation matrix (3x3)
    approach_vector: np.ndarray
    gripper_width: float
    strategy: GraspStrategy
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_transformation_matrix(self) -> np.ndarray:
        """Convert grasp pose to 4x4 transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        T = np.eye(4)
        T[:3, :3] = self.orientation
        T[:3, 3] = self.position
        return T

    def to_quaternion(self) -> np.ndarray:
        """Get orientation as quaternion (x, y, z, w).

        Returns:
            Quaternion array.
        """
        rot = Rotation.from_matrix(self.orientation)
        return rot.as_quat()


class GraspPlanner:
    """Generates grasping poses for dishes based on 3D geometry.

    This class provides multiple grasp strategies:
    - Top-down grasps for flat objects (plates)
    - Side grasps for tall objects (cups, glasses)
    - Rim grasps for containers (bowls, cups)
    - Pinch grasps for thin objects (utensils)
    """

    def __init__(self, config: GraspConfig | None = None) -> None:
        """Initialize the GraspPlanner.

        Args:
            config: Grasp planning configuration. Uses defaults if None.
        """
        self.config = config or GraspConfig()
        logger.info(f"Initialized GraspPlanner with gripper width: {self.config.gripper_width}m")

    def compute_grasp_candidates(
        self,
        dish_point_cloud: o3d.geometry.PointCloud,
        dish_type: str | None = None,
    ) -> list[GraspPose]:
        """Generate potential grasp poses for a dish.

        Args:
            dish_point_cloud: Point cloud of the dish to grasp.
            dish_type: Optional dish type hint for strategy selection.

        Returns:
            List of candidate GraspPose objects sorted by score.
        """
        # Compute oriented bounding box
        obb = dish_point_cloud.get_oriented_bounding_box()

        # Get principal axes
        R = np.array(obb.R)
        center = np.array(obb.center)
        extent = np.array(obb.extent)

        grasp_candidates: list[GraspPose] = []

        # Determine appropriate strategies based on geometry
        sorted_extent_idx = np.argsort(extent)
        height_idx = sorted_extent_idx[0]  # Smallest dimension
        height = extent[height_idx]

        # Strategy 1: Top-down grasp (for flat objects)
        if height < self.config.flat_object_threshold:
            grasp = self._compute_top_down_grasp(center, R, extent)
            grasp_candidates.append(grasp)

        # Strategy 2: Side grasp (for tall objects)
        if extent[2] > extent[0] and extent[2] > extent[1]:
            grasp = self._compute_side_grasp(center, R, extent)
            grasp_candidates.append(grasp)

        # Strategy 3: Rim grasp (for containers)
        rim_grasps = self._compute_rim_grasps(dish_point_cloud)
        grasp_candidates.extend(rim_grasps)

        # Score and sort candidates
        for grasp in grasp_candidates:
            grasp.score = self._score_grasp(grasp, dish_point_cloud)

        grasp_candidates.sort(key=lambda g: g.score, reverse=True)

        logger.info(f"Generated {len(grasp_candidates)} grasp candidates")
        return grasp_candidates

    def _compute_top_down_grasp(
        self,
        center: np.ndarray,
        orientation: np.ndarray,
        extent: np.ndarray,
    ) -> GraspPose:
        """Compute a top-down grasp for flat objects.

        Args:
            center: Center position of the object.
            orientation: Rotation matrix of the oriented bounding box.
            extent: Dimensions of the bounding box.

        Returns:
            Top-down GraspPose.
        """
        # Approach from above with gripper pointing down
        approach_position = center + np.array([0, 0, self.config.approach_distance])

        # Gripper orientation: z-axis pointing down, x-axis along smallest horizontal
        z_axis = np.array([0, 0, -1])

        # Choose gripper x-axis perpendicular to approach
        x_axis = orientation[:, np.argmin(extent[:2])]
        y_axis = np.cross(z_axis, x_axis)

        grasp_orientation = np.column_stack([x_axis, y_axis, z_axis])

        return GraspPose(
            position=approach_position,
            orientation=grasp_orientation,
            approach_vector=np.array([0, 0, -1]),
            gripper_width=self.config.gripper_width * 0.8,
            strategy=GraspStrategy.TOP_DOWN,
            metadata={"target_position": center},
        )

    def _compute_side_grasp(
        self,
        center: np.ndarray,
        orientation: np.ndarray,
        extent: np.ndarray,
    ) -> GraspPose:
        """Compute a side grasp for tall objects.

        Args:
            center: Center position of the object.
            orientation: Rotation matrix of the oriented bounding box.
            extent: Dimensions of the bounding box.

        Returns:
            Side GraspPose.
        """
        # Find the narrowest horizontal dimension for grasping
        horizontal_extents = extent[:2]
        min_dim_idx = np.argmin(horizontal_extents)

        # Approach direction is along the narrowest dimension
        approach_vector = orientation[:, min_dim_idx]

        # Position gripper to approach from side
        approach_position = center + approach_vector * (
            extent[min_dim_idx] / 2 + self.config.approach_distance
        )

        # Gripper orientation
        z_axis = -approach_vector  # Pointing toward object
        y_axis = np.array([0, 0, 1])  # Gripper up
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        grasp_orientation = np.column_stack([x_axis, y_axis, z_axis])

        return GraspPose(
            position=approach_position,
            orientation=grasp_orientation,
            approach_vector=-approach_vector,
            gripper_width=extent[min_dim_idx],
            strategy=GraspStrategy.SIDE_GRASP,
            metadata={"target_position": center},
        )

    def _compute_rim_grasps(
        self,
        point_cloud: o3d.geometry.PointCloud,
    ) -> list[GraspPose]:
        """Compute rim grasps for containers like bowls and cups.

        Args:
            point_cloud: Point cloud of the dish.

        Returns:
            List of rim GraspPose candidates.
        """
        points = np.asarray(point_cloud.points)

        if len(points) < 10:
            return []

        # Find highest points (rim candidates)
        z_threshold = np.percentile(points[:, 2], self.config.rim_percentile)
        rim_points = points[points[:, 2] > z_threshold]

        if len(rim_points) < 10:
            return []

        # Compute center for approach direction calculation
        center_xy = np.mean(points[:, :2], axis=0)
        rim_center = np.mean(rim_points, axis=0)

        grasp_candidates: list[GraspPose] = []

        # Sample grasp points on rim
        num_candidates = min(self.config.num_rim_candidates, len(rim_points))
        indices = np.random.choice(len(rim_points), num_candidates, replace=False)

        for idx in indices:
            point = rim_points[idx]

            # Compute approach direction (radially inward)
            approach_2d = center_xy - point[:2]
            approach_2d_norm = np.linalg.norm(approach_2d)

            if approach_2d_norm < 1e-6:
                continue

            approach_2d /= approach_2d_norm

            # Approach vector with slight downward angle
            approach_vector = np.array([approach_2d[0], approach_2d[1], -0.3])
            approach_vector /= np.linalg.norm(approach_vector)

            # Position gripper at approach distance from rim
            position = point - approach_vector * self.config.approach_distance

            # Compute gripper orientation
            z_axis = approach_vector
            y_axis = np.array([0, 0, 1])
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            grasp_orientation = np.column_stack([x_axis, y_axis, z_axis])

            grasp = GraspPose(
                position=position,
                orientation=grasp_orientation,
                approach_vector=approach_vector,
                gripper_width=self.config.gripper_width * 0.6,
                strategy=GraspStrategy.RIM_GRASP,
                metadata={
                    "rim_point": point,
                    "rim_center": rim_center,
                },
            )
            grasp_candidates.append(grasp)

        return grasp_candidates

    def compute_pinch_grasp(
        self,
        point_cloud: o3d.geometry.PointCloud,
    ) -> GraspPose | None:
        """Compute a pinch grasp for thin objects like utensils.

        Args:
            point_cloud: Point cloud of the thin object.

        Returns:
            Pinch GraspPose or None if not applicable.
        """
        obb = point_cloud.get_oriented_bounding_box()
        center = np.array(obb.center)
        extent = np.array(obb.extent)
        R = np.array(obb.R)

        # Check if object is thin enough for pinch grasp
        sorted_extent = np.sort(extent)
        if sorted_extent[0] > 0.02:  # Too thick for pinch
            return None

        # Grasp along the longest axis
        longest_axis_idx = np.argmax(extent)
        grasp_axis = R[:, longest_axis_idx]

        # Approach perpendicular to longest axis
        approach_vector = R[:, np.argmin(extent)]

        position = center + approach_vector * self.config.approach_distance

        # Gripper orientation
        z_axis = -approach_vector
        x_axis = grasp_axis
        y_axis = np.cross(z_axis, x_axis)

        grasp_orientation = np.column_stack([x_axis, y_axis, z_axis])

        return GraspPose(
            position=position,
            orientation=grasp_orientation,
            approach_vector=-approach_vector,
            gripper_width=sorted_extent[0] + 0.01,  # Slight margin
            strategy=GraspStrategy.PINCH_GRASP,
            metadata={"grasp_axis": grasp_axis},
        )

    def _score_grasp(
        self,
        grasp: GraspPose,
        point_cloud: o3d.geometry.PointCloud,
    ) -> float:
        """Score a grasp candidate based on multiple criteria.

        Args:
            grasp: The grasp pose to score.
            point_cloud: The target object's point cloud.

        Returns:
            Score between 0 and 1, higher is better.
        """
        scores = []

        # Stability score: prefer vertical approach
        vertical_component = abs(grasp.approach_vector[2])
        stability_score = vertical_component * self.config.stability_weight
        scores.append(stability_score)

        # Reachability score: penalize positions too high or too low
        target_height = 0.4  # Optimal working height in meters
        height_deviation = abs(grasp.position[2] - target_height)
        reachability_score = (
            max(0, 1 - height_deviation / 0.5) * self.config.reachability_weight
        )
        scores.append(reachability_score)

        # Clearance score: check for potential collisions (simplified)
        # In a real system, this would check against the full scene
        clearance_score = self.config.clearance_weight

        # Reduce score for rim grasps (more challenging)
        if grasp.strategy == GraspStrategy.RIM_GRASP:
            clearance_score *= 0.8

        scores.append(clearance_score)

        return sum(scores)

    def select_best_grasp(
        self,
        candidates: list[GraspPose],
        collision_checker: Any | None = None,
    ) -> GraspPose | None:
        """Select the best grasp from candidates.

        Args:
            candidates: List of grasp pose candidates.
            collision_checker: Optional collision checking function.

        Returns:
            Best GraspPose or None if no valid grasps.
        """
        if not candidates:
            logger.warning("No grasp candidates provided")
            return None

        # Filter by collision if checker provided
        if collision_checker is not None:
            valid_candidates = [g for g in candidates if not collision_checker(g)]
            if not valid_candidates:
                logger.warning("All grasps failed collision check")
                return candidates[0]  # Return best anyway
            candidates = valid_candidates

        # Return highest scored grasp
        best_grasp = max(candidates, key=lambda g: g.score)
        logger.info(
            f"Selected {best_grasp.strategy.value} grasp with score {best_grasp.score:.3f}"
        )
        return best_grasp

    def visualize_grasps(
        self,
        point_cloud: o3d.geometry.PointCloud,
        grasps: list[GraspPose],
        selected_index: int | None = None,
    ) -> None:
        """Visualize grasp candidates on the object.

        Args:
            point_cloud: Object point cloud.
            grasps: List of grasp poses to visualize.
            selected_index: Index of selected grasp to highlight.
        """
        geometries: list[o3d.geometry.Geometry] = [point_cloud]

        for i, grasp in enumerate(grasps):
            # Create coordinate frame at grasp pose
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            frame.rotate(grasp.orientation, center=np.zeros(3))
            frame.translate(grasp.position)

            # Color based on selection
            if selected_index is not None and i == selected_index:
                frame.paint_uniform_color([0, 1, 0])  # Green for selected
            else:
                frame.paint_uniform_color([1, 0.5, 0])  # Orange for others

            geometries.append(frame)

            # Add approach line
            line_points = [
                grasp.position,
                grasp.position + grasp.approach_vector * 0.05,
            ]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector(line_points)
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            geometries.append(line)

        # Add world coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(world_frame)

        o3d.visualization.draw_geometries(geometries)
