"""
Tests for the grasp planning module.
"""

import numpy as np
import open3d as o3d
import pytest

from dishbot.config import GraspConfig
from dishbot.grasp_planning import GraspPlanner, GraspPose, GraspStrategy


class TestGraspPlanner:
    """Tests for GraspPlanner class."""

    @pytest.fixture
    def grasp_planner(self) -> GraspPlanner:
        """Create a grasp planner instance for testing.

        Returns:
            GraspPlanner instance.
        """
        return GraspPlanner(config=GraspConfig())

    @pytest.fixture
    def sample_point_cloud(self) -> o3d.geometry.PointCloud:
        """Create a sample point cloud for testing.

        Returns:
            Sample point cloud.
        """
        # Create a flat disk shape (like a plate)
        num_points = 500
        radius = 0.12
        height = 0.02

        # Generate points on a disk
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.sqrt(np.random.uniform(0, radius**2, num_points))

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(0, height, num_points)

        points = np.column_stack([x, y, z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def test_initialization(self, grasp_planner: GraspPlanner) -> None:
        """Test grasp planner initialization.

        Args:
            grasp_planner: Grasp planner fixture.
        """
        assert grasp_planner is not None
        assert grasp_planner.config.gripper_width == 0.08

    def test_compute_grasp_candidates(
        self,
        grasp_planner: GraspPlanner,
        sample_point_cloud: o3d.geometry.PointCloud,
    ) -> None:
        """Test grasp candidate generation.

        Args:
            grasp_planner: Grasp planner fixture.
            sample_point_cloud: Sample point cloud fixture.
        """
        candidates = grasp_planner.compute_grasp_candidates(sample_point_cloud)

        assert len(candidates) > 0
        for grasp in candidates:
            assert isinstance(grasp, GraspPose)
            assert grasp.score >= 0
            assert grasp.gripper_width > 0

    def test_select_best_grasp(
        self,
        grasp_planner: GraspPlanner,
        sample_point_cloud: o3d.geometry.PointCloud,
    ) -> None:
        """Test best grasp selection.

        Args:
            grasp_planner: Grasp planner fixture.
            sample_point_cloud: Sample point cloud fixture.
        """
        candidates = grasp_planner.compute_grasp_candidates(sample_point_cloud)
        best_grasp = grasp_planner.select_best_grasp(candidates)

        assert best_grasp is not None
        assert best_grasp.score == max(g.score for g in candidates)

    def test_select_best_grasp_empty_list(self, grasp_planner: GraspPlanner) -> None:
        """Test handling of empty candidate list.

        Args:
            grasp_planner: Grasp planner fixture.
        """
        best_grasp = grasp_planner.select_best_grasp([])
        assert best_grasp is None


class TestGraspPose:
    """Tests for GraspPose dataclass."""

    def test_to_transformation_matrix(self) -> None:
        """Test conversion to transformation matrix."""
        grasp = GraspPose(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.eye(3),
            approach_vector=np.array([0, 0, -1]),
            gripper_width=0.08,
            strategy=GraspStrategy.TOP_DOWN,
        )

        T = grasp.to_transformation_matrix()

        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], grasp.position)
        assert np.allclose(T[:3, :3], grasp.orientation)
        assert T[3, 3] == 1.0

    def test_to_quaternion(self) -> None:
        """Test conversion to quaternion."""
        grasp = GraspPose(
            position=np.array([0, 0, 0]),
            orientation=np.eye(3),
            approach_vector=np.array([0, 0, -1]),
            gripper_width=0.08,
            strategy=GraspStrategy.TOP_DOWN,
        )

        quat = grasp.to_quaternion()

        assert quat.shape == (4,)
        # Identity rotation should give [0, 0, 0, 1] quaternion
        assert np.allclose(quat, [0, 0, 0, 1], atol=1e-6)
