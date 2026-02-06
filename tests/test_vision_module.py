"""
Tests for the vision module.
"""

import numpy as np
import pytest

from dishbot.config import CameraConfig, VisionConfig
from dishbot.vision_module import DishDetection, DishVisionSystem


class TestDishVisionSystem:
    """Tests for DishVisionSystem class."""

    @pytest.fixture
    def vision_system(self) -> DishVisionSystem:
        """Create a vision system instance for testing.

        Returns:
            DishVisionSystem instance.
        """
        return DishVisionSystem(
            vision_config=VisionConfig(),
            camera_config=CameraConfig(),
        )

    def test_initialization(self, vision_system: DishVisionSystem) -> None:
        """Test vision system initialization.

        Args:
            vision_system: Vision system fixture.
        """
        assert vision_system is not None
        assert vision_system.vision_config is not None
        assert vision_system.camera_config is not None
        assert not vision_system._model_loaded

    def test_reconstruct_3d_geometry(self, vision_system: DishVisionSystem) -> None:
        """Test 3D reconstruction from RGBD images.

        Args:
            vision_system: Vision system fixture.
        """
        # Create synthetic RGBD data
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.uniform(0.5, 1.5, (480, 640)).astype(np.float32)

        # Reconstruct point cloud
        pcd = vision_system.reconstruct_3d_geometry(rgb, depth)

        assert pcd is not None
        assert len(pcd.points) > 0
        assert len(pcd.colors) == len(pcd.points)

    def test_segment_individual_dishes(self, vision_system: DishVisionSystem) -> None:
        """Test dish segmentation from point cloud.

        Args:
            vision_system: Vision system fixture.
        """
        # Create synthetic point cloud with two clusters
        import open3d as o3d

        points1 = np.random.randn(200, 3) * 0.01 + np.array([0, 0, 0])
        points2 = np.random.randn(200, 3) * 0.01 + np.array([0.1, 0.1, 0])
        all_points = np.vstack([points1, points2])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)

        # Segment
        dishes = vision_system.segment_individual_dishes(pcd, eps=0.03, min_points=50)

        assert len(dishes) >= 1  # Should find at least one cluster

    def test_classify_dish_type(self, vision_system: DishVisionSystem) -> None:
        """Test dish type classification.

        Args:
            vision_system: Vision system fixture.
        """
        # Flat object (plate)
        pose_info_plate = {
            "center": np.array([0, 0, 0]),
            "extent": np.array([0.25, 0.25, 0.02]),
            "rotation": np.eye(3),
            "centroid": np.array([0, 0, 0]),
            "surface_normal": np.array([0, 0, 1]),
            "num_points": 100,
        }
        assert vision_system.classify_dish_type(pose_info_plate) == "plate"

        # Tall object (cup)
        pose_info_cup = {
            "center": np.array([0, 0, 0]),
            "extent": np.array([0.08, 0.08, 0.12]),
            "rotation": np.eye(3),
            "centroid": np.array([0, 0, 0]),
            "surface_normal": np.array([0, 0, 1]),
            "num_points": 100,
        }
        assert vision_system.classify_dish_type(pose_info_cup) == "cup"


class TestDishDetection:
    """Tests for DishDetection dataclass."""

    def test_creation(self) -> None:
        """Test DishDetection creation."""
        detection = DishDetection(
            dish_id=0,
            dish_type="plate",
            position=np.array([0.5, 0.0, 0.8]),
            bounding_box={"center": [0.5, 0.0, 0.8], "extent": [0.25, 0.25, 0.02]},
        )

        assert detection.dish_id == 0
        assert detection.dish_type == "plate"
        assert detection.confidence == 1.0
