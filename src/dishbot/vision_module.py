"""
Vision module for DishBot.

This module provides vision-based dish detection and 3D reconstruction
using Qwen2-VL for semantic understanding and Open3D for geometric processing.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from dishbot.config import CameraConfig, VisionConfig


@dataclass
class DishDetection:
    """Represents a detected dish with its properties."""

    dish_id: int
    dish_type: str
    position: np.ndarray
    bounding_box: dict[str, float]
    point_cloud: o3d.geometry.PointCloud | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DishVisionSystem:
    """Vision system combining Qwen2-VL for semantic understanding with 3D reconstruction.

    This class provides:
    - Semantic dish detection using Qwen2-VL vision-language model
    - RGBD to point cloud conversion
    - Individual dish segmentation using clustering
    - Dish type classification and pose estimation
    """

    def __init__(
        self,
        vision_config: VisionConfig | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        """Initialize the DishVisionSystem.

        Args:
            vision_config: Vision system configuration. Uses defaults if None.
            camera_config: Camera configuration. Uses defaults if None.
        """
        self.vision_config = vision_config or VisionConfig()
        self.camera_config = camera_config or CameraConfig()

        self.processor: AutoProcessor | None = None
        self.model: Qwen2VLForConditionalGeneration | None = None
        self._model_loaded = False

        logger.info(
            f"Initialized DishVisionSystem with model: {self.vision_config.model_name}"
        )

    def load_model(self) -> None:
        """Load the Qwen2-VL model and processor.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._model_loaded:
            logger.warning("Model already loaded, skipping reload")
            return

        logger.info(f"Loading model: {self.vision_config.model_name}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.vision_config.model_name)

            # Determine torch dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.vision_config.torch_dtype, torch.float16)

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.vision_config.model_name,
                torch_dtype=torch_dtype,
                device_map=self.vision_config.device,
            )

            self._model_loaded = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load vision model: {e}") from e

    def detect_dishes(
        self,
        rgb_image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> str:
        """Use Qwen2-VL for semantic dish detection.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3) or PIL Image.
            prompt: Custom prompt for detection. Uses default if None.

        Returns:
            Model response describing detected dishes.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image.astype(np.uint8))

        prompt = prompt or self.vision_config.dish_detection_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False),
            images=rgb_image,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.vision_config.max_new_tokens,
        )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Detection response: {response[:200]}...")

        return response

    def analyze_grasp_points(
        self,
        rgb_image: np.ndarray | Image.Image,
        dish_description: str | None = None,
    ) -> str:
        """Analyze potential grasp points using vision-language model.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3) or PIL Image.
            dish_description: Optional description of the dish to analyze.

        Returns:
            Model response with grasp point analysis.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image.astype(np.uint8))

        prompt = self.vision_config.grasp_analysis_prompt
        if dish_description:
            prompt = f"{prompt}\n\nDish description: {dish_description}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False),
            images=rgb_image,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.vision_config.max_new_tokens,
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def reconstruct_3d_geometry(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_intrinsics: dict[str, float] | None = None,
    ) -> o3d.geometry.PointCloud:
        """Convert RGBD images to point cloud for geometric reconstruction.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3).
            depth_image: Depth image as numpy array (H, W) in meters.
            camera_intrinsics: Camera intrinsic parameters. Uses defaults if None.

        Returns:
            Open3D PointCloud object.
        """
        if camera_intrinsics is None:
            camera_intrinsics = self.camera_config.to_intrinsic_dict()

        height, width = depth_image.shape

        # Generate pixel coordinates
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        # Unproject to 3D
        fx, fy = camera_intrinsics["fx"], camera_intrinsics["fy"]
        cx, cy = camera_intrinsics["cx"], camera_intrinsics["cy"]

        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0

        # Filter invalid points
        valid = (z.flatten() > self.camera_config.min_depth) & (
            z.flatten() < self.camera_config.max_depth
        )
        points = points[valid]
        colors = colors[valid]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Optional: downsample for efficiency
        if self.vision_config.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.vision_config.voxel_size)

        logger.debug(f"Created point cloud with {len(pcd.points)} points")
        return pcd

    def segment_individual_dishes(
        self,
        point_cloud: o3d.geometry.PointCloud,
        eps: float | None = None,
        min_points: int | None = None,
    ) -> list[o3d.geometry.PointCloud]:
        """Use clustering to segment individual dishes from point cloud.

        Args:
            point_cloud: Input point cloud containing multiple dishes.
            eps: DBSCAN epsilon parameter. Uses config default if None.
            min_points: DBSCAN minimum points parameter. Uses config default if None.

        Returns:
            List of point clouds, one per detected dish.
        """
        eps = eps or self.vision_config.dbscan_eps
        min_points = min_points or self.vision_config.dbscan_min_points

        # DBSCAN clustering for segmentation
        labels = np.array(
            point_cloud.cluster_dbscan(
                eps=eps,
                min_points=min_points,
            )
        )

        # Extract individual dish point clouds
        dishes: list[o3d.geometry.PointCloud] = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = labels == label
            indices = np.where(mask)[0]
            dish_pcd = point_cloud.select_by_index(indices.tolist())
            dishes.append(dish_pcd)

        logger.info(f"Segmented {len(dishes)} individual dishes")
        return dishes

    def estimate_dish_pose(
        self,
        dish_point_cloud: o3d.geometry.PointCloud,
    ) -> dict[str, Any]:
        """Estimate the pose and bounding box of a dish.

        Args:
            dish_point_cloud: Point cloud of a single dish.

        Returns:
            Dictionary containing pose information.
        """
        # Compute oriented bounding box
        obb = dish_point_cloud.get_oriented_bounding_box()

        # Compute centroid
        centroid = np.asarray(dish_point_cloud.points).mean(axis=0)

        # Estimate surface normal (dominant plane)
        dish_point_cloud.estimate_normals()
        normals = np.asarray(dish_point_cloud.normals)
        mean_normal = normals.mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal)

        return {
            "center": np.array(obb.center),
            "extent": np.array(obb.extent),
            "rotation": np.array(obb.R),
            "centroid": centroid,
            "surface_normal": mean_normal,
            "num_points": len(dish_point_cloud.points),
        }

    def classify_dish_type(
        self,
        pose_info: dict[str, Any],
    ) -> str:
        """Classify dish type based on geometric features.

        Args:
            pose_info: Pose information from estimate_dish_pose.

        Returns:
            Estimated dish type string.
        """
        extent = pose_info["extent"]

        # Sort extents to get consistent dimensions
        sorted_extent = np.sort(extent)
        height = sorted_extent[0]
        min_horizontal = sorted_extent[1]
        max_horizontal = sorted_extent[2]

        # Simple heuristic classification based on aspect ratios
        aspect_ratio = max_horizontal / (min_horizontal + 1e-6)
        height_ratio = height / (max_horizontal + 1e-6)

        if height_ratio < 0.1:  # Very flat
            return "plate"
        elif height_ratio < 0.3 and aspect_ratio < 1.3:  # Shallow and roughly circular
            return "bowl"
        elif height_ratio > 0.5 and aspect_ratio < 1.5:  # Tall and narrow
            return "cup"
        elif height_ratio > 0.3 and aspect_ratio > 2.0:  # Long and thin
            return "utensil"
        else:
            return "unknown"

    def process_scene(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> list[DishDetection]:
        """Process a complete scene to detect and analyze all dishes.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3).
            depth_image: Depth image as numpy array (H, W) in meters.

        Returns:
            List of DishDetection objects for each detected dish.
        """
        logger.info("Processing scene...")

        # Reconstruct 3D geometry
        point_cloud = self.reconstruct_3d_geometry(rgb_image, depth_image)

        # Segment individual dishes
        dish_point_clouds = self.segment_individual_dishes(point_cloud)

        # Process each dish
        detections: list[DishDetection] = []
        for i, dish_pcd in enumerate(dish_point_clouds):
            pose_info = self.estimate_dish_pose(dish_pcd)
            dish_type = self.classify_dish_type(pose_info)

            detection = DishDetection(
                dish_id=i,
                dish_type=dish_type,
                position=pose_info["center"],
                bounding_box={
                    "center": pose_info["center"].tolist(),
                    "extent": pose_info["extent"].tolist(),
                },
                point_cloud=dish_pcd,
                metadata={
                    "surface_normal": pose_info["surface_normal"].tolist(),
                    "num_points": pose_info["num_points"],
                },
            )
            detections.append(detection)

        logger.info(f"Detected {len(detections)} dishes in scene")
        return detections

    def visualize_detections(
        self,
        detections: list[DishDetection],
        show_bounding_boxes: bool = True,
    ) -> None:
        """Visualize detected dishes using Open3D.

        Args:
            detections: List of DishDetection objects.
            show_bounding_boxes: Whether to show oriented bounding boxes.
        """
        geometries: list[o3d.geometry.Geometry] = []

        # Color palette for different dishes
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]

        for i, detection in enumerate(detections):
            if detection.point_cloud is not None:
                # Color the point cloud
                color = colors[i % len(colors)]
                detection.point_cloud.paint_uniform_color(color)
                geometries.append(detection.point_cloud)

                if show_bounding_boxes:
                    obb = detection.point_cloud.get_oriented_bounding_box()
                    obb.color = color
                    geometries.append(obb)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord_frame)

        o3d.visualization.draw_geometries(geometries)
