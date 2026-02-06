"""
Training pipeline for DishBot.

This module provides training infrastructure for the grasp success predictor
and other learning components using data from the Isaac Sim environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from dishbot.config import TrainingConfig
from dishbot.grasp_planning import GraspPose, GraspStrategy
from dishbot.isaac_sim_env import IsaacSimDishwashingEnv


@dataclass
class GraspSample:
    """A single grasp training sample."""

    # Grasp pose features
    grasp_position: np.ndarray  # [3]
    grasp_orientation: np.ndarray  # [9] flattened rotation matrix
    approach_vector: np.ndarray  # [3]
    gripper_width: float

    # Object features
    object_center: np.ndarray  # [3]
    object_extent: np.ndarray  # [3]
    object_type_id: int

    # Outcome
    success: bool
    lift_height: float = 0.0


class GraspDataset(Dataset):
    """PyTorch Dataset for grasp training data."""

    def __init__(self, samples: list[GraspSample]) -> None:
        """Initialize the dataset.

        Args:
            samples: List of GraspSample objects.
        """
        self.samples = samples

        # Precompute tensors
        self.features = self._extract_features()
        self.labels = torch.tensor(
            [float(s.success) for s in samples],
            dtype=torch.float32,
        )

    def _extract_features(self) -> torch.Tensor:
        """Extract feature vectors from samples.

        Returns:
            Tensor of shape [N, feature_dim].
        """
        features_list = []

        for sample in self.samples:
            # Concatenate all features
            feature = np.concatenate([
                sample.grasp_position,
                sample.grasp_orientation,
                sample.approach_vector,
                [sample.gripper_width],
                sample.object_center,
                sample.object_extent,
                [sample.object_type_id / 10.0],  # Normalize
            ])
            features_list.append(feature)

        return torch.tensor(np.array(features_list), dtype=torch.float32)

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, label) tensors.
        """
        return self.features[idx], self.labels[idx]

    @property
    def feature_dim(self) -> int:
        """Get feature dimension.

        Returns:
            Number of features per sample.
        """
        return self.features.shape[1]

    def save(self, path: str | Path) -> None:
        """Save dataset to HDF5 file.

        Args:
            path: Path to save file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=self.features.numpy())
            f.create_dataset("labels", data=self.labels.numpy())

            # Store metadata
            f.attrs["num_samples"] = len(self.samples)
            f.attrs["feature_dim"] = self.feature_dim

        logger.info(f"Saved dataset with {len(self.samples)} samples to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "GraspDataset":
        """Load dataset from HDF5 file.

        Args:
            path: Path to saved file.

        Returns:
            Loaded GraspDataset.
        """
        with h5py.File(path, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
            labels = torch.tensor(f["labels"][:], dtype=torch.float32)

        # Create empty samples (we only need tensors)
        dataset = cls.__new__(cls)
        dataset.samples = []
        dataset.features = features
        dataset.labels = labels

        logger.info(f"Loaded dataset with {len(labels)} samples from {path}")
        return dataset


class GraspSuccessPredictor(nn.Module):
    """Neural network for predicting grasp success."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the predictor network.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()

        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch, input_dim].

        Returns:
            Success probability [batch, 1].
        """
        return self.network(x)


class GraspTrainingPipeline:
    """Complete training pipeline for grasp success prediction.

    This class provides:
    - Synthetic data generation using Isaac Sim
    - Training loop with validation
    - Model checkpointing
    - Evaluation and metrics
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        isaac_sim_env: IsaacSimDishwashingEnv | None = None,
    ) -> None:
        """Initialize the training pipeline.

        Args:
            config: Training configuration. Uses defaults if None.
            isaac_sim_env: Isaac Sim environment for data generation.
        """
        self.config = config or TrainingConfig()
        self.isaac_sim_env = isaac_sim_env

        self.model: GraspSuccessPredictor | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        logger.info(f"Training pipeline initialized, device: {self.device}")

    def generate_training_data(
        self,
        num_samples: int | None = None,
        save_path: str | Path | None = None,
    ) -> GraspDataset:
        """Generate synthetic training data using simulation.

        Args:
            num_samples: Number of samples to generate. Uses config if None.
            save_path: Path to save dataset. Uses config if None.

        Returns:
            Generated GraspDataset.
        """
        num_samples = num_samples or self.config.num_training_samples
        save_path = save_path or self.config.data_dir / "grasp_dataset.h5"

        logger.info(f"Generating {num_samples} training samples...")

        if self.isaac_sim_env is None:
            logger.warning("No simulation environment, generating mock data")
            return self._generate_mock_data(num_samples, save_path)

        samples: list[GraspSample] = []

        # Initialize simulation
        self.isaac_sim_env.initialize()

        progress = tqdm(total=num_samples, desc="Generating samples")

        while len(samples) < num_samples:
            # Reset environment
            obs = self.isaac_sim_env.reset()

            # Get dish states
            dish_states = self.isaac_sim_env.get_dish_states()

            for dish_state in dish_states:
                if len(samples) >= num_samples:
                    break

                # Generate random grasp for this dish
                sample = self._attempt_grasp_and_record(dish_state)
                if sample is not None:
                    samples.append(sample)
                    progress.update(1)

        progress.close()

        # Create dataset
        dataset = GraspDataset(samples)

        # Save dataset
        dataset.save(save_path)

        return dataset

    def _generate_mock_data(
        self,
        num_samples: int,
        save_path: str | Path,
    ) -> GraspDataset:
        """Generate mock training data for development.

        Args:
            num_samples: Number of samples.
            save_path: Path to save dataset.

        Returns:
            Mock GraspDataset.
        """
        logger.info("Generating mock training data...")

        samples: list[GraspSample] = []
        dish_types = {"plate": 0, "bowl": 1, "cup": 2, "mug": 3, "utensil": 4}

        for _ in tqdm(range(num_samples), desc="Generating mock samples"):
            # Random object
            dish_type = np.random.choice(list(dish_types.keys()))
            object_center = np.random.uniform(-0.3, 0.3, 3)
            object_extent = np.random.uniform(0.05, 0.2, 3)

            # Random grasp
            grasp_position = object_center + np.random.uniform(-0.1, 0.1, 3)
            grasp_orientation = np.eye(3).flatten()
            approach_vector = np.array([0, 0, -1])
            gripper_width = np.random.uniform(0.02, 0.08)

            # Simulate success (higher for aligned grasps)
            alignment = np.abs(approach_vector[2])
            success_prob = 0.3 + 0.5 * alignment
            success = np.random.random() < success_prob

            sample = GraspSample(
                grasp_position=grasp_position,
                grasp_orientation=grasp_orientation,
                approach_vector=approach_vector,
                gripper_width=gripper_width,
                object_center=object_center,
                object_extent=object_extent,
                object_type_id=dish_types[dish_type],
                success=success,
                lift_height=0.1 if success else 0.0,
            )
            samples.append(sample)

        dataset = GraspDataset(samples)
        dataset.save(save_path)

        return dataset

    def _attempt_grasp_and_record(self, dish_state: Any) -> GraspSample | None:
        """Attempt a grasp and record the outcome.

        Args:
            dish_state: State of the dish to grasp.

        Returns:
            GraspSample or None if attempt invalid.
        """
        # Generate random grasp parameters
        grasp_position = dish_state.position + np.random.uniform(-0.05, 0.05, 3)
        grasp_orientation = np.eye(3)  # Default orientation
        approach_vector = np.array([0, 0, -1])  # Top-down
        gripper_width = np.random.uniform(0.02, 0.08)

        # Get pre-grasp dish height
        initial_z = dish_state.position[2]

        # Execute grasp in simulation
        # (Simplified - actual implementation would use robot controller)
        success = np.random.random() > 0.5  # Mock success

        dish_types = {"plate": 0, "bowl": 1, "cup": 2, "mug": 3, "utensil": 4}
        type_id = dish_types.get(dish_state.dish_type, 0)

        return GraspSample(
            grasp_position=grasp_position,
            grasp_orientation=grasp_orientation.flatten(),
            approach_vector=approach_vector,
            gripper_width=gripper_width,
            object_center=dish_state.position,
            object_extent=np.array([0.1, 0.1, 0.05]),  # Estimated
            object_type_id=type_id,
            success=success,
            lift_height=0.1 if success else 0.0,
        )

    def train(
        self,
        train_dataset: GraspDataset | None = None,
        num_epochs: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train the grasp success predictor.

        Args:
            train_dataset: Training dataset. Loads from config path if None.
            num_epochs: Number of epochs. Uses config if None.
            resume_from: Path to checkpoint to resume from.

        Returns:
            Dictionary with training metrics.
        """
        num_epochs = num_epochs or self.config.num_epochs

        # Load or use provided dataset
        if train_dataset is None:
            dataset_path = self.config.data_dir / "grasp_dataset.h5"
            if not dataset_path.exists():
                logger.info("No dataset found, generating...")
                train_dataset = self.generate_training_data()
            else:
                train_dataset = GraspDataset.load(dataset_path)

        # Split dataset
        val_size = int(len(train_dataset) * self.config.validation_split)
        train_size = len(train_dataset) - val_size

        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Initialize model
        self.model = GraspSuccessPredictor(
            input_dim=train_dataset.feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._load_checkpoint(resume_from)

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(start_epoch, num_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_accuracy = self._validate(val_loader)
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch)

        return {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            loader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(features).squeeze()
            loss = F.binary_cross_entropy(predictions, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        """Validate the model.

        Args:
            loader: Validation data loader.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(features).squeeze()
                loss = F.binary_cross_entropy(predictions, labels)

                total_loss += loss.item()

                # Accuracy
                predicted_classes = (predictions > 0.5).float()
                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(loader), correct / total

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

        # Save regular checkpoint
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = self.config.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val loss: {self.best_val_loss:.4f}")

    def _load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        return checkpoint["epoch"] + 1

    def evaluate(
        self,
        test_dataset: GraspDataset,
    ) -> dict[str, float]:
        """Evaluate the model on a test set.

        Args:
            test_dataset: Test dataset.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        self.model.eval()
        all_predictions: list[float] = []
        all_labels: list[float] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                predictions = self.model(features).squeeze()

                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        # Compute metrics
        predictions_array = np.array(all_predictions)
        labels_array = np.array(all_labels)
        predicted_classes = (predictions_array > 0.5).astype(float)

        accuracy = np.mean(predicted_classes == labels_array)
        precision = np.sum((predicted_classes == 1) & (labels_array == 1)) / (
            np.sum(predicted_classes == 1) + 1e-6
        )
        recall = np.sum((predicted_classes == 1) & (labels_array == 1)) / (
            np.sum(labels_array == 1) + 1e-6
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def predict(
        self,
        grasp_pose: GraspPose,
        object_center: np.ndarray,
        object_extent: np.ndarray,
        object_type_id: int,
    ) -> float:
        """Predict grasp success probability.

        Args:
            grasp_pose: The grasp pose to evaluate.
            object_center: Center of the target object.
            object_extent: Extent of the object bounding box.
            object_type_id: ID of the object type.

        Returns:
            Success probability between 0 and 1.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Create feature vector
        features = np.concatenate([
            grasp_pose.position,
            grasp_pose.orientation.flatten(),
            grasp_pose.approach_vector,
            [grasp_pose.gripper_width],
            object_center,
            object_extent,
            [object_type_id / 10.0],
        ])

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor).item()

        return prediction

    def load_model(self, path: str | Path) -> None:
        """Load a trained model from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Infer input dimension from saved model
        first_layer_weight = checkpoint["model_state_dict"]["network.0.weight"]
        input_dim = first_layer_weight.shape[1]

        # Initialize model
        self.model = GraspSuccessPredictor(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {path}")
