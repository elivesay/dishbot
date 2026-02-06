"""
Main entry point for DishBot.

This module provides the command-line interface and orchestrates the
complete dishwashing robot pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from dishbot.config import DishBotConfig
from dishbot.grasp_planning import GraspPlanner
from dishbot.isaac_sim_env import MockIsaacSimEnv, create_environment
from dishbot.robot_controller import DishwashingRobotController
from dishbot.training_pipeline import GraspDataset, GraspTrainingPipeline
from dishbot.vision_module import DishVisionSystem

console = Console()


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure logging with loguru.

    Args:
        level: Logging level.
        log_file: Optional path to log file.
    """
    logger.remove()  # Remove default handler

    # Console logging with colors
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # File logging if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
        )


def run_demo(config: DishBotConfig, use_mock: bool = True) -> None:
    """Run a demonstration of the complete pipeline.

    Args:
        config: DishBot configuration.
        use_mock: Whether to use mock simulation.
    """
    console.print(Panel.fit("[bold blue]DishBot Demo[/bold blue]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize environment
        task = progress.add_task("Initializing environment...", total=None)
        env = create_environment(use_mock=use_mock, sim_config=config.simulation)
        env.initialize()
        progress.update(task, completed=True, description="Environment ready")

        # Initialize vision system
        task = progress.add_task("Initializing vision system...", total=None)
        vision = DishVisionSystem(
            vision_config=config.vision,
            camera_config=config.camera,
        )
        progress.update(task, completed=True, description="Vision system ready")

        # Initialize grasp planner
        task = progress.add_task("Initializing grasp planner...", total=None)
        grasp_planner = GraspPlanner(config=config.grasp)
        progress.update(task, completed=True, description="Grasp planner ready")

        # Initialize robot controller
        task = progress.add_task("Initializing robot controller...", total=None)
        robot = DishwashingRobotController(
            config=config.robot,
            simulation_env=env,
        )
        robot.initialize_robot()
        progress.update(task, completed=True, description="Robot controller ready")

    console.print("\n[green]All systems initialized![/green]\n")

    # Reset environment and spawn dishes
    console.print("Resetting environment and spawning dishes...")
    obs = env.reset()

    console.print(f"RGB image shape: {obs.rgb_image.shape}")
    console.print(f"Depth image shape: {obs.depth_image.shape}")

    # Process scene with vision
    console.print("\nProcessing scene with vision system...")

    # Reconstruct 3D geometry
    point_cloud = vision.reconstruct_3d_geometry(
        obs.rgb_image,
        obs.depth_image,
    )
    console.print(f"Created point cloud with {len(point_cloud.points)} points")

    # Segment dishes
    dish_point_clouds = vision.segment_individual_dishes(point_cloud)
    console.print(f"Segmented {len(dish_point_clouds)} dishes")

    # Plan grasps for each dish
    console.print("\nPlanning grasps...")
    for i, dish_pcd in enumerate(dish_point_clouds):
        grasps = grasp_planner.compute_grasp_candidates(dish_pcd)
        if grasps:
            best_grasp = grasp_planner.select_best_grasp(grasps)
            console.print(
                f"  Dish {i + 1}: {best_grasp.strategy.value} grasp, "
                f"score={best_grasp.score:.3f}"
            )

    # Clean up
    env.close()
    console.print("\n[green]Demo completed![/green]")


def run_training(config: DishBotConfig, args: argparse.Namespace) -> None:
    """Run the training pipeline.

    Args:
        config: DishBot configuration.
        args: Command line arguments.
    """
    console.print(Panel.fit("[bold blue]DishBot Training[/bold blue]"))

    # Initialize environment if needed for data generation
    env = None
    if args.generate_data:
        console.print("Initializing simulation for data generation...")
        env = create_environment(use_mock=True, sim_config=config.simulation)
        env.initialize()

    # Initialize training pipeline
    pipeline = GraspTrainingPipeline(
        config=config.training,
        isaac_sim_env=env,
    )

    # Generate data if requested
    if args.generate_data:
        console.print(f"Generating {args.num_samples} training samples...")
        pipeline.generate_training_data(num_samples=args.num_samples)

    # Train model
    console.print(f"Training for {args.num_epochs} epochs...")
    metrics = pipeline.train(num_epochs=args.num_epochs)

    console.print("\n[green]Training completed![/green]")
    console.print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
    console.print(f"  Final val loss: {metrics['final_val_loss']:.4f}")
    console.print(f"  Best val loss: {metrics['best_val_loss']:.4f}")

    if env is not None:
        env.close()


def run_evaluation(config: DishBotConfig, args: argparse.Namespace) -> None:
    """Run model evaluation.

    Args:
        config: DishBot configuration.
        args: Command line arguments.
    """
    console.print(Panel.fit("[bold blue]DishBot Evaluation[/bold blue]"))

    # Load model
    pipeline = GraspTrainingPipeline(config=config.training)

    checkpoint_path = args.checkpoint or config.training.checkpoint_dir / "checkpoint_best.pt"
    console.print(f"Loading model from {checkpoint_path}...")
    pipeline.load_model(checkpoint_path)

    # Load test dataset
    dataset_path = args.dataset or config.training.data_dir / "grasp_dataset.h5"
    console.print(f"Loading dataset from {dataset_path}...")
    dataset = GraspDataset.load(dataset_path)

    # Evaluate
    console.print("Evaluating model...")
    metrics = pipeline.evaluate(dataset)

    console.print("\n[green]Evaluation Results:[/green]")
    console.print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    console.print(f"  Precision: {metrics['precision']:.2%}")
    console.print(f"  Recall:    {metrics['recall']:.2%}")
    console.print(f"  F1 Score:  {metrics['f1']:.2%}")


def run_full_pipeline(config: DishBotConfig, use_mock: bool = True) -> None:
    """Run the complete dishwashing pipeline.

    Args:
        config: DishBot configuration.
        use_mock: Whether to use mock simulation.
    """
    console.print(Panel.fit("[bold blue]DishBot Full Pipeline[/bold blue]"))

    # Initialize all components
    console.print("Initializing components...")

    env = create_environment(use_mock=use_mock, sim_config=config.simulation)
    env.initialize()

    vision = DishVisionSystem(
        vision_config=config.vision,
        camera_config=config.camera,
    )

    grasp_planner = GraspPlanner(config=config.grasp)

    robot = DishwashingRobotController(
        config=config.robot,
        simulation_env=env,
    )
    robot.initialize_robot()

    # Load trained model if available
    training_pipeline = GraspTrainingPipeline(config=config.training)
    checkpoint_path = config.training.checkpoint_dir / "checkpoint_best.pt"
    if checkpoint_path.exists():
        console.print("Loading trained grasp predictor...")
        training_pipeline.load_model(checkpoint_path)
        has_predictor = True
    else:
        console.print("[yellow]No trained model found, using heuristic scoring[/yellow]")
        has_predictor = False

    # Main dishwashing loop
    console.print("\nStarting dishwashing pipeline...")

    # Reset and get initial observation
    obs = env.reset()
    dish_states = env.get_dish_states()

    console.print(f"Found {len(dish_states)} dishes to wash")

    # Process each dish
    dishes_washed = 0
    place_position = np.array([0.8, 0.0, 0.9])  # Drying rack position

    for dish_state in dish_states:
        console.print(f"\nProcessing dish: {dish_state.dish_id} ({dish_state.dish_type})")

        # Get camera observation
        obs = env.get_camera_observation()

        # Reconstruct 3D and segment
        point_cloud = vision.reconstruct_3d_geometry(obs.rgb_image, obs.depth_image)
        dish_pcds = vision.segment_individual_dishes(point_cloud)

        if not dish_pcds:
            console.print("  [yellow]No dishes detected[/yellow]")
            continue

        # Use first detected dish (simplified)
        dish_pcd = dish_pcds[0]

        # Plan grasps
        grasp_candidates = grasp_planner.compute_grasp_candidates(dish_pcd)

        if not grasp_candidates:
            console.print("  [yellow]No valid grasps found[/yellow]")
            continue

        # Select best grasp (using trained predictor if available)
        if has_predictor:
            pose_info = vision.estimate_dish_pose(dish_pcd)
            for grasp in grasp_candidates:
                predicted_score = training_pipeline.predict(
                    grasp,
                    pose_info["center"],
                    pose_info["extent"],
                    0,  # Type ID
                )
                grasp.score = predicted_score

            grasp_candidates.sort(key=lambda g: g.score, reverse=True)

        best_grasp = grasp_candidates[0]
        console.print(f"  Selected {best_grasp.strategy.value} grasp (score: {best_grasp.score:.3f})")

        # Execute pick and place
        success = robot.pick_and_place(best_grasp, place_position)

        if success:
            dishes_washed += 1
            console.print("  [green]Successfully washed![/green]")

            # Move place position slightly for next dish
            place_position[1] += 0.15
        else:
            console.print("  [red]Failed to grasp/place[/red]")

        # Step simulation
        for _ in range(10):
            env.step()

    # Summary
    console.print(f"\n[bold green]Completed! Washed {dishes_washed}/{len(dish_states)} dishes[/bold green]")

    env.close()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="DishBot: Robotic Dishwashing with Vision and 3D Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration")
    demo_parser.add_argument(
        "--real-sim",
        action="store_true",
        help="Use real Isaac Sim instead of mock",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the grasp predictor")
    train_parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate training data before training",
    )
    train_parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to evaluation dataset",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full dishwashing pipeline")
    run_parser.add_argument(
        "--real-sim",
        action="store_true",
        help="Use real Isaac Sim instead of mock",
    )

    return parser


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Load configuration
    if args.config and args.config.exists():
        config = DishBotConfig.from_yaml(args.config)
    else:
        config = DishBotConfig()

    # Print banner
    console.print(
        Panel.fit(
            "[bold cyan]DishBot[/bold cyan]\n"
            "Robotic Dishwashing with Vision and 3D Reconstruction",
            border_style="cyan",
        )
    )

    try:
        if args.command == "demo":
            run_demo(config, use_mock=not args.real_sim)
        elif args.command == "train":
            run_training(config, args)
        elif args.command == "evaluate":
            run_evaluation(config, args)
        elif args.command == "run":
            run_full_pipeline(config, use_mock=not args.real_sim)
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Unhandled exception")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
