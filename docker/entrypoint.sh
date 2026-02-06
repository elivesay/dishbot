#!/bin/bash
# Entrypoint script for DishBot Isaac Sim container
#
# This script handles initialization and provides multiple run modes:
#   - headless: Run Isaac Sim in headless mode (default)
#   - gui: Run with GUI (requires X11 forwarding)
#   - livestream: Run with WebRTC livestreaming for remote viewing
#   - shell: Start an interactive shell
#   - python: Run a Python script with Isaac Sim's Python
#   - dishbot: Run DishBot commands

set -e

# Source Isaac Sim environment
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
export CARB_APP_PATH="${ISAAC_SIM_PATH}/kit"
export LD_LIBRARY_PATH="${ISAAC_SIM_PATH}/exts/omni.isaac.core_nodes/bin:${LD_LIBRARY_PATH}"

# Function to print usage
print_usage() {
    echo "DishBot Isaac Sim Container"
    echo ""
    echo "Usage: docker run [docker-options] dishbot:latest [command] [args...]"
    echo ""
    echo "Commands:"
    echo "  --help              Show this help message"
    echo "  shell               Start an interactive bash shell"
    echo "  python <script>     Run a Python script with Isaac Sim's Python"
    echo "  dishbot <args>      Run DishBot CLI commands"
    echo "  demo                Run DishBot demo with real Isaac Sim"
    echo "  train               Run training pipeline"
    echo "  headless            Start Isaac Sim in headless mode"
    echo "  livestream          Start Isaac Sim with WebRTC livestream"
    echo "  <custom>            Run any custom command"
    echo ""
    echo "Examples:"
    echo "  # Run demo with Isaac Sim"
    echo "  docker run --gpus all dishbot:latest demo"
    echo ""
    echo "  # Generate training data"
    echo "  docker run --gpus all -v \$(pwd)/data:/workspace/data dishbot:latest train --generate-data"
    echo ""
    echo "  # Interactive shell"
    echo "  docker run --gpus all -it dishbot:latest shell"
    echo ""
    echo "  # Run with livestream (access at http://localhost:8211)"
    echo "  docker run --gpus all -p 8211:8211 dishbot:latest livestream"
    echo ""
    echo "Environment Variables:"
    echo "  DISHBOT_HEADLESS=1       Force headless mode"
    echo "  DISHBOT_LIVESTREAM=1     Enable WebRTC livestream"
    echo "  DISHBOT_CONFIG=path      Path to config file"
}

# Function to start Isaac Sim in headless mode
start_headless() {
    echo "Starting Isaac Sim in headless mode..."
    exec "${ISAAC_SIM_PATH}/python.sh" "$@"
}

# Function to start with livestream
start_livestream() {
    echo "Starting Isaac Sim with WebRTC livestream..."
    echo "Access the stream at: http://localhost:8211"
    export LIVESTREAM=1
    exec "${ISAAC_SIM_PATH}/python.sh" \
        --enable omni.kit.livestream.webrtc \
        "$@"
}

# Function to run DishBot commands
run_dishbot() {
    echo "Running DishBot command: $*"
    exec "${ISAAC_SIM_PATH}/python.sh" -m dishbot.main "$@"
}

# Wait for GPU to be available (useful in some orchestration scenarios)
wait_for_gpu() {
    local max_attempts=30
    local attempt=0
    
    echo "Checking for GPU availability..."
    while [ $attempt -lt $max_attempts ]; do
        if nvidia-smi > /dev/null 2>&1; then
            echo "GPU detected!"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
            return 0
        fi
        echo "Waiting for GPU... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt+1))
    done
    
    echo "WARNING: No GPU detected. Isaac Sim requires an NVIDIA GPU."
    return 1
}

# Main entrypoint logic
main() {
    # Check for GPU (warn but don't fail - might be testing)
    wait_for_gpu || echo "Continuing without GPU check..."
    
    # Parse command
    case "${1:-}" in
        --help|-h|"")
            print_usage
            exit 0
            ;;
        shell|bash)
            echo "Starting interactive shell..."
            exec /bin/bash
            ;;
        python)
            shift
            echo "Running Python script: $*"
            exec "${ISAAC_SIM_PATH}/python.sh" "$@"
            ;;
        dishbot)
            shift
            run_dishbot "$@"
            ;;
        demo)
            shift
            run_dishbot demo --real-sim "$@"
            ;;
        train)
            shift
            run_dishbot train "$@"
            ;;
        evaluate)
            shift
            run_dishbot evaluate "$@"
            ;;
        headless)
            shift
            start_headless "$@"
            ;;
        livestream)
            shift
            start_livestream "$@"
            ;;
        *)
            # Run custom command
            exec "$@"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
