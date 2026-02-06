# Dockerfile for DishBot Isaac Sim Environment
#
# This Dockerfile creates a containerized environment for running
# NVIDIA Isaac Sim with the DishBot robotic dishwashing system.
#
# Requirements:
#   - NVIDIA GPU with driver 525.60+ on the host
#   - NVIDIA Container Toolkit installed
#   - Docker 19.03+ with GPU support
#
# Build:
#   docker build -t dishbot:latest .
#
# Run (headless):
#   docker run --gpus all -v $(pwd):/workspace dishbot:latest
#
# Run (with GUI via X11):
#   docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix dishbot:latest

# Use NVIDIA Isaac Sim base image
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
ARG ISAAC_SIM_VERSION=4.2.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Isaac Sim environment setup
ENV ISAAC_SIM_PATH=/isaac-sim
ENV CARB_APP_PATH=${ISAAC_SIM_PATH}/kit
ENV EXP_PATH=${ISAAC_SIM_PATH}/apps

# Add Isaac Sim Python to path
ENV PATH="${ISAAC_SIM_PATH}/python.sh:${PATH}"

# Set up workspace
WORKDIR /workspace

# Install additional system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the application.
# On Windows and some base images (e.g. NVIDIA Isaac Sim), GID 1000 may already
# exist; use the existing group instead of creating a new one to avoid "group already exists".
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN set -eux; \
    if getent group ${GROUP_ID} >/dev/null 2>&1; then \
      EXISTING_GROUP=$(getent group ${GROUP_ID} | cut -d: -f1); \
      (id -u dishbot >/dev/null 2>&1 && echo "User dishbot exists") || useradd --uid ${USER_ID} --gid ${GROUP_ID} --no-create-home dishbot; \
      mkdir -p /home/dishbot && chown ${USER_ID}:${GROUP_ID} /home/dishbot; \
      chown -R ${USER_ID}:${GROUP_ID} /workspace; \
    else \
      groupadd --gid ${GROUP_ID} dishbot || true; \
      useradd --uid ${USER_ID} --gid ${GROUP_ID} -m dishbot || true; \
      chown -R ${USER_ID}:${GROUP_ID} /workspace; \
    fi

# Copy requirements first for better caching (use numeric UID:GID so it works when group name is not "dishbot")
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml README.md ./

# Install Python dependencies using Isaac Sim's Python
# Isaac Sim comes with its own Python environment
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir --upgrade pip \
    && ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir \
    # Core dependencies (torch is already in Isaac Sim)
    transformers>=4.37.0 \
    qwen-vl-utils>=0.0.8 \
    accelerate>=0.25.0 \
    open3d>=0.18.0 \
    trimesh>=4.0.0 \
    scipy>=1.11.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    tqdm>=4.66.0 \
    rich>=13.0.0 \
    loguru>=0.7.0 \
    h5py>=3.10.0 \
    pandas>=2.1.0

# Copy the rest of the application
COPY --chown=${USER_ID}:${GROUP_ID} . .

# Install DishBot package in development mode
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install --no-cache-dir -e .

# Copy and set up the entrypoint script
COPY --chown=${USER_ID}:${GROUP_ID} docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER dishbot

# Set default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]

# Expose ports for potential remote access/visualization
# 8211: Isaac Sim Livestream (WebRTC)
# 8011: Isaac Sim Livestream (WebSocket)
# 47995-47999: Omniverse streaming
EXPOSE 8211 8011 47995 47996 47997 47998 47999

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ${ISAAC_SIM_PATH}/python.sh -c "import omni.isaac.core; print('healthy')" || exit 1

# Labels for metadata
LABEL maintainer="DishBot Team"
LABEL description="DishBot robotic dishwashing system with NVIDIA Isaac Sim"
LABEL version="0.1.0"
