# ──────────────────────────────────────────────────────────────────────────────
# HSMR – Reconstructing Humans with a Biomechanically Accurate Skeleton
# Base: CUDA 12.1 + Ubuntu 20.04 (matches torch==2.3.1+cu121)
# Python: 3.10
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ---------------------------------------------------------------------------- #
# System dependencies
# ---------------------------------------------------------------------------- #
# Ubuntu 22.04 ships Python 3.10 natively — no PPA needed.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libegl-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3.10 \
        python3.10-dev \
        python3-pip \
        wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ---------------------------------------------------------------------------- #
# Python dependencies (pinned versions from docs/requirements_py3.8.txt)
# torch is installed separately to pick the correct CUDA wheel.
# ---------------------------------------------------------------------------- #
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------- #
# Extra dependencies not in requirements.txt
# ---------------------------------------------------------------------------- #
# Build detectron2 CUDA extensions for all modern GPU architectures so the image
# works regardless of which GPU it runs on (Volta, Turing, Ampere, Ada, Hopper).
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
# detectron2 — disable build isolation so the build sees the already-installed torch
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git"

# chumpy (no PyPI wheel for new Python)
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/mattloper/chumpy"

# ---------------------------------------------------------------------------- #
# Copy HSMR source
# ---------------------------------------------------------------------------- #
WORKDIR /workspace/HSMR

COPY . .

# ---------------------------------------------------------------------------- #
# Install SKEL (git submodule – clone if empty, install in-place)
# ---------------------------------------------------------------------------- #
RUN if [ ! -f thirdparty/SKEL/setup.py ]; then \
        git clone --depth 1 https://github.com/MarilynKeller/SKEL.git thirdparty/SKEL; \
    fi \
 && pip install --no-cache-dir -e thirdparty/SKEL

# ---------------------------------------------------------------------------- #
# Install HSMR itself
# ---------------------------------------------------------------------------- #
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------- #
# Runtime environment
# ---------------------------------------------------------------------------- #
# Offscreen rendering via PyRender / EGL
ENV PYOPENGL_PLATFORM=egl

# Default working directory so relative paths (data_inputs/, etc.) resolve
WORKDIR /workspace/HSMR

CMD ["bash"]
