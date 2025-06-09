ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG PY_VERSION="312"
ARG DOCKER_FROM=vastai/base-image:cuda-$CUDA_VERSION-cudnn-devel-ubuntu$UBUNTU_VERSION-py$PY_VERSION

FROM $DOCKER_FROM AS base

WORKDIR /opt/workspace-internal/

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION=3.12
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV NUM_GPUS=1

# Activate virtual environment from vast.ai base image
RUN . /venv/main/bin/activate

# Install dependencies required for Miniconda
RUN apt-get update -y && \
    apt-get install -y wget bzip2 ca-certificates git curl && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    jq \
    libcurl4-openssl-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libssl-dev \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    openssh-server \
    openssh-client \
    git-lfs \
    vim \
    zip \
    unzip \
    zlib1g-dev \
    libc6-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Create environment with Python 3.12 and MPI
RUN $CONDA_DIR/bin/conda create -n pyenv python=3.12 -y && \
    $CONDA_DIR/bin/conda install -n pyenv -c conda-forge openmpi mpi4py -y 


# Define PyTorch versions via arguments
ARG PYTORCH="2.6.0"
ARG CUDA="124"

# Install PyTorch with specified version and CUDA
RUN $CONDA_DIR/bin/conda run -n pyenv \
    pip install torch==$PYTORCH torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA

RUN $CONDA_DIR/bin/conda install -n pyenv nvidia/label/cuda-$CUDA_VERSION::cuda-nvcc

# Debug
# RUN $CONDA_DIR/bin/conda run -n pyenv \
#     pip install debugpy

# EXPOSE 5678

RUN echo "Copying project to /workspace/app"
# Copy the entire project
COPY . /workspace/app

CMD [ "/root/onstart.sh" ]
