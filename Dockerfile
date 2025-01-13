FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" \
    PATH=/MOAI_yolo/yolo/bin:$PATH

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libsm6 libxext6 libxrender-dev && \
    wget -qO- https://astral.sh/uv/install.sh | sh && \
    source "$HOME/.local/bin/env" && \
    git clone https://github.com/luckycontrol/MOAI_yolo.git && \
    cd MOAI_yolo && \
    uv init --python 3.11 && \
    uv venv yolo && \
    source yolo/bin/activate && \
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install ultralytics tensorboard opencv-python-headless && \
    uv pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /MOAI_yolo

CMD ["/bin/bash"]

VOLUME ["/MOAI_yolo", "/moai"]