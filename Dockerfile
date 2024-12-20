FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget $MINICONDA_URL -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda create -n yolo python=3.11 -y && \
    conda activate yolo && \
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    git clone https://github.com/luckycontrol/MOAI_yolo.git && \
    cd MOAI_yolo && \
    pip install -r requirements.txt && \
    cd / && \
    conda clean -afy"

RUN mkdir /app && \
    cp -r /MOAI_yolo /app/MOAI_yolo && \
    cp -r /opt/conda /app/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" \
    PATH=/opt/conda/bin:$PATH

COPY --from=builder /app/MOAI_yolo /MOAI_yolo
COPY --from=builder /app/conda /opt/conda

WORKDIR /MOAI_yolo

VOLUME ["/MOAI_yolo", "/moai"]