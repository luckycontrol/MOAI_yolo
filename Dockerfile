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

# conda 초기화 및 yolo 가상환경 생성 및 활성화
RUN /opt/conda/bin/conda init bash && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n yolo python=3.11 -y && \
    conda activate yolo && \
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    git clone https://github.com/luckycontrol/MOAI_yolo.git && \
    cd MOAI_yolo && \
    pip install -r requirements.txt && \
    pip install opencv-python-headless && \
    cd / && \
    conda clean -afy"

RUN mkdir /app && \
    cp -r /MOAI_yolo /app/MOAI_yolo && \
    cp -r /opt/conda /app/conda

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/MOAI_yolo /MOAI_yolo
COPY --from=builder /app/conda /opt/conda

WORKDIR /MOAI_yolo

# 컨테이너 시작 시 conda 환경 활성화
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate yolo" >> /root/.bashrc

CMD ["/bin/bash"]

VOLUME ["/MOAI_yolo", "/moai"]