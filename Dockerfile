FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV HF_HOME=/cache/huggingface
ENV HF_HUB_CACHE=/cache/huggingface

ENV MODEL_REPO=caikybaldo999/ZYI-1.6-BASE-180M-BETA
ENV MODEL_DIR=/models/zyi

ENV ZYI_DEVICE=cuda
ENV ZYI_DTYPE=float16

# GTX 1050 Ti 4GB
ENV ZYI_TEXT_DEVICE=cpu
ENV ZYI_VAE_DEVICE=cpu
ENV ZYI_TORCH_COMPILE=0

ENV ZYI_MAX_CONCURRENT=1
ENV ZYI_DEFAULT_STEPS=8
ENV ZYI_DEFAULT_CFG=1.0

ENV PORT=8000

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        git-lfs \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch com CUDA compatível com GTX 1050 Ti
RUN pip3 install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

COPY server/ /app/server/

RUN mkdir -p \
    /models/zyi \
    /cache/huggingface \
    /app/outputs

EXPOSE 8000

CMD ["python3", "-u", "-m", "uvicorn", "server.app:app", "--host", "::", "--port", "8000", "--workers", "1", "--log-level", "info"]
