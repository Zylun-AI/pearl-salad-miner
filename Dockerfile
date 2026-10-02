# ============================================================
# ZYI 1.6 BASE 180M - SALAD CLOUD
# GTX 1050 Ti 4GB
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Hugging Face cache
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
ENV HF_HUB_CACHE=/cache/huggingface

# PyTorch / CUDA
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ZYI defaults
ENV MODEL_REPO=caikybaldo999/ZYI-1.6-BASE-180M-BETA
ENV MODEL_DIR=/models/zyi

ENV ZYI_DEVICE=cuda
ENV ZYI_DTYPE=float16

# GTX 1050 Ti só tem 4GB:
# texto e VAE ficam na CPU
ENV ZYI_TEXT_DEVICE=cpu
ENV ZYI_VAE_DEVICE=cpu

ENV ZYI_TORCH_COMPILE=0
ENV ZYI_MAX_CONCURRENT=1

ENV ZYI_DEFAULT_STEPS=8
ENV ZYI_DEFAULT_CFG=1.0

ENV PORT=8000

WORKDIR /app


# ============================================================
# SISTEMA
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip setuptools wheel


# ============================================================
# PYTORCH CUDA 11.8
#
# CUDA 11.8 é uma escolha mais segura para Pascal / GTX 1050 Ti.
# ============================================================

RUN pip3 install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu118


# ============================================================
# DEPENDÊNCIAS ZYI
# ============================================================

RUN pip3 install \
    fastapi \
    "uvicorn[standard]" \
    huggingface_hub \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    pillow \
    numpy \
    einops \
    scipy \
    requests \
    python-multipart \
    psutil


# ============================================================
# PASTAS
# ============================================================

RUN mkdir -p \
    /models/zyi \
    /cache/huggingface \
    /app/outputs


# ============================================================
# SEU SERVIDOR
#
# Estrutura esperada:
#
# Dockerfile
# server.py
#
# Se você tiver outros arquivos Python, troque por:
# COPY . /app/
# ============================================================

COPY server/app.py /app/server.py
COPY server/model_zyi.py /app/model_zyi.py


# ============================================================
# DOWNLOAD DO MODELO
#
# NÃO baixa durante o docker build.
# Baixa quando o container inicia.
#
# Isso permite usar HF_TOKEN como variável secreta no Salad.
# ============================================================

RUN cat > /app/start.py <<'PY'
import os
import sys
import subprocess

from huggingface_hub import snapshot_download


repo = os.environ.get(
    "MODEL_REPO",
    "caikybaldo999/ZYI-1.6-BASE-180M-BETA",
)

model_dir = os.environ.get(
    "MODEL_DIR",
    "/models/zyi",
)

token = os.environ.get("HF_TOKEN") or None


print("=" * 70)
print("ZYI SALAD CLOUD SERVER")
print("=" * 70)

print("Model repo:", repo)
print("Model dir :", model_dir)

os.makedirs(
    model_dir,
    exist_ok=True,
)


# ------------------------------------------------------------
# VER CUDA
# ------------------------------------------------------------

import torch

print()
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    total = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM: {total:.2f} GB"
    )

else:

    print(
        "ERRO: nenhuma GPU CUDA detectada."
    )

    sys.exit(1)


# ------------------------------------------------------------
# BAIXAR MODELO
# ------------------------------------------------------------

print()
print("Baixando/verificando modelo...")

snapshot_download(
    repo_id=repo,
    repo_type="model",
    local_dir=model_dir,
    token=token,
)

print()
print("Modelo pronto.")


# ------------------------------------------------------------
# INICIAR FASTAPI
# ------------------------------------------------------------

port = os.environ.get(
    "PORT",
    "8000",
)

print()
print(
    f"Iniciando servidor na porta {port}..."
)

os.execvp(
    "uvicorn",
    [
        "uvicorn",
        "server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),

        # IMPORTANTE:
        # somente um processo,
        # senão cada worker carregaria outra cópia do modelo.
        "--workers",
        "1",

        "--timeout-keep-alive",
        "120",
    ],
)
PY


# ============================================================
# PORTA
# ============================================================

EXPOSE 8000


# ============================================================
# HEALTH CHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=180s \
    --retries=5 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1


# ============================================================
# CONTAINER SEMPRE ATIVO
#
# O processo abaixo é o PID principal do container.
# Fechar browser/SSH/Jupyter NÃO encerra o servidor.
# ============================================================

CMD ["python3", "/app/start.py"]
