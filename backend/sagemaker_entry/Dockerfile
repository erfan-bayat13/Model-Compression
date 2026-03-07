# Stage 1: grab NeMo LLM scripts from the official container (no need to ship full image)
FROM nvcr.io/nvidia/nemo:25.04 AS nemo-scripts

# Stage 2: slim CUDA runtime + PyTorch + NeMo toolkit
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

RUN pip install --no-cache-dir \
    "nemo_toolkit[llm]" \
    "nemo-run" \
    transformers \
    huggingface_hub \
    boto3

RUN mkdir -p /opt/NeMo/scripts
COPY --from=nemo-scripts /opt/NeMo/scripts /opt/NeMo/scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/compression:$PYTHONPATH

WORKDIR /opt/compression
COPY . .

RUN mkdir -p /opt/ml/code && \
    printf '#!/bin/bash\npython3 /opt/compression/compress.py \"$@\"\n' > /opt/ml/code/train && \
    chmod +x /opt/ml/code/train

ENTRYPOINT ["python3", "/opt/compression/compress.py"]