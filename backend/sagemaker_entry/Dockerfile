# =============================================================================
# NeMo Compression Container
# =============================================================================
# Pinned to nvcr.io/nvidia/nemo:25.04 — the last container where NeMo 2.0 LLM
# support (gpt_prune.py, gpt_train.py, ptq.py) is fully working.
# NeMo 2.0 LLM was deprecated in 25.11 when NVIDIA pivoted to audio/speech.
# DO NOT upgrade to 25.11+ — gpt_prune.py and friends will not be available.
# =============================================================================
FROM nvcr.io/nvidia/nemo:25.04

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Copy our codebase into the container
# We copy the whole backend/ directory into /opt/compression/
# The entrypoint will live at /opt/compression/entrypoint.py
# -----------------------------------------------------------------------------
WORKDIR /opt/compression
COPY . .

# -----------------------------------------------------------------------------
# Install our Python dependencies on top of the NeMo base image
# NeMo, PyTorch, CUDA are already installed in the base image — we only
# add what's specific to our service
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    transformers \
    huggingface_hub \
    boto3

# -----------------------------------------------------------------------------
# SageMaker entrypoint
# SageMaker Training Jobs call the container with:
#   train  (default command for training jobs)
# We map that to our compress.py via the train script convention.
# SageMaker looks for an executable at /opt/ml/code/train — we create that.
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/opt/compression:$PYTHONPATH

# Create the SageMaker-expected train script
# SageMaker executes /opt/ml/code/train when the job starts
RUN mkdir -p /opt/ml/code
RUN echo '#!/bin/bash\npython /opt/compression/compress.py "$@"' \
    > /opt/ml/code/train && chmod +x /opt/ml/code/train

# Default entrypoint for local testing (not used by SageMaker)
ENTRYPOINT ["python", "/opt/compression/compress.py"]