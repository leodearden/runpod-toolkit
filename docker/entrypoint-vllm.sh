#!/bin/bash
set -e

# Start SSH daemon for RunPod console access and health checks
/usr/sbin/sshd

# Required
MODEL_NAME="${MODEL_NAME:?MODEL_NAME env var required}"

# Defaults
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Build vLLM command
CMD="python3 -m vllm.entrypoints.openai.api_server"
CMD="$CMD --model $MODEL_NAME"
CMD="$CMD --dtype $DTYPE"
CMD="$CMD --tensor-parallel-size $TP_SIZE"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --host $HOST --port $PORT"
CMD="$CMD --api-key dummy"

# Optional quantization
if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Enable prefix caching by default (great for code completion)
CMD="$CMD --enable-prefix-caching"

# Use models cache directory if it exists (volume mount)
if [ -d "/workspace" ]; then
    export HF_HOME="${HF_HOME:-/workspace/models}"
    mkdir -p /workspace/tmp /workspace/torch_cache
    export TMPDIR=/workspace/tmp
    export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache
fi

echo "Starting vLLM: $CMD"
exec $CMD
