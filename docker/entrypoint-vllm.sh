#!/bin/bash
set -e

# Inject PUBLIC_KEY if provided (RunPod convention)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

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
# Always trust remote code: MiniMax variants and other newer architectures
# ship custom modeling code in their HF repos that vLLM must execute to load
# the model. We control the model list, so this is safe.
CMD="$CMD --trust-remote-code"

# GPU memory utilization. vLLM defaults to 0.9, but in v0.19 the new
# CUDA-graph memory profiler reserves graph memory inside the same budget,
# leaving very little KV cache headroom on tight large-model pods. 0.95
# matches vLLM's own recommended bump and works on both H200 and 96 GB
# Blackwell. Override per-pod via the GPU_MEMORY_UTIL env var.
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTIL"

# Max concurrent sequences. vLLM defaults to 1024, which pre-allocates a
# sampler softmax buffer per slot during warm-up — at 0.97 GMU on tight
# large-model pods this OOMs the sampler warmup step. Each eval pod only
# ever serves one implementer at a time, so 16 is plenty.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"

# Optional eager mode: disables CUDA-graph capture. Workaround for the
# qwen3-coder-next-fp8 startup hang where vLLM stays alive with the model
# loaded into VRAM but /health never returns 200 (vLLM #35504, #34437).
# Set ENFORCE_EAGER=1 in the pod env to enable. Costs ~10-20% throughput.
if [ -n "$ENFORCE_EAGER" ]; then
    CMD="$CMD --enforce-eager"
fi

# No --api-key: Claude Code sends OAuth tokens that won't match a fixed key.
# Network access is controlled by pod isolation instead.

# Optional quantization
if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# Optional tokenizer mode (use 'slow' to fix Mistral Tekkenizer add_special_tokens bug)
if [ -n "$TOKENIZER_MODE" ]; then
    CMD="$CMD --tokenizer-mode $TOKENIZER_MODE"
fi

# Enable prefix caching by default (great for code completion)
CMD="$CMD --enable-prefix-caching"

# Enable tool calling (Claude Code sends tool_use blocks). Parser is per-model:
# - Devstral / Mistral models → mistral
# - Qwen models → hermes
# - Empty → disable tool calling
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
if [ -n "$TOOL_CALL_PARSER" ]; then
    CMD="$CMD --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER"
fi

# Use models cache directory if it exists (volume mount)
if [ -d "/workspace" ]; then
    export HF_HOME="${HF_HOME:-/workspace/models}"
    mkdir -p /workspace/tmp /workspace/torch_cache
    export TMPDIR=/workspace/tmp
    export TORCHINDUCTOR_CACHE_DIR=/workspace/torch_cache
fi

echo "Starting vLLM: $CMD"
exec $CMD
