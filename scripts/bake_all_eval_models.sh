#!/bin/bash
# Sequentially bake + push every vLLM eval model image, then delete the
# local image to free disk before the next build.
#
# Why sequential?
#   - dockerd serializes large overlay2 I/O; concurrent build+push contend
#     for the same disk and cause "Preparing" hangs
#     (memory: feedback_dockerd_push_build_contention.md)
#   - Internal-2nd has 948 GB free at session start; total baked size
#     across all 5 models is ~593 GB. Push-and-delete keeps headroom.
#
# Logs go to /var/tmp/dark-factory-bake/<tag>-{build,push}.log so they
# survive reboots (unlike /tmp tmpfs).

set -uo pipefail

LOG_DIR=/var/tmp/dark-factory-bake
mkdir -p "$LOG_DIR"

BAKE=/home/leo/src/runpod-toolkit/scripts/bake_model_image.py
HUB=leosiriusdawn/runpod-vllm

# model_hf_name -> docker tag suffix
declare -a MODELS=(
    "Qwen/Qwen3-Coder-Next-FP8|qwen3-coder-next-fp8-baked"
    "lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4|reap-139b-nvfp4-baked"
    "saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10|reap-172b-nvfp4-gb10-baked"
    "nvidia/MiniMax-M2.5-NVFP4|minimax-m25-nvfp4-baked"
    "MiniMaxAI/MiniMax-M2.5|minimax-m25-fp8-baked"
)

# Allow caller to skip already-built models. Caller passes a regex of tags
# to skip via SKIP=<regex>; default skips nothing.
SKIP="${SKIP:-^$}"

for entry in "${MODELS[@]}"; do
    HF_MODEL="${entry%%|*}"
    TAG_SUFFIX="${entry##*|}"
    TAG="$HUB:$TAG_SUFFIX"

    if [[ "$TAG_SUFFIX" =~ $SKIP ]]; then
        echo "[$(date +%H:%M:%S)] SKIP $TAG (matches SKIP=$SKIP)"
        continue
    fi

    BUILD_LOG="$LOG_DIR/$TAG_SUFFIX-build.log"
    PUSH_LOG="$LOG_DIR/$TAG_SUFFIX-push.log"

    echo "[$(date +%H:%M:%S)] === $HF_MODEL → $TAG ==="
    echo "[$(date +%H:%M:%S)] BUILD start (log: $BUILD_LOG)"
    if ! python3 "$BAKE" \
        --model "$HF_MODEL" \
        --tag "$TAG" \
        --build \
        --build-log "$BUILD_LOG"; then
        echo "[$(date +%H:%M:%S)] BUILD FAILED for $TAG; see $BUILD_LOG"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] BUILD ok"

    echo "[$(date +%H:%M:%S)] PUSH start (log: $PUSH_LOG)"
    if ! docker push "$TAG" >"$PUSH_LOG" 2>&1; then
        echo "[$(date +%H:%M:%S)] PUSH FAILED for $TAG; see $PUSH_LOG"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] PUSH ok"

    # Reclaim disk before next build. Keep :latest base layer cached.
    echo "[$(date +%H:%M:%S)] removing local image $TAG"
    docker image rm "$TAG" || true
    docker builder prune -f >/dev/null 2>&1 || true

    AVAIL=$(df -h /media/leo/Internal-2nd | awk 'NR==2 {print $4}')
    echo "[$(date +%H:%M:%S)] Internal-2nd free: $AVAIL"
done

echo "[$(date +%H:%M:%S)] All bakes complete."
