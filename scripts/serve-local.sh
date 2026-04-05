#!/bin/bash
set -e

case "${1:-}" in
    qwen3-30b)
        MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
        ;;
    devstral)
        MODEL="mistralai/Devstral-Small-2505"
        ;;
    qwen25-32b)
        MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
        ;;
    *)
        echo "Usage: $0 {qwen3-30b|devstral|qwen25-32b}"
        echo ""
        echo "Starts vLLM on localhost:8000 for workstation 3090 evaluation."
        echo "Use with: orchestrator eval --vllm-url http://localhost:8000"
        exit 1
        ;;
esac

echo "Starting vLLM for: $MODEL"
echo "Endpoint: http://localhost:8000"
echo ""

exec vllm serve "$MODEL" \
    --dtype auto \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --api-key dummy \
    --port 8000 \
    --host 0.0.0.0
