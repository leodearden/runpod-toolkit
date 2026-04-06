"""Manage vLLM inference pods on RunPod."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from runpod_toolkit.compute.runpod_client import RunPodClient, PodInfo
from runpod_toolkit.serve.health import check_health
from runpod_toolkit.serve.model_registry import ModelConfig, ModelVariant

logger = logging.getLogger(__name__)

VLLM_DOCKER_IMAGE = "leosiriusdawn/runpod-vllm:latest"


@dataclass
class VllmPodSpec:
    """Resolved specification for launching a vLLM pod."""
    model_id: str
    variant_name: str
    dtype: str
    quantization: str | None
    tp_size: int
    max_model_len: int
    gpu_type: str
    gpu_count: int
    volume_id: str | None
    extra_vllm_args: dict


def resolve_pod_spec(
    config: ModelConfig,
    variant_name: str,
    gpu_type: str,
    gpu_count: int | None = None,
    volume_id: str | None = None,
) -> VllmPodSpec:
    """Build a VllmPodSpec from a ModelConfig and variant name."""
    variant = config.variants.get(variant_name)
    if variant is None:
        available = list(config.variants.keys())
        raise ValueError(f"Unknown variant {variant_name!r} for {config.model}. Available: {available}")

    return VllmPodSpec(
        model_id=config.model,
        variant_name=variant_name,
        dtype=variant.dtype,
        quantization=variant.quantization,
        tp_size=variant.gpu_requirements.min_tp,
        max_model_len=variant.max_model_len,
        gpu_type=gpu_type,
        gpu_count=gpu_count or variant.gpu_requirements.min_tp,
        volume_id=volume_id,
        extra_vllm_args=config.vllm_args,
    )


def start_vllm_pod(
    client: RunPodClient,
    spec: VllmPodSpec,
    pod_name: str | None = None,
) -> PodInfo:
    """Create and start a RunPod pod running vLLM."""
    env_vars = {
        "MODEL_NAME": spec.model_id,
        "DTYPE": spec.dtype,
        "TP_SIZE": str(spec.tp_size),
        "MAX_MODEL_LEN": str(spec.max_model_len),
    }
    if spec.quantization:
        env_vars["QUANTIZATION"] = spec.quantization

    name = pod_name or f"vllm-{spec.variant_name}"

    pod = client.create_pod(
        name=name,
        gpu_type=spec.gpu_type,
        gpu_count=spec.gpu_count,
        image=VLLM_DOCKER_IMAGE,
        network_volume_id=spec.volume_id,
        env_vars=env_vars,
    )
    logger.info("Created vLLM pod %s (%s) for %s", pod.id, pod.name, spec.model_id)
    return pod


def stop_vllm_pod(client: RunPodClient, pod_id: str) -> None:
    """Terminate a vLLM pod."""
    client.terminate_pod(pod_id)
    logger.info("Terminated vLLM pod %s", pod_id)


def wait_for_healthy(
    client: RunPodClient,
    pod_id: str,
    timeout: float = 600,
    poll_interval: float = 15,
) -> bool:
    """Wait until the vLLM pod reports healthy, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod = client.get_pod(pod_id)
        if pod and pod.ssh_host and pod.ssh_port:
            ok, _ = check_health(pod.ssh_host, pod.ssh_port)
            if ok:
                logger.info("Pod %s is healthy", pod_id)
                return True
        time.sleep(poll_interval)
    logger.warning("Pod %s did not become healthy within %.0fs", pod_id, timeout)
    return False
