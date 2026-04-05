"""YAML-based model configuration registry."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class GpuRequirements:
    min_vram_gb: int = 80
    min_tp: int = 1


@dataclasses.dataclass(frozen=True)
class ModelVariant:
    name: str
    dtype: str = "auto"
    quantization: str | None = None
    gpu_requirements: GpuRequirements = dataclasses.field(default_factory=GpuRequirements)
    max_model_len: int = 131072


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model: str  # HuggingFace model ID
    variants: dict[str, ModelVariant] = dataclasses.field(default_factory=dict)
    vllm_args: dict[str, Any] = dataclasses.field(default_factory=dict)


def load_model_config(path: Path) -> ModelConfig:
    """Load a single model config from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    variants = {}
    for vname, vdata in data.get('variants', {}).items():
        gpu_req_data = vdata.pop('gpu_requirements', {})
        gpu_req = GpuRequirements(**gpu_req_data)
        variants[vname] = ModelVariant(name=vname, gpu_requirements=gpu_req, **vdata)

    return ModelConfig(
        model=data['model'],
        variants=variants,
        vllm_args=data.get('vllm_args', {}),
    )


_DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / 'configs' / 'models'


def load_registry(configs_dir: Path | None = None) -> dict[str, ModelConfig]:
    """Load all model configs from a directory of YAML files."""
    d = configs_dir or _DEFAULT_CONFIGS_DIR
    if not d.exists():
        return {}
    registry: dict[str, ModelConfig] = {}
    for p in sorted(d.glob('*.yaml')):
        cfg = load_model_config(p)
        registry[cfg.model] = cfg
    return registry
