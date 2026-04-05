"""GPU specifications lookup table with region-aware RAM values.

RunPod API provides VRAM but NOT system RAM. System RAM varies by datacenter/region.
This module provides a lookup table with known configurations.

Usage:
    >>> specs = get_gpu_specs("NVIDIA GeForce RTX 4090", datacenter="EU-RO-1")
    >>> print(f"Min RAM: {specs.min_ram_gb}GB")
    Min RAM: 60GB

    >>> specs = get_gpu_specs("NVIDIA GeForce RTX 4090")  # No datacenter = conservative default
    >>> print(f"Min RAM: {specs.min_ram_gb}GB")
    Min RAM: 31GB
"""

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass(frozen=True)
class GpuSpecs:
    """Specifications for a GPU type in a specific region.

    Attributes:
        name: Full GPU name as returned by RunPod API.
        vram_gb: Video RAM in GB.
        min_ram_gb: Conservative system RAM estimate (use for calculations).
        typical_ram_gb: Typical system RAM observed.
        max_ram_gb: Maximum system RAM observed.
        typical_cpu_cores: Typical CPU cores available.
        compute_tier: "consumer" | "prosumer" | "datacenter"
    """

    name: str
    vram_gb: int
    min_ram_gb: int
    typical_ram_gb: int
    max_ram_gb: int
    typical_cpu_cores: int
    compute_tier: str  # "consumer" | "prosumer" | "datacenter"


# Region-aware GPU specifications lookup table.
# Structure: GPU_SPECS[gpu_name][datacenter_or_default] = GpuSpecs
#
# Known RAM variations by region (discovered empirically):
# - EU-RO-1: Higher RAM configurations (60GB for RTX 4090, 251GB for A100 PCIe)
# - US/Other: Lower RAM configurations (31GB for RTX 4090, 117GB for A100 PCIe)
#
# "default" key provides conservative fallback for unknown datacenters.

GPU_SPECS: Dict[str, Dict[str, GpuSpecs]] = {
    # Consumer GPUs
    "NVIDIA GeForce RTX 3090": {
        "default": GpuSpecs(
            name="NVIDIA GeForce RTX 3090",
            vram_gb=24,
            min_ram_gb=32,
            typical_ram_gb=64,
            max_ram_gb=128,
            typical_cpu_cores=8,
            compute_tier="consumer",
        ),
    },
    "NVIDIA GeForce RTX 4090": {
        "EU-RO-1": GpuSpecs(
            name="NVIDIA GeForce RTX 4090",
            vram_gb=24,
            min_ram_gb=60,
            typical_ram_gb=60,
            max_ram_gb=60,
            typical_cpu_cores=12,
            compute_tier="consumer",
        ),
        "default": GpuSpecs(
            name="NVIDIA GeForce RTX 4090",
            vram_gb=24,
            min_ram_gb=31,
            typical_ram_gb=64,
            max_ram_gb=128,
            typical_cpu_cores=12,
            compute_tier="consumer",
        ),
    },
    # Prosumer GPUs
    "NVIDIA RTX A4000": {
        "default": GpuSpecs(
            name="NVIDIA RTX A4000",
            vram_gb=16,
            min_ram_gb=32,
            typical_ram_gb=64,
            max_ram_gb=128,
            typical_cpu_cores=8,
            compute_tier="prosumer",
        ),
    },
    "NVIDIA RTX A5000": {
        "default": GpuSpecs(
            name="NVIDIA RTX A5000",
            vram_gb=24,
            min_ram_gb=32,
            typical_ram_gb=64,
            max_ram_gb=128,
            typical_cpu_cores=8,
            compute_tier="prosumer",
        ),
    },
    "NVIDIA RTX A6000": {
        "default": GpuSpecs(
            name="NVIDIA RTX A6000",
            vram_gb=48,
            min_ram_gb=64,
            typical_ram_gb=128,
            max_ram_gb=256,
            typical_cpu_cores=16,
            compute_tier="prosumer",
        ),
    },
    # Datacenter GPUs
    "NVIDIA A40": {
        "default": GpuSpecs(
            name="NVIDIA A40",
            vram_gb=48,
            min_ram_gb=64,
            typical_ram_gb=128,
            max_ram_gb=256,
            typical_cpu_cores=16,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA A100-SXM4-80GB": {
        "default": GpuSpecs(
            name="NVIDIA A100-SXM4-80GB",
            vram_gb=80,
            min_ram_gb=117,
            typical_ram_gb=256,
            max_ram_gb=512,
            typical_cpu_cores=32,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA A100 80GB PCIe": {
        "EU-RO-1": GpuSpecs(
            name="NVIDIA A100 80GB PCIe",
            vram_gb=80,
            min_ram_gb=251,
            typical_ram_gb=251,
            max_ram_gb=251,
            typical_cpu_cores=32,
            compute_tier="datacenter",
        ),
        "default": GpuSpecs(
            name="NVIDIA A100 80GB PCIe",
            vram_gb=80,
            min_ram_gb=117,
            typical_ram_gb=256,
            max_ram_gb=512,
            typical_cpu_cores=32,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA L40": {
        "default": GpuSpecs(
            name="NVIDIA L40",
            vram_gb=48,
            min_ram_gb=64,
            typical_ram_gb=128,
            max_ram_gb=256,
            typical_cpu_cores=16,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA L40S": {
        "default": GpuSpecs(
            name="NVIDIA L40S",
            vram_gb=48,
            min_ram_gb=64,
            typical_ram_gb=128,
            max_ram_gb=256,
            typical_cpu_cores=16,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA H100 PCIe": {
        "default": GpuSpecs(
            name="NVIDIA H100 PCIe",
            vram_gb=80,
            min_ram_gb=128,
            typical_ram_gb=256,
            max_ram_gb=512,
            typical_cpu_cores=32,
            compute_tier="datacenter",
        ),
    },
    "NVIDIA H100 80GB HBM3": {
        "default": GpuSpecs(
            name="NVIDIA H100 80GB HBM3",
            vram_gb=80,
            min_ram_gb=128,
            typical_ram_gb=256,
            max_ram_gb=512,
            typical_cpu_cores=32,
            compute_tier="datacenter",
        ),
    },
}

# Default specs for unknown GPUs (very conservative)
_UNKNOWN_GPU_SPECS = GpuSpecs(
    name="Unknown GPU",
    vram_gb=8,
    min_ram_gb=32,
    typical_ram_gb=32,
    max_ram_gb=32,
    typical_cpu_cores=8,
    compute_tier="consumer",
)


def get_gpu_specs(
    gpu_name: str,
    datacenter: Optional[str] = None,
) -> GpuSpecs:
    """Get GPU specifications, optionally region-aware.

    Args:
        gpu_name: Full GPU name as returned by RunPod API.
        datacenter: Datacenter ID (e.g., "EU-RO-1"). If None, uses conservative default.

    Returns:
        GpuSpecs for the GPU in the specified datacenter, or default if not found.

    Example:
        >>> specs = get_gpu_specs("NVIDIA GeForce RTX 4090", datacenter="EU-RO-1")
        >>> specs.min_ram_gb
        60
        >>> specs = get_gpu_specs("NVIDIA GeForce RTX 4090")
        >>> specs.min_ram_gb
        31
    """
    if gpu_name not in GPU_SPECS:
        # Return unknown specs with the actual GPU name
        return GpuSpecs(
            name=gpu_name,
            vram_gb=_UNKNOWN_GPU_SPECS.vram_gb,
            min_ram_gb=_UNKNOWN_GPU_SPECS.min_ram_gb,
            typical_ram_gb=_UNKNOWN_GPU_SPECS.typical_ram_gb,
            max_ram_gb=_UNKNOWN_GPU_SPECS.max_ram_gb,
            typical_cpu_cores=_UNKNOWN_GPU_SPECS.typical_cpu_cores,
            compute_tier=_UNKNOWN_GPU_SPECS.compute_tier,
        )

    region_specs = GPU_SPECS[gpu_name]

    # Try region-specific specs first, then fall back to default
    if datacenter and datacenter in region_specs:
        return region_specs[datacenter]

    return region_specs["default"]


def get_all_known_gpus() -> List[str]:
    """Get list of all GPU names with known specifications.

    Returns:
        List of GPU names in the lookup table.
    """
    return list(GPU_SPECS.keys())


def is_gpu_known(gpu_name: str) -> bool:
    """Check if a GPU has known specifications.

    Args:
        gpu_name: Full GPU name as returned by RunPod API.

    Returns:
        True if the GPU is in the lookup table.
    """
    return gpu_name in GPU_SPECS


def get_regions_for_gpu(gpu_name: str) -> List[str]:
    """Get list of regions with specific specs for a GPU.

    Args:
        gpu_name: Full GPU name as returned by RunPod API.

    Returns:
        List of datacenter IDs with region-specific specs (excludes "default").
    """
    if gpu_name not in GPU_SPECS:
        return []

    return [dc for dc in GPU_SPECS[gpu_name].keys() if dc != "default"]
