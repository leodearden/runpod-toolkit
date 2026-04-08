"""RunPod API client wrapper.

Provides a high-level interface for RunPod operations:
- GPU availability queries
- Pod lifecycle management (create, list, terminate)
- Network Volume management (create, list, delete)

Example:
    >>> from runpod_toolkit.compute import RunPodClient
    >>> from runpod_toolkit.config import RunPodConfig
    >>>
    >>> config = RunPodConfig.from_env()
    >>> client = RunPodClient(config)
    >>>
    >>> # Check GPU availability
    >>> gpus = client.get_gpu_availability(min_memory_gb=24)
    >>> for gpu in gpus:
    ...     print(f"{gpu.name}: {gpu.available} available")
    >>>
    >>> # Create a pod
    >>> pod = client.create_pod(
    ...     name="experiment-1",
    ...     gpu_type="NVIDIA RTX A5000",
    ...     network_volume_id="vol_xxx",
    ... )
"""

import logging
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

import backoff
import requests.exceptions

from runpod_toolkit.config import RunPodConfig

logger = logging.getLogger(__name__)

# Transient network errors worth retrying on. Intentionally explicit —
# do NOT use requests.exceptions.RequestException (parent class covers
# auth and malformed-request errors that should not be retried).
_TRANSIENT = (
    requests.exceptions.ConnectionError,
    socket.gaierror,
    ConnectionResetError,
    TimeoutError,
)

# Lazy import runpod to handle missing SDK gracefully
_runpod = None


def _get_runpod():
    """Lazy import of runpod SDK."""
    global _runpod
    if _runpod is None:
        try:
            import runpod
            _runpod = runpod
        except ImportError:
            raise ImportError(
                "runpod SDK not installed. Install with: pip install runpod>=1.8.0"
            )
    return _runpod


class PodStatus(Enum):
    """Pod status enum."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    EXITED = "EXITED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


class VolumeStatus(Enum):
    """Network volume status enum."""
    AVAILABLE = "AVAILABLE"
    IN_USE = "IN_USE"
    CREATING = "CREATING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"


@dataclass
class GpuInfo:
    """Information about GPU availability."""
    id: str
    name: str
    memory_gb: int
    available: int
    hourly_cost: float
    datacenter: str

    @classmethod
    def from_api(cls, data: Dict[str, Any], datacenter: str = "") -> "GpuInfo":
        """Create from RunPod API response."""
        return cls(
            id=data.get("id", ""),
            name=data.get("displayName", data.get("id", "")),
            memory_gb=data.get("memoryInGb", 0),
            available=data.get("stockStatus", 0),
            hourly_cost=data.get("securePrice", 0.0),
            datacenter=datacenter,
        )


@dataclass
class PodInfo:
    """Information about a RunPod pod."""
    id: str
    name: str
    status: PodStatus
    gpu_type: str
    gpu_count: int
    created_at: str
    cost_per_hour: float
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    network_volume_id: Optional[str] = None
    datacenter: Optional[str] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "PodInfo":
        """Create from RunPod API response."""
        # Parse runtime info for SSH details
        runtime = data.get("runtime", {}) or {}
        ports = runtime.get("ports", []) or []

        ssh_host = None
        ssh_port = None
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_host = port.get("ip")
                ssh_port = port.get("publicPort")
                break

        # Parse status
        status_str = data.get("desiredStatus", "UNKNOWN")
        try:
            status = PodStatus(status_str)
        except ValueError:
            status = PodStatus.UNKNOWN

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=status,
            gpu_type=data.get("machine", {}).get("gpuDisplayName", "Unknown"),
            gpu_count=data.get("gpuCount", 1),
            created_at=data.get("createdAt", ""),
            cost_per_hour=data.get("costPerHr", 0.0),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            network_volume_id=data.get("networkVolumeId"),
            datacenter=data.get("machine", {}).get("location", ""),
        )


@dataclass
class VolumeInfo:
    """Information about a RunPod network volume."""
    id: str
    name: str
    size_gb: int
    status: VolumeStatus
    datacenter: str
    created_at: str
    used_by_pods: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "VolumeInfo":
        """Create from RunPod API response."""
        # Parse status
        status_str = data.get("status", "UNKNOWN")
        try:
            status = VolumeStatus(status_str)
        except ValueError:
            status = VolumeStatus.UNKNOWN

        # Get pod IDs using this volume
        pods = data.get("pods", []) or []
        pod_ids = [p.get("id") for p in pods if p.get("id")]

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            size_gb=data.get("size", 0),
            status=status,
            datacenter=data.get("dataCenterId", ""),
            created_at=data.get("createdAt", ""),
            used_by_pods=pod_ids,
        )


class RunPodError(Exception):
    """RunPod API error."""
    pass


class RunPodClient:
    """Client for RunPod API operations.

    Provides methods for:
    - Querying GPU availability
    - Creating and managing pods
    - Creating and managing network volumes
    """

    # Default container image (custom image with mamba-ssm pre-built)
    DEFAULT_IMAGE = "leosiriusdawn/market-predictor:cu126"

    # Default container disk size (GB)
    DEFAULT_CONTAINER_DISK = 50

    # Default volume mount path
    DEFAULT_VOLUME_MOUNT = "/workspace"

    def __init__(self, config: RunPodConfig):
        """Initialize RunPod client.

        Args:
            config: RunPod configuration with API key.
        """
        self.config = config
        self._runpod = _get_runpod()

        # Set API key
        self._runpod.api_key = config.api_key

    # =========================================================================
    # GPU Queries
    # =========================================================================

    # GraphQL query for GPU types with pricing and availability
    _GPU_TYPES_QUERY = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        securePrice
        communityPrice
        maxGpuCount
        maxGpuCountSecureCloud
        maxGpuCountCommunityCloud
      }
    }
    """

    def get_gpu_availability(
        self,
        min_memory_gb: int = 24,
        datacenter: Optional[str] = None,
        include_community: bool = False,
    ) -> List[GpuInfo]:
        """Get available GPUs meeting criteria.

        Args:
            min_memory_gb: Minimum GPU memory in GB.
            datacenter: Optional datacenter filter (not used in current API).
            include_community: Include community cloud GPUs (default: False).

        Returns:
            List of available GPU types.
        """
        import requests

        try:
            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": self._GPU_TYPES_QUERY},
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            gpus = data.get("data", {}).get("gpuTypes", [])

            results = []
            for gpu in gpus:
                memory = gpu.get("memoryInGb", 0)

                # Filter by memory
                if memory < min_memory_gb:
                    continue

                # Check cloud availability
                has_secure = gpu.get("secureCloud", False)
                has_community = gpu.get("communityCloud", False) and include_community

                if not has_secure and not has_community:
                    continue

                # Get price and availability (prefer secure cloud)
                if has_secure and gpu.get("securePrice"):
                    price = gpu.get("securePrice", 0)
                    available = gpu.get("maxGpuCountSecureCloud", 0)
                elif has_community and gpu.get("communityPrice"):
                    price = gpu.get("communityPrice", 0)
                    available = gpu.get("maxGpuCountCommunityCloud", 0)
                else:
                    continue

                # Skip if no availability (maxGpuCount indicates capacity, not current stock)
                # We use maxGpuCount as a proxy for "available"
                if available == 0:
                    continue

                results.append(GpuInfo(
                    id=gpu.get("id", ""),
                    name=gpu.get("id", ""),  # Full GPU name from id field
                    memory_gb=memory,
                    available=available,
                    hourly_cost=price,
                    datacenter=datacenter or "",
                ))

            # Sort by cost
            results.sort(key=lambda g: g.hourly_cost)
            return results

        except requests.RequestException as e:
            raise RunPodError(f"Failed to query RunPod API: {e}") from e
        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to get GPU availability: {e}") from e

    # =========================================================================
    # Datacenter Queries
    # =========================================================================

    _DATACENTERS_QUERY = """
    query DataCenters {
      dataCenters {
        id
        name
        location
        listed
      }
    }
    """

    def get_datacenters(self, listed_only: bool = True) -> List[Dict[str, Any]]:
        """Get available datacenters.

        Args:
            listed_only: Only return listed (publicly available) datacenters.

        Returns:
            List of datacenter dicts with id, name, location, listed fields.
        """
        import requests

        try:
            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": self._DATACENTERS_QUERY},
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            dcs = data.get("data", {}).get("dataCenters", [])
            if listed_only:
                dcs = [dc for dc in dcs if dc.get("listed")]
            return sorted(dcs, key=lambda dc: dc.get("id", ""))

        except requests.RequestException as e:
            raise RunPodError(f"Failed to query datacenters: {e}") from e
        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to get datacenters: {e}") from e

    # =========================================================================
    # Pod Operations
    # =========================================================================

    def create_pod(
        self,
        name: str,
        gpu_type: str,
        network_volume_id: Optional[str] = None,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        docker_args: Optional[str] = None,
        gpu_count: int = 1,
        container_disk_gb: int = None,
        volume_mount_path: str = None,
        datacenter: Optional[str] = None,
        min_memory_gb: Optional[int] = None,
    ) -> PodInfo:
        """Create a new pod.

        Args:
            name: Pod name.
            gpu_type: GPU type (e.g., "NVIDIA RTX A5000").
            network_volume_id: Optional network volume to attach.
            image: Docker image (default: PyTorch image).
            env_vars: Environment variables.
            docker_args: Docker arguments.
            gpu_count: Number of GPUs.
            container_disk_gb: Container disk size in GB.
            volume_mount_path: Mount path for network volume.
            datacenter: Preferred datacenter.
            min_memory_gb: Minimum system RAM in GB (filters available hosts).

        Returns:
            PodInfo for the created pod.

        Raises:
            RunPodError: If pod creation fails.
        """
        try:
            # Build pod spec using snake_case keys for the SDK
            pod_spec = {
                "name": name,
                "image_name": image or self.DEFAULT_IMAGE,
                "gpu_type_id": gpu_type,
                "gpu_count": gpu_count,
                "container_disk_in_gb": container_disk_gb or self.DEFAULT_CONTAINER_DISK,
                "volume_in_gb": 0,  # Use network volume instead
                "start_ssh": True,  # Enable SSH access
                "ports": "22/tcp",  # Expose SSH port
            }

            # Add network volume
            if network_volume_id:
                pod_spec["network_volume_id"] = network_volume_id
                pod_spec["volume_mount_path"] = volume_mount_path or self.DEFAULT_VOLUME_MOUNT

            # Add environment variables - always include RunPod API key for self-termination
            default_env = {
                "RUNPOD_API_KEY": "{{ RUNPOD_SECRET_runpod_api_key }}",
            }
            if env_vars:
                default_env.update(env_vars)
            pod_spec["env"] = default_env

            # Add docker args
            if docker_args:
                pod_spec["docker_args"] = docker_args

            # Add datacenter preference
            if datacenter:
                pod_spec["data_center_id"] = datacenter

            # Add minimum RAM requirement
            if min_memory_gb:
                pod_spec["min_memory_in_gb"] = min_memory_gb

            logger.info(f"Creating pod '{name}' with GPU {gpu_type}")
            response = self._runpod.create_pod(**pod_spec)

            if not response or "id" not in response:
                raise RunPodError(f"Invalid response from RunPod: {response}")

            return PodInfo.from_api(response)

        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to create pod: {e}") from e

    # Default CPU image (RunPod base image with SSH support)
    DEFAULT_CPU_IMAGE = "runpod/base:1.0.2-ubuntu2404"

    # Default CPU flavor (cpu3c = 3rd gen compute optimized)
    DEFAULT_CPU_FLAVOR = "cpu3c"

    # Bidirectional CPU flavor fallbacks (3rd gen ↔ 5th gen)
    # Used when primary flavor has no available instances
    CPU_FLAVOR_FALLBACKS: Dict[str, str] = {
        "cpu3c": "cpu5c", "cpu5c": "cpu3c",
        "cpu3g": "cpu5g", "cpu5g": "cpu3g",
        "cpu3m": "cpu5m", "cpu5m": "cpu3m",
    }

    @staticmethod
    def select_cpu_flavor(vcpu_count: int, memory_gb: int) -> str:
        """Select optimal CPU flavor based on RAM/vCPU ratio.

        RunPod CPU flavors have fixed memory ratios:
        - cpu3c (compute): 2GB/vCPU
        - cpu3g (general): 4GB/vCPU
        - cpu3m (memory): 8GB/vCPU

        Args:
            vcpu_count: Number of vCPUs requested.
            memory_gb: RAM in GB requested.

        Returns:
            CPU flavor string (cpu3c, cpu3g, or cpu3m).

        Raises:
            ValueError: If ratio exceeds 8GB/vCPU (max for cpu3m).
        """
        ratio = memory_gb / vcpu_count
        if ratio <= 2:
            return "cpu3c"
        elif ratio <= 4:
            return "cpu3g"
        elif ratio <= 8:
            return "cpu3m"
        else:
            raise ValueError(
                f"RAM ratio {ratio:.1f}GB/vCPU exceeds max (8GB/vCPU for cpu3m). "
                f"Requested {memory_gb}GB for {vcpu_count} vCPUs."
            )

    def create_cpu_pod(
        self,
        name: str,
        network_volume_id: Optional[str] = None,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        docker_args: Optional[str] = None,
        container_disk_gb: int = None,
        volume_mount_path: str = None,
        datacenter: Optional[str] = None,
        vcpu_count: int = 2,
        memory_gb: int = 4,
        cpu_flavor: str = None,
    ) -> PodInfo:
        """Create a CPU-only pod.

        CPU pods are useful for data transfer operations where GPU is not needed.

        Args:
            name: Pod name.
            network_volume_id: Optional network volume to attach.
            image: Docker image (default: runpod/ubuntu:22.04).
            env_vars: Environment variables.
            docker_args: Docker arguments.
            container_disk_gb: Container disk size in GB.
            volume_mount_path: Mount path for network volume.
            datacenter: Preferred datacenter.
            vcpu_count: Number of vCPUs (default: 2).
            memory_gb: RAM in GB (default: 4).
            cpu_flavor: CPU flavor (cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m).
                        Default: cpu3c (compute optimized).

        Returns:
            PodInfo for the created pod.

        Raises:
            RunPodError: If pod creation fails.
        """
        try:
            # Auto-select flavor based on RAM/vCPU ratio if not provided
            if cpu_flavor is None:
                primary_flavor = self.select_cpu_flavor(vcpu_count, memory_gb)
                logger.info(
                    f"Auto-selected CPU flavor: {primary_flavor} "
                    f"(ratio: {memory_gb / vcpu_count:.1f}GB/vCPU)"
                )
            else:
                primary_flavor = cpu_flavor

            # Build list of flavors to try (primary + fallback if available)
            flavors_to_try = [primary_flavor]
            fallback = self.CPU_FLAVOR_FALLBACKS.get(primary_flavor)
            if fallback:
                flavors_to_try.append(fallback)

            # Build pod spec (instance_id set per-flavor in loop)
            pod_spec = {
                "name": name,
                "image_name": image or self.DEFAULT_CPU_IMAGE,
                "container_disk_in_gb": container_disk_gb or 20,  # Smaller disk for CPU pods
                "volume_in_gb": 0,  # Use network volume instead
                "start_ssh": True,
                "ports": "22/tcp",
            }

            # Add network volume
            if network_volume_id:
                pod_spec["network_volume_id"] = network_volume_id
                pod_spec["volume_mount_path"] = volume_mount_path or self.DEFAULT_VOLUME_MOUNT

            # Add environment variables - always include RunPod API key for self-termination
            default_env = {
                "RUNPOD_API_KEY": "{{ RUNPOD_SECRET_runpod_api_key }}",
            }
            if env_vars:
                default_env.update(env_vars)
            pod_spec["env"] = default_env

            # Add docker args
            if docker_args:
                pod_spec["docker_args"] = docker_args

            # Add datacenter preference
            if datacenter:
                pod_spec["data_center_id"] = datacenter

            # Try each flavor until one succeeds
            last_error = None
            for i, flavor in enumerate(flavors_to_try):
                instance_id = f"{flavor}-{vcpu_count}-{memory_gb}"
                pod_spec["instance_id"] = instance_id

                try:
                    logger.info(f"Creating CPU pod '{name}' (instance: {instance_id})")
                    response = self._runpod.create_pod(**pod_spec)

                    if not response or "id" not in response:
                        raise RunPodError(f"Invalid response from RunPod: {response}")

                    return PodInfo.from_api(response)

                except Exception as e:
                    error_str = str(e).lower()
                    # Only fallback on availability errors
                    if "no longer any instances available" in error_str:
                        last_error = e
                        if i < len(flavors_to_try) - 1:
                            next_flavor = flavors_to_try[i + 1]
                            logger.warning(
                                f"No {flavor} instances available, trying {next_flavor}..."
                            )
                            continue
                    # Non-availability error or last flavor - fail
                    if isinstance(e, RunPodError):
                        raise
                    raise RunPodError(f"Failed to create CPU pod: {e}") from e

            # All flavors exhausted (only reached if last attempt was availability error)
            raise RunPodError(
                f"Failed to create CPU pod (no {' or '.join(flavors_to_try)} available): {last_error}"
            ) from last_error

        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to create CPU pod: {e}") from e

    @backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=3, logger=logger)
    def get_pod(self, pod_id: str) -> PodInfo:
        """Get pod information.

        Retries on transient network errors (ConnectionError, DNS
        failures, ConnectionReset, TimeoutError) up to 3 times with
        exponential backoff.

        Args:
            pod_id: Pod ID.

        Returns:
            PodInfo for the pod.

        Raises:
            RunPodError: If pod not found or API error.
        """
        try:
            response = self._runpod.get_pod(pod_id)

            if not response:
                raise RunPodError(f"Pod not found: {pod_id}")

            return PodInfo.from_api(response)

        except _TRANSIENT:
            raise  # let backoff retry
        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to get pod {pod_id}: {e}") from e

    @backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=3, logger=logger)
    def list_pods(self) -> List[PodInfo]:
        """List all pods.

        Retries on transient network errors up to 3 times with
        exponential backoff.

        Returns:
            List of PodInfo objects.
        """
        try:
            response = self._runpod.get_pods()

            if not response:
                return []

            return [PodInfo.from_api(pod) for pod in response]

        except _TRANSIENT:
            raise  # let backoff retry
        except Exception as e:
            raise RunPodError(f"Failed to list pods: {e}") from e

    @backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=5, logger=logger)
    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod.

        Retries on transient network errors up to 5 times (more than
        read operations) because a leaked pod costs real money
        (~$1.69/hr). Uses exponential backoff.

        Args:
            pod_id: Pod ID.

        Returns:
            True if terminated successfully.

        Raises:
            RunPodError: If termination fails.
        """
        try:
            logger.info(f"Terminating pod {pod_id}")
            response = self._runpod.terminate_pod(pod_id)

            # Response is typically empty on success
            return True

        except _TRANSIENT:
            raise  # let backoff retry
        except Exception as e:
            raise RunPodError(f"Failed to terminate pod {pod_id}: {e}") from e

    def wait_for_pod(
        self,
        pod_id: str,
        status: PodStatus = PodStatus.RUNNING,
        timeout: int = 600,
        poll_interval: int = 10,
        wait_for_ssh: bool = False,
    ) -> PodInfo:
        """Wait for pod to reach a status.

        Args:
            pod_id: Pod ID.
            status: Target status to wait for.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.
            wait_for_ssh: If True, also wait for SSH port info to become available.
                          This can take ~60 seconds after RUNNING status is reached.

        Returns:
            PodInfo when status reached (and SSH available if wait_for_ssh=True).

        Raises:
            RunPodError: If timeout or pod enters error state.
        """
        start_time = time.time()

        while True:
            pod = self.get_pod(pod_id)

            if pod.status == status:
                # If we need to wait for SSH, check if it's available
                if wait_for_ssh and (pod.ssh_host is None or pod.ssh_port is None):
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise RunPodError(
                            f"Timeout waiting for SSH info for pod {pod_id}"
                        )
                    logger.debug(
                        f"Pod {pod_id} is {status.value} but SSH not yet available, waiting..."
                    )
                    time.sleep(poll_interval)
                    continue

                logger.info(f"Pod {pod_id} reached status {status.value}")
                if wait_for_ssh:
                    logger.info(f"Pod {pod_id} SSH available at {pod.ssh_host}:{pod.ssh_port}")
                return pod

            if pod.status == PodStatus.TERMINATED:
                raise RunPodError(f"Pod {pod_id} was terminated")

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise RunPodError(
                    f"Timeout waiting for pod {pod_id} to reach {status.value} "
                    f"(current: {pod.status.value})"
                )

            logger.debug(
                f"Pod {pod_id} status: {pod.status.value}, waiting for {status.value}"
            )
            time.sleep(poll_interval)

    # =========================================================================
    # Network Volume Operations
    # =========================================================================

    def create_volume(
        self,
        name: str,
        size_gb: int,
        datacenter: str,
    ) -> VolumeInfo:
        """Create a network volume.

        Args:
            name: Volume name.
            size_gb: Size in GB.
            datacenter: Datacenter ID (e.g., "EU-RO-1").

        Returns:
            VolumeInfo for the created volume.

        Raises:
            RunPodError: If creation fails.
        """
        try:
            logger.info(f"Creating volume '{name}' ({size_gb}GB) in {datacenter}")

            # Use GraphQL directly since SDK doesn't have volume functions
            from runpod.api.graphql import run_graphql_query

            query = """
            mutation createNetworkVolume($input: CreateNetworkVolumeInput!) {
                createNetworkVolume(input: $input) {
                    id
                    name
                    size
                    dataCenterId
                }
            }
            """
            variables = {
                "input": {
                    "name": name,
                    "size": size_gb,
                    "dataCenterId": datacenter,
                }
            }

            # run_graphql_query expects the query string with variables embedded
            # We need to use the raw API endpoint instead
            import requests

            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": query, "variables": variables},
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            volume_data = data.get("data", {}).get("createNetworkVolume")
            if not volume_data or "id" not in volume_data:
                raise RunPodError(f"Invalid response from RunPod: {data}")

            return VolumeInfo.from_api(volume_data)

        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to create volume: {e}") from e

    def get_volume(self, volume_id: str) -> VolumeInfo:
        """Get network volume information.

        Args:
            volume_id: Volume ID.

        Returns:
            VolumeInfo for the volume.

        Raises:
            RunPodError: If volume not found.
        """
        # RunPod doesn't have a direct get_volume endpoint
        # We need to list and filter
        volumes = self.list_volumes()
        for vol in volumes:
            if vol.id == volume_id:
                return vol

        raise RunPodError(f"Volume not found: {volume_id}")

    def list_volumes(self) -> List[VolumeInfo]:
        """List all network volumes.

        Returns:
            List of VolumeInfo objects.
        """
        try:
            # Use GraphQL directly since SDK doesn't have volume functions
            import requests

            query = """
            query {
                myself {
                    networkVolumes {
                        id
                        name
                        size
                        dataCenterId
                    }
                }
            }
            """

            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": query},
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            volumes = data.get("data", {}).get("myself", {}).get("networkVolumes", [])
            if not volumes:
                return []

            return [VolumeInfo.from_api(vol) for vol in volumes]

        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to list volumes: {e}") from e

    def delete_volume(self, volume_id: str) -> bool:
        """Delete a network volume.

        Note: Volume must not be in use by any pods.

        Args:
            volume_id: Volume ID.

        Returns:
            True if deleted successfully.

        Raises:
            RunPodError: If deletion fails.
        """
        try:
            # Check if volume is in use
            volume = self.get_volume(volume_id)
            if volume.used_by_pods:
                raise RunPodError(
                    f"Volume {volume_id} is in use by pods: {volume.used_by_pods}"
                )

            logger.info(f"Deleting volume {volume_id}")

            # Use GraphQL directly since SDK doesn't have volume functions
            import requests

            query = """
            mutation deleteNetworkVolume($id: String!) {
                deleteNetworkVolume(input: { id: $id })
            }
            """

            response = requests.post(
                "https://api.runpod.io/graphql",
                json={"query": query, "variables": {"id": volume_id}},
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                raise RunPodError(f"GraphQL error: {data['errors']}")

            return True

        except Exception as e:
            if isinstance(e, RunPodError):
                raise
            raise RunPodError(f"Failed to delete volume {volume_id}: {e}") from e
