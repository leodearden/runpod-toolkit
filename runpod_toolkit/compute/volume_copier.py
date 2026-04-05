"""Volume copy operations between datacenters.

Orchestrates copying network volumes between RunPod datacenters using
CPU pods and rsync over SSH.

Example:
    >>> from runpod_toolkit.compute import RunPodClient, VolumeCopier
    >>> from runpod_toolkit.config import CloudConfig
    >>>
    >>> config = CloudConfig.load()
    >>> client = RunPodClient(config.runpod)
    >>> copier = VolumeCopier(client)
    >>>
    >>> result = copier.copy(
    ...     source_volume_id="vol_xxx",
    ...     dest_datacenter="EU-CZ-1",
    ...     progress_callback=lambda msg: print(msg),
    ... )
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from runpod_toolkit.compute.runpod_client import (
    RunPodClient,
    PodInfo,
    VolumeInfo,
    PodStatus,
)

logger = logging.getLogger(__name__)


class VolumeCopyError(Exception):
    """Error during volume copy operation."""

    pass


@dataclass
class VolumeCopyResult:
    """Result of a volume copy operation."""

    success: bool
    source_volume: VolumeInfo
    dest_volume: VolumeInfo
    error: Optional[str] = None


class VolumeCopier:
    """Copies network volumes between datacenters using CPU pods and rsync.

    Creates temporary CPU pods in source and destination datacenters,
    sets up SSH between them, and uses rsync to transfer data.
    """

    def __init__(
        self,
        client: RunPodClient,
        ssh_key_path: Optional[Path] = None,
    ):
        """Initialize volume copier.

        Args:
            client: RunPod API client.
            ssh_key_path: Path to SSH private key for connecting to pods.
                         Auto-discovers from ~/.runpod/ssh/ or ~/.ssh/ if not provided.
        """
        self.client = client
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()

    def _find_ssh_key(self) -> Path:
        """Find SSH key for RunPod access.

        Returns:
            Path to SSH private key.

        Raises:
            VolumeCopyError: If no SSH key found.
        """
        # Try RunPod key first
        runpod_key = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
        if runpod_key.exists():
            return runpod_key

        # Fall back to default SSH key
        default_key = Path.home() / ".ssh" / "id_rsa"
        if default_key.exists():
            return default_key

        raise VolumeCopyError(
            "No SSH key found. Please set up SSH keys for RunPod "
            "(~/.runpod/ssh/RunPod-Key-Go or ~/.ssh/id_rsa)"
        )

    def copy(
        self,
        source_volume_id: str,
        dest_datacenter: str,
        dest_volume_id: Optional[str] = None,
        dest_volume_name: Optional[str] = None,
        source_path: str = "/workspace/",
        dest_path: str = "/workspace/",
        vcpu: int = 4,
        memory_gb: int = 8,
        keep_pods: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> VolumeCopyResult:
        """Copy volume to another datacenter.

        Creates CPU pods in both datacenters, sets up SSH between them,
        and uses rsync to transfer data.

        Args:
            source_volume_id: Source volume ID.
            dest_datacenter: Destination datacenter ID (e.g., "EU-CZ-1").
            dest_volume_id: Existing destination volume ID (creates new if None).
            dest_volume_name: Name for new volume (auto-generated if None).
            source_path: Path within source volume to copy.
            dest_path: Path within destination volume to copy to.
            vcpu: vCPUs for transfer pods.
            memory_gb: Memory for transfer pods.
            keep_pods: Keep pods running after copy (for debugging).
            progress_callback: Called with status messages.
            output_callback: Called with rsync output lines.

        Returns:
            VolumeCopyResult with outcome details.

        Raises:
            VolumeCopyError: If copy fails.
        """

        def log_progress(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        def log_output(line: str) -> None:
            if output_callback:
                output_callback(line)

        # Get source volume info
        log_progress("Getting source volume info...")
        try:
            source_volume = self.client.get_volume(source_volume_id)
        except Exception as e:
            raise VolumeCopyError(f"Failed to get source volume: {e}") from e

        log_progress(
            f"Source: {source_volume.name} ({source_volume.size_gb}GB) "
            f"in {source_volume.datacenter}"
        )

        # Create or get destination volume
        if dest_volume_id:
            log_progress("Getting destination volume info...")
            try:
                dest_volume = self.client.get_volume(dest_volume_id)
            except Exception as e:
                raise VolumeCopyError(f"Failed to get destination volume: {e}") from e

            if dest_volume.datacenter != dest_datacenter:
                raise VolumeCopyError(
                    f"Destination volume is in {dest_volume.datacenter}, "
                    f"not {dest_datacenter}"
                )
        else:
            # Create new volume in destination
            dest_name = (
                dest_volume_name
                or f"{source_volume.name}-{dest_datacenter.lower()}"
            )
            log_progress(
                f"Creating destination volume '{dest_name}' "
                f"({source_volume.size_gb}GB) in {dest_datacenter}..."
            )

            try:
                dest_volume = self.client.create_volume(
                    name=dest_name,
                    size_gb=source_volume.size_gb,
                    datacenter=dest_datacenter,
                )
                log_progress(f"Created volume: {dest_volume.id}")
            except Exception as e:
                raise VolumeCopyError(
                    f"Failed to create destination volume: {e}"
                ) from e

        log_progress(
            f"Destination: {dest_volume.name} ({dest_volume.size_gb}GB) "
            f"in {dest_volume.datacenter}"
        )

        source_pod: Optional[PodInfo] = None
        dest_pod: Optional[PodInfo] = None

        try:
            # Create CPU pods in both datacenters
            log_progress(
                f"Creating CPU transfer pods ({vcpu} vCPU, {memory_gb}GB RAM)..."
            )

            log_progress(f"Creating source pod in {source_volume.datacenter}...")
            source_pod = self.client.create_cpu_pod(
                name=f"volume-copy-src-{source_volume_id[:8]}",
                network_volume_id=source_volume_id,
                datacenter=source_volume.datacenter,
                vcpu_count=vcpu,
                memory_gb=memory_gb,
            )
            log_progress(f"Created source pod: {source_pod.id}")

            log_progress(f"Creating destination pod in {dest_volume.datacenter}...")
            dest_pod = self.client.create_cpu_pod(
                name=f"volume-copy-dst-{dest_volume.id[:8]}",
                network_volume_id=dest_volume.id,
                datacenter=dest_volume.datacenter,
                vcpu_count=vcpu,
                memory_gb=memory_gb,
            )
            log_progress(f"Created destination pod: {dest_pod.id}")

            # Wait for pods to be ready with SSH
            log_progress("Waiting for pods to be ready...")

            source_pod = self.client.wait_for_pod(
                source_pod.id, PodStatus.RUNNING, timeout=300, wait_for_ssh=True
            )
            log_progress(
                f"Source pod ready: {source_pod.ssh_host}:{source_pod.ssh_port}"
            )

            dest_pod = self.client.wait_for_pod(
                dest_pod.id, PodStatus.RUNNING, timeout=300, wait_for_ssh=True
            )
            log_progress(
                f"Destination pod ready: {dest_pod.ssh_host}:{dest_pod.ssh_port}"
            )

            # Set up SSH between pods
            log_progress("Setting up SSH between pods...")
            self._setup_pod_ssh(source_pod, dest_pod, log_progress)

            # Run rsync
            log_progress("Starting rsync transfer...")
            log_progress(f"From: {source_path}")
            log_progress(f"To: {dest_pod.ssh_host}:{dest_pod.ssh_port}:{dest_path}")

            self._run_rsync(
                source_pod, dest_pod, source_path, dest_path, log_output
            )

            log_progress("Transfer completed successfully!")

            return VolumeCopyResult(
                success=True,
                source_volume=source_volume,
                dest_volume=dest_volume,
            )

        except Exception as e:
            logger.error(f"Volume copy failed: {e}")
            return VolumeCopyResult(
                success=False,
                source_volume=source_volume,
                dest_volume=dest_volume,
                error=str(e),
            )

        finally:
            # Cleanup pods
            if not keep_pods:
                log_progress("Cleaning up pods...")
                if source_pod:
                    try:
                        self.client.terminate_pod(source_pod.id)
                        log_progress(f"Terminated source pod: {source_pod.id}")
                    except Exception as e:
                        log_progress(f"Warning: Failed to terminate source pod: {e}")
                if dest_pod:
                    try:
                        self.client.terminate_pod(dest_pod.id)
                        log_progress(f"Terminated destination pod: {dest_pod.id}")
                    except Exception as e:
                        log_progress(f"Warning: Failed to terminate dest pod: {e}")
            else:
                log_progress("Keeping pods running (--keep-pods)")
                if source_pod:
                    log_progress(
                        f"Source: ssh root@{source_pod.ssh_host} -p {source_pod.ssh_port}"
                    )
                if dest_pod:
                    log_progress(
                        f"Dest: ssh root@{dest_pod.ssh_host} -p {dest_pod.ssh_port}"
                    )

    def _setup_pod_ssh(
        self,
        source_pod: PodInfo,
        dest_pod: PodInfo,
        log_progress: Callable[[str], None],
    ) -> None:
        """Set up SSH key exchange between pods.

        Generates a temporary key pair on source pod and adds public key
        to destination pod's authorized_keys.

        Args:
            source_pod: Source pod info.
            dest_pod: Destination pod info.
            log_progress: Progress logging callback.

        Raises:
            VolumeCopyError: If SSH setup fails.
        """
        ssh_opts = (
            f"-i {self.ssh_key_path} "
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null"
        )

        # Generate temporary key pair on source pod
        log_progress("Generating temporary SSH key on source pod...")
        keygen_cmd = (
            f"ssh {ssh_opts} -p {source_pod.ssh_port} root@{source_pod.ssh_host} "
            "'ssh-keygen -t ed25519 -f /tmp/transfer_key -N \"\" -q && "
            "cat /tmp/transfer_key.pub'"
        )
        result = subprocess.run(keygen_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise VolumeCopyError(f"Failed to generate SSH key: {result.stderr}")
        pubkey = result.stdout.strip()

        # Add public key to destination pod
        log_progress("Adding key to destination pod...")
        auth_cmd = (
            f"ssh {ssh_opts} -p {dest_pod.ssh_port} root@{dest_pod.ssh_host} "
            f"'mkdir -p ~/.ssh && "
            f'echo "{pubkey}" >> ~/.ssh/authorized_keys && '
            f"chmod 600 ~/.ssh/authorized_keys'"
        )
        result = subprocess.run(auth_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise VolumeCopyError(f"Failed to add SSH key: {result.stderr}")

    def _run_rsync(
        self,
        source_pod: PodInfo,
        dest_pod: PodInfo,
        source_path: str,
        dest_path: str,
        log_output: Callable[[str], None],
    ) -> None:
        """Run rsync between pods.

        Args:
            source_pod: Source pod info.
            dest_pod: Destination pod info.
            source_path: Path within source to copy.
            dest_path: Path within destination.
            log_output: Output line callback.

        Raises:
            VolumeCopyError: If rsync fails.
        """
        ssh_opts = (
            f"-i {self.ssh_key_path} "
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null"
        )

        # Build rsync command to run on source pod
        rsync_cmd = (
            f"rsync -avz --progress "
            f"-e 'ssh -i /tmp/transfer_key "
            f"-o StrictHostKeyChecking=no "
            f"-o UserKnownHostsFile=/dev/null "
            f"-p {dest_pod.ssh_port}' "
            f"{source_path} root@{dest_pod.ssh_host}:{dest_path}"
        )

        # Execute rsync on source pod, streaming output
        full_cmd = (
            f"ssh {ssh_opts} -p {source_pod.ssh_port} "
            f'root@{source_pod.ssh_host} "{rsync_cmd}"'
        )

        logger.debug(f"Running rsync command: {rsync_cmd}")

        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output
        for line in process.stdout:
            log_output(line.rstrip())

        process.wait()

        if process.returncode != 0:
            raise VolumeCopyError(f"rsync failed with exit code {process.returncode}")
