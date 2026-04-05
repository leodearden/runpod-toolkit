"""Pod lifecycle management.

Higher-level abstraction over RunPodClient and SSHManager for
complete pod lifecycle including creation, setup, execution, and cleanup.

Example:
    >>> from runpod_toolkit.compute import PodManager
    >>> from runpod_toolkit.config import CloudConfig
    >>>
    >>> config = CloudConfig.load()
    >>> manager = PodManager(config)
    >>>
    >>> # Create pod with volume
    >>> pod = manager.create_experiment_pod(
    ...     name="experiment-1",
    ...     volume_id="vol_xxx",
    ...     gpu_type="NVIDIA RTX A5000",
    ... )
    >>>
    >>> # Run setup and commands
    >>> manager.setup_pod_environment(pod)
    >>> result = manager.run_command_on_pod(pod, "python train.py")
    >>>
    >>> # Cleanup
    >>> manager.terminate_pod(pod)
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from runpod_toolkit.config import CloudConfig, RunPodConfig
from runpod_toolkit.compute.runpod_client import (
    RunPodClient,
    RunPodError,
    PodInfo,
    PodStatus,
)
from runpod_toolkit.compute.ssh import (
    SSHManager,
    SSHConnection,
    SSHError,
    CommandResult,
)

logger = logging.getLogger(__name__)


class PodManagerError(Exception):
    """Pod management error."""
    pass


def code_dir_for_tag(tag: str) -> str:
    """Return a per-experiment code directory path on a pod.

    Isolates concurrent experiments so push_code() for one experiment
    doesn't destroy another experiment's working directory.
    """
    return f"/root/code-{tag}"


@dataclass
class SetupResult:
    """Result of pod environment setup."""
    success: bool
    setup_time_seconds: float
    steps_completed: List[str]
    error: Optional[str] = None


class PodManager:
    """Manages pod lifecycle for experiments.

    Combines RunPodClient for pod/volume operations with SSHManager
    for remote execution. Provides:

    - Pod creation with network volume
    - Pod environment setup (clone repo, install deps)
    - Remote command execution
    - Result collection and pod termination
    """

    # Default setup script template
    DEFAULT_SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== Pod Setup Starting ==="

# Increase file descriptor limit for large datasets (556+ pairs need many open files)
echo "Setting ulimit..."
ulimit -n 65535 2>/dev/null || echo "Warning: Could not increase file descriptor limit"

# Verify key packages are available (fail fast if container image is incomplete)
echo "Verifying container image packages..."
python -c "import torch; import stable_baselines3; import optuna; print('Core packages OK')" || {
    echo "ERROR: Required packages missing from container image - rebuild image"
    exit 1
}

echo "=== Pod Setup Complete ==="
"""

    def __init__(
        self,
        config: CloudConfig,
        ssh_manager: Optional[SSHManager] = None,
    ):
        """Initialize pod manager.

        Args:
            config: Cloud configuration.
            ssh_manager: Optional SSHManager (created if not provided).
        """
        self.config = config
        self.runpod = RunPodClient(config.runpod)

        # Use RunPod SSH key if available
        if ssh_manager:
            self.ssh = ssh_manager
        else:
            runpod_key = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
            if runpod_key.exists():
                self.ssh = SSHManager(private_key_path=runpod_key)
            else:
                self.ssh = SSHManager()

    def create_experiment_pod(
        self,
        name: str,
        volume_id: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_count: int = 1,
        env_vars: Optional[Dict[str, str]] = None,
        wait_for_ready: bool = True,
        timeout: int = 600,
        image: Optional[str] = None,
        min_memory_gb: Optional[int] = None,
        max_retries: int = 2,
    ) -> PodInfo:
        """Create a pod for running experiments.

        Args:
            name: Pod name.
            volume_id: Optional network volume ID.
            gpu_type: GPU type (uses config default if not specified).
            gpu_count: Number of GPUs.
            env_vars: Environment variables for the pod.
            wait_for_ready: Wait for pod to be running and SSH available.
            timeout: Timeout for waiting in seconds.
            image: Docker image (uses default if not specified).
            min_memory_gb: Minimum system RAM in GB.
            max_retries: Max retries on stuck pods (cleanup and retry).

        Returns:
            PodInfo for the created pod.

        Raises:
            PodManagerError: If pod creation fails after all retries.
        """
        gpu_type = gpu_type or self.config.runpod.default_gpu_type
        datacenter = self.config.runpod.default_datacenter

        last_error = None

        for attempt in range(max_retries + 1):
            pod = None
            try:
                if attempt > 0:
                    logger.warning(f"Retry {attempt}/{max_retries} for pod creation")

                logger.info(f"Creating experiment pod '{name}'")

                pod = self.runpod.create_pod(
                    name=name,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    network_volume_id=volume_id,
                    env_vars=env_vars,
                    datacenter=datacenter,
                    image=image,
                    min_memory_gb=min_memory_gb,
                )

                logger.info(f"Pod created: {pod.id}")

                if wait_for_ready:
                    logger.info("Waiting for pod to be ready...")
                    pod = self.runpod.wait_for_pod(
                        pod.id,
                        PodStatus.RUNNING,
                        timeout=timeout,
                    )

                    # Wait for SSH to be available
                    pod = self._wait_for_ssh(pod, timeout=480)

                    # Persist RunPod env vars for SSH sessions
                    self._persist_runpod_env(pod)

                return pod

            except (RunPodError, PodManagerError) as e:
                last_error = e

                # Clean up stuck pod before retrying
                if pod:
                    logger.warning(f"Pod {pod.id} failed to become ready, terminating...")
                    try:
                        self.runpod.terminate_pod(pod.id)
                        logger.info(f"Terminated stuck pod {pod.id}")
                    except Exception as cleanup_err:
                        logger.error(f"Failed to terminate stuck pod: {cleanup_err}")

                if attempt < max_retries:
                    continue

                raise PodManagerError(
                    f"Failed to create pod after {max_retries + 1} attempts: {last_error}"
                ) from e

        # Should not reach here, but satisfy type checker
        raise PodManagerError(f"Failed to create pod: {last_error}")

    def create_cpu_experiment_pod(
        self,
        name: str,
        volume_id: Optional[str] = None,
        vcpu_count: int = 4,
        memory_gb: int = 16,
        env_vars: Optional[Dict[str, str]] = None,
        wait_for_ready: bool = True,
        timeout: int = 600,
        image: Optional[str] = None,
        cpu_flavor: Optional[str] = None,
        max_retries: int = 2,
    ) -> PodInfo:
        """Create a CPU-only pod for running experiments.

        CPU pods are cheaper and suitable for stages that don't require GPU,
        such as Optuna hyperparameter tuning and final PPO training.

        Args:
            name: Pod name.
            volume_id: Optional network volume ID.
            vcpu_count: Number of vCPUs.
            memory_gb: RAM in GB.
            env_vars: Environment variables for the pod.
            wait_for_ready: Wait for pod to be running and SSH available.
            timeout: Timeout for waiting in seconds.
            image: Docker image (uses default CPU image if not specified).
            cpu_flavor: CPU flavor (cpu3c, cpu3g, cpu3m, cpu5c, cpu5g, cpu5m).
                        If None, auto-selects based on memory_gb/vcpu_count ratio.
            max_retries: Max retries on stuck pods (cleanup and retry).

        Returns:
            PodInfo for the created pod.

        Raises:
            PodManagerError: If pod creation fails after all retries.
        """
        # Resolve datacenter from volume (must match) or fall back to default
        datacenter = self.config.runpod.default_datacenter
        if volume_id:
            try:
                volume = self.runpod.get_volume(volume_id)
                if volume and volume.datacenter:
                    datacenter = volume.datacenter
                    logger.info(f"Using volume datacenter: {datacenter}")
            except Exception as e:
                logger.warning(f"Failed to resolve volume datacenter: {e}")

        last_error = None

        for attempt in range(max_retries + 1):
            pod = None
            try:
                if attempt > 0:
                    logger.warning(f"Retry {attempt}/{max_retries} for CPU pod creation")

                logger.info(f"Creating CPU experiment pod '{name}' ({vcpu_count} vCPU, {memory_gb}GB)")

                pod = self.runpod.create_cpu_pod(
                    name=name,
                    vcpu_count=vcpu_count,
                    memory_gb=memory_gb,
                    network_volume_id=volume_id,
                    env_vars=env_vars,
                    datacenter=datacenter,
                    image=image,
                    cpu_flavor=cpu_flavor,
                )

                logger.info(f"CPU pod created: {pod.id}")

                if wait_for_ready:
                    logger.info("Waiting for CPU pod to be ready...")
                    pod = self.runpod.wait_for_pod(
                        pod.id,
                        PodStatus.RUNNING,
                        timeout=timeout,
                    )

                    # Wait for SSH to be available
                    pod = self._wait_for_ssh(pod, timeout=300)

                    # Persist RunPod env vars for SSH sessions
                    self._persist_runpod_env(pod)

                return pod

            except (RunPodError, PodManagerError) as e:
                last_error = e

                # Clean up stuck pod before retrying
                if pod:
                    logger.warning(f"CPU pod {pod.id} failed to become ready, terminating...")
                    try:
                        self.runpod.terminate_pod(pod.id)
                        logger.info(f"Terminated stuck CPU pod {pod.id}")
                    except Exception as cleanup_err:
                        logger.error(f"Failed to terminate stuck CPU pod: {cleanup_err}")

                if attempt < max_retries:
                    continue

                raise PodManagerError(
                    f"Failed to create CPU pod after {max_retries + 1} attempts: {last_error}"
                ) from e

        # Should not reach here, but satisfy type checker
        raise PodManagerError(f"Failed to create CPU pod: {last_error}")

    def _wait_for_ssh(
        self,
        pod: PodInfo,
        timeout: int = 120,
        poll_interval: int = 10,
    ) -> PodInfo:
        """Wait for SSH to be available on pod.

        Args:
            pod: Pod to wait for.
            timeout: Timeout in seconds.
            poll_interval: Polling interval.

        Returns:
            Updated PodInfo with SSH details.

        Raises:
            PodManagerError: If SSH not available within timeout.
        """
        start_time = time.time()

        while True:
            # Refresh pod info
            pod = self.runpod.get_pod(pod.id)

            if pod.ssh_host and pod.ssh_port:
                # Try to connect
                try:
                    with self.ssh.connect(
                        pod.ssh_host,
                        pod.ssh_port,
                        timeout=10,
                    ) as conn:
                        # Test connection
                        result = self.ssh.run_command(conn, "echo ok", timeout=10)
                        if result.success:
                            logger.info(f"SSH available at {pod.ssh_host}:{pod.ssh_port}")
                            return pod

                except SSHError as e:
                    logger.debug(f"SSH not ready: {e}")

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise PodManagerError(
                    f"SSH not available after {timeout}s. "
                    f"Host: {pod.ssh_host}, Port: {pod.ssh_port}"
                )

            logger.debug("SSH not ready, waiting...")
            time.sleep(poll_interval)

    def _persist_runpod_env(self, pod: PodInfo) -> bool:
        """Persist RunPod environment variables to /etc/runpod.env.

        SSH sessions don't inherit the container's environment variables, but
        scripts need RUNPOD_POD_ID and RUNPOD_API_KEY for self-termination.
        This method reads these from PID 1's environment (where RunPod injects
        them) and writes them to a file that can be sourced by SSH sessions.

        Args:
            pod: Pod to configure.

        Returns:
            True if successful, False otherwise.
        """
        if not pod.ssh_host or not pod.ssh_port:
            logger.warning("Cannot persist runpod env: no SSH info")
            return False

        try:
            with self.ssh.connect(pod.ssh_host, pod.ssh_port, timeout=30) as conn:
                # Read RUNPOD_POD_ID and RUNPOD_API_KEY from PID 1's environment
                # These are injected by RunPod at container startup
                script = '''
                    # Extract env vars from PID 1's environment
                    POD_ID=$(cat /proc/1/environ 2>/dev/null | tr '\\0' '\\n' | grep '^RUNPOD_POD_ID=' | cut -d= -f2-)
                    API_KEY=$(cat /proc/1/environ 2>/dev/null | tr '\\0' '\\n' | grep '^RUNPOD_API_KEY=' | cut -d= -f2-)

                    # Write to /etc/runpod.env
                    echo "RUNPOD_POD_ID=$POD_ID" > /etc/runpod.env
                    echo "RUNPOD_API_KEY=$API_KEY" >> /etc/runpod.env
                    chmod 600 /etc/runpod.env

                    # Verify
                    if [ -n "$POD_ID" ] && [ -n "$API_KEY" ]; then
                        echo "OK: Persisted RUNPOD_POD_ID=$POD_ID"
                    else
                        echo "WARN: Missing env vars (POD_ID=$POD_ID, API_KEY set: $([ -n "$API_KEY" ] && echo yes || echo no))"
                    fi
                '''

                result = self.ssh.run_command(conn, script, timeout=30)

                if result.success:
                    logger.debug(f"Persisted runpod env: {result.stdout.strip()}")
                    return True
                else:
                    logger.warning(f"Failed to persist runpod env: {result.stderr}")
                    return False

        except SSHError as e:
            logger.warning(f"Failed to persist runpod env: {e}")
            return False

    def setup_pod_environment(
        self,
        pod: PodInfo,
        setup_script: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SetupResult:
        """Set up pod environment for experiments.

        Executes setup script on pod to install dependencies
        and prepare for training.

        Args:
            pod: Pod to set up.
            setup_script: Custom setup script (uses default if not provided).
            progress_callback: Optional callback for progress updates.

        Returns:
            SetupResult with status and timing.
        """
        setup_script = setup_script or self.DEFAULT_SETUP_SCRIPT
        start_time = time.time()
        steps_completed = []

        def progress(msg: str):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        try:
            if not pod.ssh_host or not pod.ssh_port:
                return SetupResult(
                    success=False,
                    setup_time_seconds=time.time() - start_time,
                    steps_completed=steps_completed,
                    error="Pod has no SSH connection info",
                )

            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                # Upload and run setup script
                progress("Uploading setup script...")
                setup_path = "/tmp/pod_setup.sh"
                self.ssh.upload_bytes(
                    conn,
                    setup_script.encode("utf-8"),
                    setup_path,
                )
                steps_completed.append("upload_script")

                progress("Making script executable...")
                self.ssh.run_command(conn, f"chmod +x {setup_path}")
                steps_completed.append("chmod")

                progress("Running setup script...")
                result = self.ssh.run_command(
                    conn,
                    f"bash {setup_path}",
                    timeout=600,  # 10 min for setup
                )

                if not result.success:
                    # Check if pip specifically failed
                    pip_failed = (
                        "pip" in result.stderr.lower()
                        or "Could not install" in result.stderr
                    )
                    error_msg = "Setup script failed"
                    if pip_failed:
                        error_msg = "Pip installation failed during setup"
                        logger.warning(
                            "Pip failed during pod setup - check package versions in container image"
                        )
                    return SetupResult(
                        success=False,
                        setup_time_seconds=time.time() - start_time,
                        steps_completed=steps_completed,
                        error=f"{error_msg}: {result.stderr}",
                    )

                steps_completed.append("setup_script")

                # Warn if pip actually installed packages (should be pre-installed in container)
                if "Successfully installed" in result.stdout:
                    logger.warning(
                        "Pod setup installed new packages - consider updating the container image. "
                        "Output: %s",
                        result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                    )

                progress("Setup complete")

            return SetupResult(
                success=True,
                setup_time_seconds=time.time() - start_time,
                steps_completed=steps_completed,
            )

        except SSHError as e:
            return SetupResult(
                success=False,
                setup_time_seconds=time.time() - start_time,
                steps_completed=steps_completed,
                error=str(e),
            )

    def run_command_on_pod(
        self,
        pod: PodInfo,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
    ) -> CommandResult:
        """Execute command on pod.

        Args:
            pod: Pod to run command on.
            command: Command to execute.
            timeout: Command timeout in seconds.
            working_dir: Working directory for command.

        Returns:
            CommandResult with exit code, stdout, stderr.

        Raises:
            PodManagerError: If execution fails.
        """
        if not pod.ssh_host or not pod.ssh_port:
            raise PodManagerError("Pod has no SSH connection info")

        # Wrap command with working directory if specified
        if working_dir:
            command = f"cd {working_dir} && {command}"

        try:
            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                return self.ssh.run_command(conn, command, timeout=timeout)

        except SSHError as e:
            raise PodManagerError(f"Command execution failed: {e}") from e

    def upload_to_pod(
        self,
        pod: PodInfo,
        local_path: Path,
        remote_path: str,
    ) -> None:
        """Upload file to pod.

        Args:
            pod: Target pod.
            local_path: Local file path.
            remote_path: Remote destination path.

        Raises:
            PodManagerError: If upload fails.
        """
        if not pod.ssh_host or not pod.ssh_port:
            raise PodManagerError("Pod has no SSH connection info")

        try:
            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                self.ssh.upload_file(conn, local_path, remote_path)

        except SSHError as e:
            raise PodManagerError(f"Upload failed: {e}") from e

    def download_from_pod(
        self,
        pod: PodInfo,
        remote_path: str,
        local_path: Path,
    ) -> Path:
        """Download file from pod.

        Args:
            pod: Source pod.
            remote_path: Remote file path.
            local_path: Local destination path.

        Returns:
            Path to downloaded file.

        Raises:
            PodManagerError: If download fails.
        """
        if not pod.ssh_host or not pod.ssh_port:
            raise PodManagerError("Pod has no SSH connection info")

        try:
            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                return self.ssh.download_file(conn, remote_path, local_path)

        except SSHError as e:
            raise PodManagerError(f"Download failed: {e}") from e

    def terminate_pod(
        self,
        pod: PodInfo,
        collect_results: bool = False,
        results_dir: Optional[Path] = None,
    ) -> bool:
        """Terminate pod and optionally collect results.

        Args:
            pod: Pod to terminate.
            collect_results: Download results before terminating.
            results_dir: Local directory for results (required if collect_results).

        Returns:
            True if terminated successfully.

        Raises:
            PodManagerError: If termination fails.
        """
        if collect_results:
            if not results_dir:
                raise PodManagerError("results_dir required when collect_results=True")

            try:
                self._collect_results(pod, results_dir)
            except Exception as e:
                logger.error(f"Failed to collect results: {e}")
                # Continue with termination even if result collection fails

        try:
            logger.info(f"Terminating pod {pod.id}")
            return self.runpod.terminate_pod(pod.id)

        except RunPodError as e:
            raise PodManagerError(f"Failed to terminate pod: {e}") from e

    def _collect_results(
        self,
        pod: PodInfo,
        results_dir: Path,
    ) -> None:
        """Collect results from pod before termination.

        Args:
            pod: Pod to collect from.
            results_dir: Local directory for results.
        """
        results_dir.mkdir(parents=True, exist_ok=True)

        with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
            # Check for common result locations
            result_paths = [
                "/workspace/outputs",
                "/workspace/results",
                "/workspace/logs",
            ]

            for remote_path in result_paths:
                if self.ssh.file_exists(conn, remote_path):
                    logger.info(f"Collecting {remote_path}")

                    # Create tarball of results
                    tar_name = remote_path.split("/")[-1] + ".tar.gz"
                    tar_path = f"/tmp/{tar_name}"

                    result = self.ssh.run_command(
                        conn,
                        f"tar -czf {tar_path} -C /workspace {remote_path.split('/')[-1]}",
                    )

                    if result.success:
                        local_tar = results_dir / tar_name
                        self.ssh.download_file(conn, tar_path, local_tar)
                        logger.info(f"Downloaded {local_tar}")

    def push_code(
        self,
        pod: PodInfo,
        local_repo_path: Optional[Path] = None,
        branch: str = "main",  # unused, kept for API compatibility
        remote_name: str = "pod",  # unused, kept for API compatibility
        code_dir: str = "/root/code",
    ) -> bool:
        """Push code to pod via git archive + upload.

        Creates a tarball of the current HEAD, uploads via SSH, and extracts
        to the specified directory on the pod. This avoids shared volume
        corruption issues that occurred with git push to a bare repo on the
        network volume.

        Args:
            pod: Target pod with SSH info.
            local_repo_path: Path to local git repo (default: current directory).
            branch: Unused, kept for API compatibility.
            remote_name: Unused, kept for API compatibility.
            code_dir: Remote directory to extract code into (default: /root/code).
                Use code_dir_for_tag() for per-experiment isolation.

        Returns:
            True if push succeeded.

        Raises:
            PodManagerError: If push fails.
        """
        import subprocess
        import tempfile

        if not pod.ssh_host or not pod.ssh_port:
            raise PodManagerError("Pod has no SSH connection info")

        local_repo_path = local_repo_path or Path.cwd()

        # Create git archive of current HEAD
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as f:
            archive_path = f.name

        try:
            # Get local HEAD for logging
            local_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=local_repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            local_head_short = local_head[:7]
            logger.info(f"Creating code archive (HEAD: {local_head_short})...")

            subprocess.run(
                ["git", "archive", "HEAD", "--format=tar.gz", "-o", archive_path],
                cwd=local_repo_path,
                check=True,
            )

            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                # Upload archive
                remote_archive = "/tmp/code.tar.gz"
                logger.info(f"Uploading code to pod {pod.id}...")
                self.ssh.upload_file(conn, archive_path, remote_archive)

                # Extract to /root/code (rm -rf first in case pod is reused)
                logger.info("Extracting code on pod...")
                result = self.ssh.run_command(
                    conn,
                    f"rm -rf {code_dir} && "
                    f"mkdir -p {code_dir} && "
                    f"tar xzf {remote_archive} -C {code_dir} && "
                    f"rm -f {remote_archive} && "
                    f"cd {code_dir} && git init -q",
                    timeout=120,
                )

                if not result.success:
                    raise PodManagerError(f"Code extraction failed: {result.stderr}")

                # Verify extraction (check for either pyproject.toml or setup.py)
                verify = self.ssh.run_command(
                    conn,
                    f"(test -f {code_dir}/pyproject.toml || test -f {code_dir}/setup.py) && echo OK",
                    timeout=10,
                )
                if "OK" not in verify.stdout:
                    raise PodManagerError("Code extraction verification failed - no pyproject.toml or setup.py found")

                logger.info(f"Code deployed to {code_dir} on pod {pod.id}")
                return True

        except subprocess.CalledProcessError as e:
            raise PodManagerError(f"Git archive failed: {e}") from e

        finally:
            # Clean up local archive
            Path(archive_path).unlink(missing_ok=True)

    def has_running_experiments(self, pod: PodInfo) -> bool:
        """Check if pod has running training processes.

        Used to determine if it's safe to terminate a pod when an experiment
        finishes. Returns True (assume running) if the check fails, to avoid
        accidentally terminating a pod with active experiments.

        Args:
            pod: Pod to check.

        Returns:
            True if training processes are running (or check failed), False if safe to terminate.
        """
        if not pod.ssh_host or not pod.ssh_port:
            logger.warning("Pod has no SSH info, cannot check for running experiments (pod NOT TERMINATED)")
            return True

        try:
            with self.ssh.connect(pod.ssh_host, pod.ssh_port) as conn:
                result = self.ssh.run_command(
                    conn,
                    "pgrep -f 'python.*train' | wc -l",
                    timeout=10,
                )
                count = int(result.stdout.strip())
                return count > 0
        except Exception as e:
            logger.warning(f"Could not check for running experiments (pod NOT TERMINATED): {e}")
            return True  # Assume running if we can't check (safe default)
