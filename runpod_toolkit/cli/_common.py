"""Common helpers for CLI modules.

Shared functions and constants used across multiple CLI command modules.
"""

import functools
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from runpod_toolkit.config import (
    CloudConfig,
    ConfigurationError,
    RunPodS3Config,
)
from runpod_toolkit.storage.b2_client import B2Client
from runpod_toolkit.storage.runpod_s3_client import RunPodS3Client

# Shared console instance
console = Console()

# Project root for relative paths
project_root = Path(__file__).parent.parent.parent

# Status color mapping used across multiple commands
STATUS_COLORS = {
    "running": "blue",
    "completed": "green",
    "success": "green",
    "failed": "red",
    "terminated": "yellow",
    "launching": "blue",
    "pending": "dim",
    "available": "green",
    "exited": "yellow",
}


def get_status_color(status: str) -> str:
    """Get Rich color for a status string.

    Args:
        status: Status string (case-insensitive).

    Returns:
        Rich color name for the status.
    """
    return STATUS_COLORS.get(status.lower(), "yellow")


def handle_cloud_errors(func: Callable) -> Callable:
    """Decorator for standard cloud command error handling.

    Catches exceptions, prints red error message, and exits with code 1.
    Commands needing custom error handling (e.g., cleanup in finally)
    should not use this decorator.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise  # Re-raise Exit to preserve exit codes
        except typer.Abort:
            raise  # Re-raise Abort for user cancellation
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    return wrapper


def require_ssh(pod, console: Console) -> None:
    """Validate pod has SSH access available.

    Args:
        pod: PodInfo object to check.
        console: Console for output.

    Raises:
        typer.Exit: If SSH is not available.
    """
    if not pod.ssh_host or not pod.ssh_port:
        console.print("[red]Pod does not have SSH access available[/red]")
        console.print(f"Pod status: {pod.status.value}")
        raise typer.Exit(1)


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "1.5 GB", "256 MB").
    """
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def create_progress_context(console: Console, description: str = "Processing..."):
    """Create a standard progress bar context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


def get_cloud_config() -> CloudConfig:
    """Load cloud configuration with error handling.

    Returns:
        CloudConfig instance.

    Raises:
        typer.Exit: If configuration loading fails.
    """
    try:
        return CloudConfig.load()
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(1)


def get_b2_client() -> B2Client:
    """Create B2 client with error handling.

    Returns:
        B2Client instance.

    Raises:
        typer.Exit: If client creation fails.
    """
    config = get_cloud_config()
    return B2Client(config.b2)


def require_runpod():
    """Check if runpod SDK is available.

    Returns:
        RunPodClient class.

    Raises:
        typer.Exit: If RunPod SDK not installed.
    """
    try:
        from runpod_toolkit.compute import RunPodClient

        return RunPodClient
    except ImportError:
        console.print(
            "[red]RunPod SDK not installed. Install with: pip install runpod[/red]"
        )
        raise typer.Exit(1)


def get_runpod_client():
    """Create RunPod client.

    Returns:
        RunPodClient instance.

    Raises:
        typer.Exit: If client creation fails.
    """
    RunPodClient = require_runpod()
    config = get_cloud_config()
    return RunPodClient(config.runpod)


def get_runpod_s3_config() -> RunPodS3Config:
    """Load RunPod S3 configuration with error handling.

    Returns:
        RunPodS3Config instance.

    Raises:
        typer.Exit: If configuration loading fails.
    """
    try:
        return RunPodS3Config.from_env()
    except ConfigurationError as e:
        console.print(f"[red]RunPod S3 Configuration Error:[/red] {e}")
        console.print(
            "[dim]Create ~/.secrets/runpod_s3.env with "
            "RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY[/dim]"
        )
        console.print(
            "[dim]Get keys at: RunPod Console → Settings → S3 API Keys[/dim]"
        )
        raise typer.Exit(1)


def get_runpod_s3_client(datacenter: str) -> RunPodS3Client:
    """Create RunPod S3 client for datacenter.

    Args:
        datacenter: Datacenter ID (e.g., "EU-RO-1").

    Returns:
        RunPodS3Client instance.

    Raises:
        typer.Exit: If client creation fails.
    """
    config = get_runpod_s3_config()
    return RunPodS3Client(config, datacenter)


def show_pod_logs(
    pod,
    log_path: str | None = None,
    tail: int = 50,
    follow: bool = False,
    search_paths: list[str] | None = None,
) -> None:
    """Show logs from a pod via SSH.

    Args:
        pod: PodInfo object with SSH details.
        log_path: Explicit log file path. If None, auto-discovers.
        tail: Number of lines to show (ignored if follow=True).
        follow: If True, follow log output like tail -f.
        search_paths: Paths to check before falling back to discovery.
            If None, searches /workspace/outputs.

    Raises:
        typer.Exit: On any error.
    """
    import os
    from runpod_toolkit.compute import PodManager

    require_ssh(pod, console)

    config = get_cloud_config()
    manager = PodManager(config)

    # Discover log file if not specified
    if not log_path:
        # Try explicit search paths first
        if search_paths:
            for candidate in search_paths:
                try:
                    result = manager.run_command_on_pod(
                        pod,
                        f"test -f {candidate} && echo EXISTS",
                        timeout=10,
                    )
                    if "EXISTS" in result.stdout:
                        log_path = candidate
                        break
                except Exception:
                    continue

        # Fall back to discovering most recent log in /workspace/outputs
        if not log_path:
            search_dir = "/workspace/outputs" if not search_paths else "/workspace"
            discover_cmd = (
                f"find {search_dir} -name '*.log' -type f -printf '%T@ %p\\n' 2>/dev/null | "
                "sort -rn | head -1 | cut -d' ' -f2"
            )
            try:
                result = manager.run_command_on_pod(pod, discover_cmd, timeout=30)
                log_path = result.stdout.strip()
            except Exception as e:
                console.print(f"[red]Error discovering logs: {e}[/red]")
                raise typer.Exit(1)

        if not log_path:
            console.print("[yellow]No log files found on pod[/yellow]")
            console.print("Specify a path with --path or check if the process has started")
            raise typer.Exit(1)

    console.print(f"[dim]Log file: {log_path}[/dim]\n")

    if follow:
        console.print(f"Following [cyan]{log_path}[/cyan] (Ctrl+C to stop)...\n")

        # Use RunPod SSH key if available
        from pathlib import Path
        runpod_key = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"

        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if runpod_key.exists():
            ssh_cmd.extend(["-i", str(runpod_key)])
        ssh_cmd.extend([
            "-p", str(pod.ssh_port),
            f"root@{pod.ssh_host}",
            f"tail -f {log_path}"
        ])
        try:
            os.execvp("ssh", ssh_cmd)
        except FileNotFoundError:
            console.print("[red]SSH client not found[/red]")
            raise typer.Exit(1)
    else:
        try:
            tail_cmd = f"tail -n {tail} {log_path}"
            result = manager.run_command_on_pod(pod, tail_cmd, timeout=30)

            if result.exit_code != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                raise typer.Exit(1)

            console.print(result.stdout)

        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
