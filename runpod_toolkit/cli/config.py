"""Configuration management commands."""

import typer
from rich.console import Console

from runpod_toolkit.config import (
    B2Config,
    ConfigurationError,
    validate_b2_config,
)
from runpod_toolkit.cli._common import get_cloud_config

app = typer.Typer(help="Configuration management", no_args_is_help=True)
console = Console()


@app.command("test")
def config_test():
    """Test B2 and RunPod connectivity."""
    console.print("[bold]Testing cloud service connectivity...[/bold]\n")

    # Test B2
    console.print("[cyan]B2 Storage:[/cyan]")
    try:
        b2_config = B2Config.from_env()
        console.print(f"  Bucket: {b2_config.bucket}")
        console.print(f"  Region: {b2_config.region}")

        # Validate connection
        with console.status("  Connecting to B2..."):
            validate_b2_config(b2_config)
        console.print("  [green]Connected successfully[/green]")

    except ConfigurationError as e:
        console.print(f"  [red]Error: {e}[/red]")

    console.print()

    # Test RunPod (if available)
    console.print("[cyan]RunPod:[/cyan]")
    try:
        from runpod_toolkit.config import RunPodConfig, validate_runpod_config

        runpod_config = RunPodConfig.from_env()
        console.print(f"  Default GPU: {runpod_config.default_gpu_type}")
        console.print(f"  Datacenter: {runpod_config.default_datacenter}")

        with console.status("  Connecting to RunPod..."):
            validate_runpod_config(runpod_config)
        console.print("  [green]Connected successfully[/green]")

    except ConfigurationError as e:
        console.print(f"  [yellow]Not configured: {e}[/yellow]")
    except ImportError:
        console.print("  [yellow]RunPod SDK not installed (pip install runpod)[/yellow]")


@app.command("show")
def config_show():
    """Show current configuration (credentials redacted)."""
    console.print("[bold]Cloud Configuration[/bold]\n")

    try:
        config = get_cloud_config()
        redacted = config.to_redacted_dict()

        # B2 section
        console.print("[cyan]B2 Storage:[/cyan]")
        b2 = redacted.get("b2", {})
        for key, value in b2.items():
            console.print(f"  {key}: {value}")

        console.print()

        # RunPod section
        console.print("[cyan]RunPod:[/cyan]")
        runpod = redacted.get("runpod", {})
        for key, value in runpod.items():
            console.print(f"  {key}: {value}")

        console.print()

        # Paths section
        console.print("[cyan]Paths:[/cyan]")
        console.print(f"  pod_workspace: {config.pod_workspace}")
        console.print(f"  pod_code_dir: {config.pod_code_dir}")

    except ConfigurationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
