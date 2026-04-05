"""Network volume management commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from runpod_toolkit.cli._common import (
    console,
    get_cloud_config,
    get_runpod_client,
)
from runpod_toolkit.compute import VolumeCopier, VolumeCopyError

app = typer.Typer(help="Network volume management", no_args_is_help=True)
console = Console()


@app.command("create")
def volume_create(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Volume name",
    ),
    size: int = typer.Option(
        100,
        "--size",
        "-s",
        help="Volume size in GB",
    ),
    datacenter: Optional[str] = typer.Option(
        None,
        "--datacenter",
        "-d",
        help="Datacenter ID (default: from config)",
    ),
):
    """Create a new network volume."""
    config = get_cloud_config()
    datacenter = datacenter or config.runpod.default_datacenter

    client = get_runpod_client()

    console.print(f"[bold]Creating volume '{name}' ({size}GB) in {datacenter}...[/bold]")

    try:
        volume = client.create_volume(
            name=name,
            size_gb=size,
            datacenter=datacenter,
        )

        console.print()
        console.print("[green]Volume created successfully![/green]")
        console.print(f"  ID: [cyan]{volume.id}[/cyan]")
        console.print(f"  Name: {volume.name}")
        console.print(f"  Size: {volume.size_gb} GB")
        console.print(f"  Datacenter: {volume.datacenter}")
        console.print(f"  Status: {volume.status.value}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def volume_list():
    """List all network volumes."""
    client = get_runpod_client()

    with console.status("Fetching volumes..."):
        try:
            volumes = client.list_volumes()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    if not volumes:
        console.print("[yellow]No volumes found[/yellow]")
        return

    table = Table(title="Network Volumes")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Datacenter")
    table.add_column("Status")
    table.add_column("In Use")

    for vol in volumes:
        status_color = "green" if vol.status.value == "AVAILABLE" else "yellow"
        in_use = "Yes" if vol.used_by_pods else "No"
        table.add_row(
            vol.id,
            vol.name,
            str(vol.size_gb),
            vol.datacenter,
            f"[{status_color}]{vol.status.value}[/{status_color}]",
            in_use,
        )

    console.print(table)


@app.command("delete")
def volume_delete(
    volume_id: str = typer.Argument(
        ...,
        help="Volume ID to delete",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
):
    """Delete a network volume."""
    client = get_runpod_client()

    # Get volume info first
    try:
        volume = client.get_volume(volume_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"Volume: [cyan]{volume.name}[/cyan] ({volume.size_gb}GB)")

    if volume.used_by_pods:
        console.print(f"[red]Volume is in use by pods: {volume.used_by_pods}[/red]")
        console.print("Terminate pods first before deleting volume.")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm("Are you sure you want to delete this volume?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        client.delete_volume(volume_id)
        console.print("[green]Volume deleted successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("status")
def volume_status(
    volume_id: str = typer.Argument(
        ...,
        help="Volume ID",
    ),
):
    """Show detailed volume status."""
    client = get_runpod_client()

    try:
        volume = client.get_volume(volume_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Volume: {volume.name}[/bold]\n")
    console.print(f"  ID: [cyan]{volume.id}[/cyan]")
    console.print(f"  Size: {volume.size_gb} GB")
    console.print(f"  Datacenter: {volume.datacenter}")
    console.print(f"  Status: {volume.status.value}")
    console.print(f"  Created: {volume.created_at}")

    if volume.used_by_pods:
        console.print(f"  Used by: {', '.join(volume.used_by_pods)}")
    else:
        console.print("  Used by: (none)")


@app.command("copy")
def volume_copy(
    source_volume_id: str = typer.Argument(
        ...,
        help="Source volume ID to copy from",
    ),
    dest_datacenter: str = typer.Argument(
        ...,
        help="Destination datacenter ID (e.g., EU-CZ-1)",
    ),
    dest_volume_id: Optional[str] = typer.Option(
        None,
        "--dest-volume",
        "-d",
        help="Existing destination volume ID (creates new if not specified)",
    ),
    dest_volume_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for new destination volume (default: {source_name}-{dest_dc})",
    ),
    source_path: str = typer.Option(
        "/workspace/",
        "--source-path",
        "-s",
        help="Path within source volume to copy",
    ),
    dest_path: str = typer.Option(
        "/workspace/",
        "--dest-path",
        help="Path within destination volume to copy to",
    ),
    keep_pods: bool = typer.Option(
        False,
        "--keep-pods",
        help="Keep pods running after copy (for debugging)",
    ),
    vcpu: int = typer.Option(
        4,
        "--vcpu",
        help="vCPUs for transfer pods",
    ),
    memory: int = typer.Option(
        8,
        "--memory",
        help="Memory (GB) for transfer pods",
    ),
):
    """Copy a network volume to another datacenter using CPU pods and rsync.

    This command:
    1. Creates CPU pods in source and destination datacenters
    2. Uses rsync over SSH to transfer data
    3. Terminates pods when done (unless --keep-pods)

    Example:
        cloud volume copy hvdiy0svrv EU-CZ-1

    The transfer uses CPU-only pods (~$0.06/hr) which are very cost-effective.
    """
    client = get_runpod_client()

    def progress_callback(msg: str) -> None:
        console.print(f"  {msg}")

    def output_callback(line: str) -> None:
        console.print(line)

    copier = VolumeCopier(client)

    console.print("[bold]Starting volume copy...[/bold]\n")

    result = copier.copy(
        source_volume_id=source_volume_id,
        dest_datacenter=dest_datacenter,
        dest_volume_id=dest_volume_id,
        dest_volume_name=dest_volume_name,
        source_path=source_path,
        dest_path=dest_path,
        vcpu=vcpu,
        memory_gb=memory,
        keep_pods=keep_pods,
        progress_callback=progress_callback,
        output_callback=output_callback,
    )

    if result.success:
        console.print(f"\n[green]Transfer completed successfully![/green]")
        console.print(
            f"  Destination volume: [cyan]{result.dest_volume.id}[/cyan] "
            f"in {result.dest_volume.datacenter}"
        )
    else:
        console.print(f"\n[red]Transfer failed: {result.error}[/red]")
        raise typer.Exit(1)
