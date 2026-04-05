"""RunPod S3 API for network volumes."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from runpod_toolkit.cli._common import (
    get_runpod_s3_client,
    format_size,
)

app = typer.Typer(help="RunPod S3 API for network volumes", no_args_is_help=True)
console = Console()


@app.command("test")
def s3_test(
    volume_id: str = typer.Argument(..., help="Network volume ID to test"),
    datacenter: str = typer.Option("EU-RO-1", "--datacenter", "-d", help="Datacenter ID"),
):
    """Test S3 API connection to a network volume."""
    from runpod_toolkit.config import ConfigurationError

    console.print(f"[bold]Testing RunPod S3 API connection...[/bold]")
    console.print(f"  Volume: {volume_id}")
    console.print(f"  Datacenter: {datacenter}")
    console.print()

    try:
        client = get_runpod_s3_client(datacenter)
        console.print(f"  Endpoint: {client.endpoint}")

        with console.status("  Connecting..."):
            client.test_connection(volume_id)

        console.print("  [green]Connection successful![/green]")

        # Show volume contents summary
        with console.status("  Getting volume info..."):
            size_info = client.get_volume_size(volume_id)

        console.print()
        console.print(f"  Objects: {size_info['object_count']}")
        console.print(f"  Total size: {size_info['total_gb']:.2f} GB")

    except ConfigurationError as e:
        console.print(f"  [red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"  [red]Connection failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("upload")
def s3_upload(
    local_path: Path = typer.Argument(..., help="Local file to upload"),
    volume_id: str = typer.Argument(..., help="Network volume ID"),
    datacenter: str = typer.Option("EU-RO-1", "--datacenter", "-d", help="Datacenter ID"),
    dest: Optional[str] = typer.Option(None, "--dest", help="Remote path/prefix (default: filename)"),
):
    """Upload a file to a network volume via S3 API."""
    if not local_path.exists():
        console.print(f"[red]File not found: {local_path}[/red]")
        raise typer.Exit(1)

    file_size = local_path.stat().st_size
    file_size_gb = file_size / (1024 * 1024 * 1024)

    # Determine remote key
    remote_key = dest if dest else local_path.name
    if dest and dest.endswith('/'):
        remote_key = f"{dest}{local_path.name}"

    console.print(f"[bold]Uploading to RunPod Network Volume[/bold]")
    console.print(f"  Source: {local_path}")
    console.print(f"  Size: {file_size_gb:.2f} GB")
    console.print(f"  Destination: {volume_id}/{remote_key}")
    console.print(f"  Datacenter: {datacenter}")
    console.print()

    try:
        client = get_runpod_s3_client(datacenter)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("({task.completed:.1f}/{task.total:.1f} GB)"),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading...", total=file_size_gb)

            def update_progress(bytes_transferred: int):
                gb_transferred = bytes_transferred / (1024 * 1024 * 1024)
                progress.update(task, completed=gb_transferred)

            result = client.upload_file(
                local_path,
                volume_id,
                remote_key,
                progress_callback=update_progress,
            )

        console.print()
        console.print(f"[green]Upload complete![/green]")
        console.print(f"  S3 URI: {result.s3_uri}")
        console.print(f"  ETag: {result.etag}")

    except Exception as e:
        console.print(f"[red]Upload failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("download")
def s3_download(
    volume_id: str = typer.Argument(..., help="Network volume ID"),
    remote_key: str = typer.Argument(..., help="Remote file path in volume"),
    local_path: Path = typer.Argument(..., help="Local path to save file"),
    datacenter: str = typer.Option("EU-RO-1", "--datacenter", "-d", help="Datacenter ID"),
):
    """Download a file from a network volume via S3 API."""
    console.print(f"[bold]Downloading from RunPod Network Volume[/bold]")
    console.print(f"  Source: {volume_id}/{remote_key}")
    console.print(f"  Destination: {local_path}")
    console.print(f"  Datacenter: {datacenter}")
    console.print()

    try:
        client = get_runpod_s3_client(datacenter)

        # Get file size first
        metadata = client.get_object_metadata(volume_id, remote_key)
        if metadata is None:
            console.print(f"[red]Object not found: {remote_key}[/red]")
            raise typer.Exit(1)

        file_size_gb = metadata.size / (1024 * 1024 * 1024)
        console.print(f"  Size: {file_size_gb:.2f} GB")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=file_size_gb)

            def update_progress(bytes_transferred: int):
                gb_transferred = bytes_transferred / (1024 * 1024 * 1024)
                progress.update(task, completed=gb_transferred)

            client.download_file(
                volume_id,
                remote_key,
                local_path,
                progress_callback=update_progress,
            )

        console.print()
        console.print(f"[green]Download complete![/green]")
        console.print(f"  Saved to: {local_path}")

    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def s3_list(
    volume_id: str = typer.Argument(..., help="Network volume ID"),
    datacenter: str = typer.Option("EU-RO-1", "--datacenter", "-d", help="Datacenter ID"),
    prefix: str = typer.Option("", "--prefix", "-p", help="Filter by prefix"),
):
    """List contents of a network volume."""
    console.print(f"[bold]Contents of {volume_id}[/bold]")
    if prefix:
        console.print(f"[dim]Prefix: {prefix}[/dim]")
    console.print()

    try:
        client = get_runpod_s3_client(datacenter)

        objects = list(client.list_objects(volume_id, prefix))

        if not objects:
            console.print("[yellow]No objects found[/yellow]")
            return

        table = Table()
        table.add_column("Key", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Last Modified")

        total_size = 0
        for obj in objects:
            size_str = format_size(obj.size)
            table.add_row(obj.key, size_str, obj.last_modified[:19])
            total_size += obj.size

        console.print(table)
        console.print()
        console.print(f"Total: {len(objects)} objects, {format_size(total_size)}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("delete")
def s3_delete(
    volume_id: str = typer.Argument(..., help="Network volume ID"),
    remote_key: str = typer.Argument(..., help="Remote file path to delete"),
    datacenter: str = typer.Option("EU-RO-1", "--datacenter", "-d", help="Datacenter ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a file from a network volume."""
    console.print(f"[bold]Delete from RunPod Network Volume[/bold]")
    console.print(f"  Target: {volume_id}/{remote_key}")
    console.print()

    try:
        client = get_runpod_s3_client(datacenter)

        # Check if exists
        metadata = client.get_object_metadata(volume_id, remote_key)
        if metadata is None:
            console.print(f"[yellow]Object not found: {remote_key}[/yellow]")
            return

        console.print(f"  Size: {format_size(metadata.size)}")

        if not force:
            confirm = typer.confirm("Are you sure you want to delete this file?")
            if not confirm:
                console.print("[dim]Cancelled[/dim]")
                return

        client.delete_object(volume_id, remote_key)
        console.print(f"[green]Deleted: {remote_key}[/green]")

    except Exception as e:
        console.print(f"[red]Delete failed: {e}[/red]")
        raise typer.Exit(1)
