"""CLI commands for vLLM model serving on RunPod."""
import typer
from rich.console import Console
from rich.table import Table

from runpod_toolkit.cli._common import get_runpod_client, handle_cloud_errors
from runpod_toolkit.serve.model_registry import load_registry
from runpod_toolkit.serve.vllm_pod import resolve_pod_spec, start_vllm_pod, stop_vllm_pod, wait_for_healthy
from runpod_toolkit.serve.health import check_health, list_models

app = typer.Typer(help="vLLM model serving management")
console = Console()


@app.command()
@handle_cloud_errors
def start(
    model: str = typer.Argument(help="HuggingFace model ID"),
    gpu: str = typer.Option(..., "--gpu", help="GPU type (e.g. 'NVIDIA H100 80GB HBM3')"),
    variant: str = typer.Option("fp8", help="Model variant"),
    volume: str | None = typer.Option(None, "--volume", help="Volume ID for model cache"),
    gpu_count: int | None = typer.Option(None, "--gpu-count", help="Number of GPUs"),
    name: str | None = typer.Option(None, "--name", help="Pod name"),
):
    """Start a vLLM inference pod."""
    registry = load_registry()
    config = registry.get(model)
    if not config:
        console.print(f"[red]Model {model!r} not found in registry[/red]")
        console.print("Available models:", ", ".join(registry.keys()) or "(none — add YAML to configs/models/)")
        raise typer.Exit(1)

    spec = resolve_pod_spec(config, variant, gpu, gpu_count, volume)
    client = get_runpod_client()
    pod = start_vllm_pod(client, spec, pod_name=name)
    console.print(f"[green]Pod {pod.id} created ({pod.name})[/green]")
    console.print(f"  Model: {spec.model_id}")
    console.print(f"  GPU: {spec.gpu_count}x {spec.gpu_type}")
    if volume:
        console.print(f"  Volume: {volume}")


@app.command()
@handle_cloud_errors
def stop(pod_id: str = typer.Argument(help="Pod ID to terminate")):
    """Stop a vLLM inference pod."""
    client = get_runpod_client()
    stop_vllm_pod(client, pod_id)
    console.print(f"[green]Pod {pod_id} terminated[/green]")


@app.command()
@handle_cloud_errors
def health(pod_id: str = typer.Argument(help="Pod ID to check")):
    """Check health of a vLLM pod."""
    client = get_runpod_client()
    pod = client.get_pod(pod_id)
    if not pod or not pod.ssh_host:
        console.print(f"[red]Pod {pod_id} not found or no SSH access[/red]")
        raise typer.Exit(1)

    ok, detail = check_health(pod.ssh_host, pod.ssh_port)
    if ok:
        models = list_models(pod.ssh_host, pod.ssh_port)
        console.print(f"[green]Healthy[/green] — models: {', '.join(models) or '(loading)'}")
    else:
        console.print(f"[red]Unhealthy[/red] — {detail}")
        raise typer.Exit(1)


@app.command("list")
@handle_cloud_errors
def list_cmd():
    """List running inference pods."""
    client = get_runpod_client()
    pods = client.list_pods()
    vllm_pods = [p for p in pods if p.name and p.name.startswith("vllm-")]

    if not vllm_pods:
        console.print("No vLLM pods running")
        return

    table = Table(title="vLLM Inference Pods")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("Cost/hr")

    for p in vllm_pods:
        table.add_row(p.id, p.name, p.status.value, p.gpu_type or "?", f"${p.cost_per_hr:.2f}" if p.cost_per_hr else "?")

    console.print(table)
