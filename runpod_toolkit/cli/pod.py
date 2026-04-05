"""Pod management commands."""

import os
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from runpod_toolkit.cli._common import (
    get_cloud_config,
    get_runpod_client,
    require_ssh,
    get_status_color,
    show_pod_logs,
)

app = typer.Typer(help="Pod management", no_args_is_help=True)
console = Console()


@app.command("create")
def pod_create(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Pod name",
    ),
    gpu_type: Optional[str] = typer.Option(
        None,
        "--gpu",
        "-g",
        help="GPU type (default: from config)",
    ),
    volume_id: Optional[str] = typer.Option(
        None,
        "--volume",
        "-v",
        help="Network volume ID to attach (default: from config)",
    ),
    no_volume: bool = typer.Option(
        False,
        "--no-volume",
        help="Don't attach any network volume (override default)",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Docker image",
    ),
    min_ram: Optional[int] = typer.Option(
        None,
        "--min-ram",
        help="Minimum system RAM in GB (filters available hosts)",
    ),
    datacenter: Optional[str] = typer.Option(
        None,
        "--datacenter",
        "-d",
        help="Datacenter ID (e.g. EUR-IS-1). Default: from config (EU-RO-1)",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for pod to be ready",
    ),
):
    """Create a new pod."""
    from runpod_toolkit.compute import PodStatus

    config = get_cloud_config()
    gpu_type = gpu_type or config.runpod.default_gpu_type

    # Use default volume from config if not specified (unless --no-volume)
    if not no_volume:
        volume_id = volume_id or config.runpod.default_network_volume_id

    client = get_runpod_client()

    console.print(f"[bold]Creating pod '{name}' with {gpu_type}...[/bold]")
    if datacenter:
        console.print(f"  Datacenter: {datacenter}")
    if volume_id:
        console.print(f"  Volume: {volume_id}")
    if image:
        console.print(f"  Image: {image}")
    if min_ram:
        console.print(f"  Min RAM: {min_ram}GB")

    try:
        pod = client.create_pod(
            name=name,
            gpu_type=gpu_type,
            network_volume_id=volume_id,
            datacenter=datacenter or config.runpod.default_datacenter,
            image=image,
            min_memory_gb=min_ram,
        )

        console.print(f"\n[green]Pod created: {pod.id}[/green]")

        if wait:
            with console.status("Waiting for pod to be ready..."):
                pod = client.wait_for_pod(pod.id, PodStatus.RUNNING, timeout=600)

            console.print("[green]Pod is ready![/green]")

        console.print(f"\n  ID: [cyan]{pod.id}[/cyan]")
        console.print(f"  Status: {pod.status.value}")
        console.print(f"  GPU: {pod.gpu_type}")
        console.print(f"  Cost: ${pod.cost_per_hour:.2f}/hr")

        if pod.ssh_host and pod.ssh_port:
            console.print(f"\n  SSH: ssh root@{pod.ssh_host} -p {pod.ssh_port}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("create-cpu")
def pod_create_cpu(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Pod name",
    ),
    vcpu: int = typer.Option(
        4,
        "--vcpu",
        "-c",
        help="Number of vCPUs",
    ),
    memory: int = typer.Option(
        16,
        "--memory",
        "-m",
        help="RAM in GB",
    ),
    volume_id: Optional[str] = typer.Option(
        None,
        "--volume",
        "-v",
        help="Network volume ID to attach (default: from config)",
    ),
    no_volume: bool = typer.Option(
        False,
        "--no-volume",
        help="Don't attach any network volume (override default)",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Docker image",
    ),
    cpu_flavor: str = typer.Option(
        "cpu3c",
        "--flavor",
        "-f",
        help="CPU flavor: cpu3c (compute), cpu3g (general), cpu3m (memory), cpu5c/cpu5g/cpu5m (5th gen)",
    ),
    datacenter: Optional[str] = typer.Option(
        None,
        "--datacenter",
        "-d",
        help="Datacenter ID (e.g. EUR-IS-1). Default: from config (EU-RO-1)",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for pod to be ready",
    ),
):
    """Create a CPU-only pod (no GPU).

    CPU pods are much cheaper (~$0.03-0.10/hr) and useful for:
    - Data processing
    - Transfer operations
    - Lightweight workloads

    CPU flavors:
    - cpu3c: 3rd gen compute optimized
    - cpu3g: 3rd gen general purpose
    - cpu3m: 3rd gen memory optimized
    - cpu5c/cpu5g/cpu5m: 5th gen variants

    Example:
        cloud pod create-cpu -n data-proc --vcpu 32 --memory 64 --volume hvdiy0svrv
    """
    from runpod_toolkit.compute import PodStatus

    config = get_cloud_config()

    # Use default volume from config if not specified (unless --no-volume)
    if not no_volume:
        volume_id = volume_id or config.runpod.default_network_volume_id

    client = get_runpod_client()

    console.print(f"[bold]Creating CPU pod '{name}' ({vcpu} vCPU, {memory}GB RAM)...[/bold]")
    console.print(f"  Flavor: {cpu_flavor}")
    if datacenter:
        console.print(f"  Datacenter: {datacenter}")
    if volume_id:
        console.print(f"  Volume: {volume_id}")
    if image:
        console.print(f"  Image: {image}")

    try:
        pod = client.create_cpu_pod(
            name=name,
            vcpu_count=vcpu,
            memory_gb=memory,
            network_volume_id=volume_id,
            datacenter=datacenter or config.runpod.default_datacenter,
            image=image,
            cpu_flavor=cpu_flavor,
        )

        console.print(f"\n[green]CPU pod created: {pod.id}[/green]")

        if wait:
            with console.status("Waiting for pod to be ready..."):
                pod = client.wait_for_pod(pod.id, PodStatus.RUNNING, timeout=600)

            console.print("[green]Pod is ready![/green]")

        console.print(f"\n  ID: [cyan]{pod.id}[/cyan]")
        console.print(f"  Status: {pod.status.value}")
        console.print(f"  Type: CPU ({vcpu} vCPU, {memory}GB)")
        console.print(f"  Cost: ${pod.cost_per_hour:.2f}/hr")

        if pod.ssh_host and pod.ssh_port:
            console.print(f"\n  SSH: ssh root@{pod.ssh_host} -p {pod.ssh_port}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("probe-cpu")
def pod_probe_cpu(
    vcpu: str = typer.Option(
        "16,32",
        "--vcpu",
        "-c",
        help="vCPU counts to test (comma-separated)",
    ),
    memory: str = typer.Option(
        "64,128",
        "--memory",
        "-m",
        help="RAM in GB per vCPU config (comma-separated, matching vcpu list)",
    ),
    flavor: str = typer.Option(
        "cpu5g",
        "--flavor",
        "-f",
        help="CPU flavor",
    ),
    image: str = typer.Option(
        "ubuntu:22.04",
        "--image",
        "-i",
        help="Docker image",
    ),
    datacenters: Optional[str] = typer.Option(
        None,
        "--datacenters",
        "-d",
        help="Comma-separated DC list (default: all listed DCs from API)",
    ),
    workers: int = typer.Option(
        10,
        "--workers",
        "-w",
        help="Max parallel workers",
    ),
    timeout: int = typer.Option(
        600,
        "--timeout",
        "-t",
        help="Total timeout per pod in seconds",
    ),
):
    """Probe datacenters for CPU pod availability.

    Creates pods concurrently in each datacenter, waits for SSH, then
    terminates. Reports which datacenters successfully accepted the pod.

    Example:
        cloud pod probe-cpu --datacenters EU-CZ-1 --vcpu 16 --memory 64
        cloud pod probe-cpu
    """
    import concurrent.futures
    import subprocess
    import time
    from dataclasses import dataclass

    from runpod_toolkit.compute import RunPodClient, PodStatus
    from runpod_toolkit.compute.runpod_client import RunPodError
    from runpod_toolkit.config import RunPodConfig

    RUNPOD_KEY = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"

    # Parse comma-separated options
    vcpu_list = [int(v.strip()) for v in vcpu.split(",")]
    memory_list = [int(m.strip()) for m in memory.split(",")]
    if len(vcpu_list) != len(memory_list):
        console.print("[red]Error: --vcpu and --memory must have the same number of entries[/red]")
        raise typer.Exit(1)
    memory_for_vcpu = dict(zip(vcpu_list, memory_list))

    # Get datacenter list
    config = get_cloud_config()
    client = get_runpod_client()

    if datacenters:
        dc_list = [d.strip() for d in datacenters.split(",")]
    else:
        with console.status("Querying listed datacenters..."):
            dcs = client.get_datacenters(listed_only=True)
            dc_list = [dc["id"] for dc in dcs]

    if not dc_list:
        console.print("[red]No datacenters found[/red]")
        raise typer.Exit(1)

    tasks = [(dc, v) for v in vcpu_list for dc in dc_list]
    console.print(
        f"[bold]Probing {len(dc_list)} datacenters x {len(vcpu_list)} vCPU configs "
        f"= {len(tasks)} pods[/bold]"
    )
    console.print(f"  Image: {image}  Flavor: {flavor}  Timeout: {timeout}s")

    if not RUNPOD_KEY.exists():
        console.print(f"[yellow]Warning: RunPod SSH key not found at {RUNPOD_KEY} — SSH tests will fail[/yellow]")

    @dataclass
    class ProbeResult:
        datacenter: str
        vcpu: int
        pod_id: Optional[str] = None
        created: bool = False
        ssh_ok: bool = False
        error: Optional[str] = None
        elapsed_s: float = 0.0

    def test_ssh(host: str, port: int) -> bool:
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-p", str(port),
        ]
        if RUNPOD_KEY.exists():
            ssh_cmd.extend(["-i", str(RUNPOD_KEY)])
        ssh_cmd.extend([f"root@{host}", "echo ok"])
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=20)
            return result.returncode == 0 and "ok" in result.stdout
        except Exception:
            return False

    def probe_datacenter(dc: str, vcpu_count: int, memory_gb: int) -> ProbeResult:
        result = ProbeResult(datacenter=dc, vcpu=vcpu_count)
        t0 = time.time()
        # Each thread gets its own client (thread safety)
        thread_client = RunPodClient(RunPodConfig.from_env())
        pod_id = None

        try:
            name = f"probe-{dc.lower()}-{vcpu_count}v"
            console.print(f"  [{dc}/{vcpu_count}v] Creating pod...")
            pod = thread_client.create_cpu_pod(
                name=name,
                vcpu_count=vcpu_count,
                memory_gb=memory_gb,
                datacenter=dc,
                image=image,
                cpu_flavor=flavor,
            )
            pod_id = pod.id
            result.pod_id = pod_id
            result.created = True
            console.print(f"  [{dc}/{vcpu_count}v] Pod created: {pod_id}")

            pod = thread_client.wait_for_pod(
                pod_id, PodStatus.RUNNING,
                timeout=timeout,
                poll_interval=15,
                wait_for_ssh=True,
            )
            console.print(f"  [{dc}/{vcpu_count}v] RUNNING, SSH at {pod.ssh_host}:{pod.ssh_port}")

            ssh_deadline = time.time() + 300
            while time.time() < ssh_deadline:
                if test_ssh(pod.ssh_host, pod.ssh_port):
                    result.ssh_ok = True
                    console.print(f"  [{dc}/{vcpu_count}v] [green]SSH OK![/green] ({time.time() - t0:.0f}s)")
                    break
                time.sleep(15)

            if not result.ssh_ok:
                result.error = "SSH auth/connect failed after retries"
                console.print(f"  [{dc}/{vcpu_count}v] [yellow]SSH never connected[/yellow]")

        except Exception as e:
            result.error = str(e)[:200]
            console.print(f"  [{dc}/{vcpu_count}v] [red]Error: {e}[/red]")
        finally:
            if pod_id:
                try:
                    thread_client.terminate_pod(pod_id)
                    console.print(f"  [{dc}/{vcpu_count}v] Terminated {pod_id}")
                except Exception as e:
                    console.print(f"  [{dc}/{vcpu_count}v] [red]Failed to terminate {pod_id}: {e}[/red]")

        result.elapsed_s = time.time() - t0
        return result

    # Run probes concurrently
    all_results: list[ProbeResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_datacenter, dc, v, memory_for_vcpu[v]
            ): (dc, v)
            for dc, v in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            dc, v = futures[future]
            try:
                r = future.result()
                all_results.append(r)
            except Exception as e:
                console.print(f"  [{dc}/{v}v] [red]Unexpected: {e}[/red]")
                all_results.append(ProbeResult(datacenter=dc, vcpu=v, error=str(e)))

    # Results table
    all_results.sort(key=lambda r: (r.datacenter, r.vcpu))
    table = Table(title=f"CPU Pod Probe Results — {flavor}")
    table.add_column("Datacenter", style="cyan")
    table.add_column("vCPU", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Created", justify="center")
    table.add_column("SSH OK", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Error")

    for r in all_results:
        mem = memory_for_vcpu[r.vcpu]
        created_str = "[green]YES[/green]" if r.created else "[red]no[/red]"
        ssh_str = "[green]YES[/green]" if r.ssh_ok else "[red]no[/red]"
        err = (r.error or "")[:50]
        table.add_row(
            r.datacenter,
            str(r.vcpu),
            f"{mem}G",
            created_str,
            ssh_str,
            f"{r.elapsed_s:.0f}s",
            err,
        )

    console.print()
    console.print(table)

    # Summary of working DCs
    console.print()
    console.print(f"[bold]Datacenters with working {flavor} pods (SSH verified)[/bold]")
    dc_results: dict[str, dict[int, bool]] = {}
    for r in all_results:
        dc_results.setdefault(r.datacenter, {})[r.vcpu] = r.ssh_ok

    any_found = False
    for dc in dc_list:
        working_configs = []
        for v in vcpu_list:
            if dc_results.get(dc, {}).get(v, False):
                working_configs.append(f"{v} vCPU / {memory_for_vcpu[v]}G")
        if working_configs:
            any_found = True
            console.print(f"  [green]{dc:<12}[/green] {', '.join(working_configs)}")

    if not any_found:
        console.print("  [yellow](none)[/yellow]")


@app.command("list")
def pod_list(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (RUNNING, CREATED, EXITED)",
    ),
):
    """List all pods."""
    client = get_runpod_client()

    with console.status("Fetching pods..."):
        try:
            pods = client.list_pods()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Filter by status if specified
    if status:
        pods = [p for p in pods if p.status.value.upper() == status.upper()]

    if not pods:
        console.print("[yellow]No pods found[/yellow]")
        return

    table = Table(title="Pods")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("$/hr", justify="right")
    table.add_column("SSH")

    for pod in pods:
        status_color = get_status_color(pod.status.value)
        ssh_info = f"{pod.ssh_host}:{pod.ssh_port}" if pod.ssh_host else "-"
        table.add_row(
            pod.id,
            pod.name,
            f"[{status_color}]{pod.status.value}[/{status_color}]",
            pod.gpu_type,
            f"{pod.cost_per_hour:.2f}",
            ssh_info,
        )

    console.print(table)


@app.command("terminate")
def pod_terminate(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID to terminate",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
):
    """Terminate a pod."""
    client = get_runpod_client()

    # Get pod info first
    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"Pod: [cyan]{pod.name}[/cyan] ({pod.gpu_type})")
    console.print(f"Status: {pod.status.value}")

    if not force:
        confirm = typer.confirm("Are you sure you want to terminate this pod?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        client.terminate_pod(pod_id)
        console.print("[green]Pod terminated successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ssh")
def pod_ssh(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID to SSH into",
    ),
):
    """SSH into a running pod."""
    client = get_runpod_client()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    console.print(f"Connecting to [cyan]{pod.name}[/cyan]...")
    console.print(f"  Host: {pod.ssh_host}")
    console.print(f"  Port: {pod.ssh_port}")
    console.print()

    # Build SSH command with RunPod key if available
    runpod_key = Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go"
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if runpod_key.exists():
        ssh_cmd.extend(["-i", str(runpod_key)])
    ssh_cmd.extend(["-p", str(pod.ssh_port), f"root@{pod.ssh_host}"])

    # Use os.execvp to replace this process with SSH
    try:
        os.execvp("ssh", ssh_cmd)
    except FileNotFoundError:
        console.print("[red]SSH client not found. Install OpenSSH.[/red]")
        raise typer.Exit(1)


@app.command("status")
def pod_status(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
):
    """Show detailed pod status."""
    client = get_runpod_client()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Pod: {pod.name}[/bold]\n")
    console.print(f"  ID: [cyan]{pod.id}[/cyan]")
    console.print(f"  Status: {pod.status.value}")
    console.print(f"  GPU: {pod.gpu_type} x {pod.gpu_count}")
    console.print(f"  Cost: ${pod.cost_per_hour:.2f}/hr")
    console.print(f"  Datacenter: {pod.datacenter}")
    console.print(f"  Created: {pod.created_at}")

    if pod.network_volume_id:
        console.print(f"  Volume: {pod.network_volume_id}")

    if pod.ssh_host and pod.ssh_port:
        console.print(f"\n  SSH: ssh root@{pod.ssh_host} -p {pod.ssh_port}")


@app.command("gpus")
def pod_gpus(
    min_memory: int = typer.Option(
        24,
        "--min-memory",
        "-m",
        help="Minimum GPU memory in GB",
    ),
):
    """List available GPU types."""
    client = get_runpod_client()

    with console.status("Fetching GPU availability..."):
        try:
            gpus = client.get_gpu_availability(min_memory_gb=min_memory)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    if not gpus:
        console.print(f"[yellow]No GPUs available with >= {min_memory}GB memory[/yellow]")
        return

    table = Table(title=f"Available GPUs (>= {min_memory}GB)")
    table.add_column("GPU Type", style="cyan")
    table.add_column("Memory", justify="right")
    table.add_column("Available", justify="right")
    table.add_column("$/hr", justify="right")

    for gpu in gpus:
        table.add_row(
            gpu.name,
            f"{gpu.memory_gb}GB",
            str(gpu.available),
            f"${gpu.hourly_cost:.2f}",
        )

    console.print(table)


@app.command("push")
def pod_push(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID to push code to",
    ),
    branch: str = typer.Option(
        "main",
        "--branch",
        "-b",
        help="Branch to push",
    ),
    repo_path: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="Local repo path (default: current directory)",
    ),
):
    """Push code to pod via git.

    Pushes local code to the pod's bare git repository. The pod's post-receive
    hook will automatically checkout the code and install the package.
    """
    from pathlib import Path
    from runpod_toolkit.compute import PodManager

    client = get_runpod_client()
    config = get_cloud_config()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    console.print(f"Pushing code to [cyan]{pod.name}[/cyan]...")

    try:
        manager = PodManager(config)
        repo = Path(repo_path) if repo_path else None
        success = manager.push_code(pod, local_repo_path=repo, branch=branch)

        if success:
            console.print("[green]Code pushed successfully![/green]")
        else:
            console.print("[red]Failed to push code[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("exec")
def pod_exec(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
    command: List[str] = typer.Argument(
        ...,
        help="Command to execute (use -- before command if it has flags)",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Timeout in seconds",
    ),
    workdir: str = typer.Option(
        "/root/code",
        "--workdir",
        "-w",
        help="Working directory",
    ),
):
    """Execute command on pod via SSH.

    Example:
        cloud pod exec abc123 -- ls -la
        cloud pod exec abc123 -w /workspace -- nvidia-smi
    """
    from runpod_toolkit.compute import PodManager

    client = get_runpod_client()
    config = get_cloud_config()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    cmd_str = " ".join(command)
    console.print(f"Executing on [cyan]{pod.name}[/cyan]: {cmd_str}")
    console.print()

    try:
        manager = PodManager(config)
        result = manager.run_command_on_pod(
            pod,
            cmd_str,
            timeout=timeout,
            working_dir=workdir
        )

        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]")

        raise typer.Exit(result.exit_code)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("gpu")
def pod_gpu(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
):
    """Show GPU utilization and memory."""
    from runpod_toolkit.compute import PodManager

    client = get_runpod_client()
    config = get_cloud_config()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"

    try:
        manager = PodManager(config)
        result = manager.run_command_on_pod(pod, cmd, timeout=30)

        if result.exit_code != 0:
            console.print(f"[red]nvidia-smi failed: {result.stderr}[/red]")
            raise typer.Exit(1)

        # Parse CSV output and display as table
        table = Table(title=f"GPU Status: {pod.name}")
        table.add_column("#", style="dim")
        table.add_column("GPU", style="cyan")
        table.add_column("Util", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Temp", justify="right")

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                idx, name, util, mem_used, mem_total, temp = parts[:6]
                util_color = "green" if float(util) < 80 else "yellow" if float(util) < 95 else "red"
                table.add_row(
                    idx,
                    name,
                    f"[{util_color}]{util}%[/{util_color}]",
                    f"{float(mem_used)/1024:.1f}/{float(mem_total)/1024:.0f} GB",
                    f"{temp}C",
                )

        console.print(table)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("top")
def pod_top(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
):
    """Show system resource usage (CPU, memory, disk, GPU)."""
    from runpod_toolkit.compute import PodManager

    client = get_runpod_client()
    config = get_cloud_config()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    cmd = """echo "=== CPU ===" && nproc && uptime && \
echo "=== Memory ===" && free -h | head -2 && \
echo "=== Disk ===" && df -h /workspace 2>/dev/null || df -h / && \
echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader && \
echo "=== Top Processes ===" && ps aux --sort=-%mem | head -6"""

    try:
        manager = PodManager(config)
        result = manager.run_command_on_pod(pod, cmd, timeout=30)

        console.print(f"[bold]System Resources: {pod.name}[/bold]\n")

        # Display raw output with some formatting
        lines = result.stdout.split("\n")

        for line in lines:
            if line.startswith("=== "):
                current_section = line.strip("= ")
                console.print(f"[cyan]{current_section}[/cyan]")
            elif line.strip():
                console.print(f"  {line}")

        if result.stderr:
            console.print(f"\n[yellow]{result.stderr}[/yellow]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("health")
def pod_health(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
):
    """Run health checks on pod."""
    from runpod_toolkit.compute import PodManager

    client = get_runpod_client()
    config = get_cloud_config()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    require_ssh(pod, console)

    # Run all health checks in one command
    cmd = """echo "GPU:$(nvidia-smi > /dev/null 2>&1 && echo OK || echo FAIL)" && \
echo "DISK:$(df /workspace 2>/dev/null | awk 'NR==2 {gsub(/%/,""); if ($5 < 90) print "OK:"100-$5"%"; else print "FAIL:"100-$5"%"}')" && \
echo "MEM:$(free | awk '/Mem:/ {pct=100*$3/$2; if (pct < 95) print "OK:"int(100-pct)"%"; else print "FAIL:"int(100-pct)"%"}')" """

    try:
        manager = PodManager(config)
        result = manager.run_command_on_pod(pod, cmd, timeout=30)

        console.print(f"[bold]Pod Health: {pod.name}[/bold]\n")

        checks = {}
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                checks[key] = value

        passed = 0
        total = 3

        # GPU check
        if checks.get("GPU") == "OK":
            console.print("  [green]OK[/green] GPU responding")
            passed += 1
        else:
            console.print("  [red]FAIL[/red] GPU not responding")

        # Disk check
        disk_val = checks.get("DISK", "")
        if disk_val.startswith("OK:"):
            console.print(f"  [green]OK[/green] Disk space: {disk_val[3:]} free")
            passed += 1
        else:
            pct = disk_val.replace("FAIL:", "") if ":" in disk_val else "?"
            console.print(f"  [red]FAIL[/red] Disk space low: {pct} free")

        # Memory check
        mem_val = checks.get("MEM", "")
        if mem_val.startswith("OK:"):
            console.print(f"  [green]OK[/green] Memory: {mem_val[3:]} free")
            passed += 1
        else:
            pct = mem_val.replace("FAIL:", "") if ":" in mem_val else "?"
            console.print(f"  [red]FAIL[/red] Memory low: {pct} free")

        console.print()
        if passed >= 3:
            status = "[green]HEALTHY[/green]"
        elif passed >= 2:
            status = "[yellow]DEGRADED[/yellow]"
        else:
            status = "[red]UNHEALTHY[/red]"

        console.print(f"Status: {status} ({passed}/{total} checks passed)")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _load_runpod_env_file() -> dict[str, str]:
    """Load environment variables from /etc/runpod.env if it exists.

    This file is created at pod startup to persist RUNPOD_POD_ID and
    RUNPOD_API_KEY for SSH sessions that don't inherit container env vars.

    Returns:
        Dict of environment variables from the file.
    """
    env_file = "/etc/runpod.env"
    result = {}
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    result[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    except Exception as e:
        console.print(f"[dim]Warning: Failed to read {env_file}: {e}[/dim]")
    return result


@app.command("self-terminate")
def pod_self_terminate(
    pod_id: Optional[str] = typer.Option(
        None,
        "--pod-id",
        "-p",
        help="Pod ID (auto-detected from env/file if not specified)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="RunPod API key (auto-detected from env/file if not specified)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
):
    """Terminate this pod (run from within the pod).

    Discovers pod ID and API key from multiple sources (in order):
    1. CLI arguments (--pod-id, --api-key)
    2. /etc/runpod.env file (persisted at pod startup)
    3. Environment variables (RUNPOD_POD_ID, RUNPOD_API_KEY)

    Designed to be called by scripts when they complete.

    Example:
        cloud pod self-terminate --force
        cloud pod self-terminate --pod-id abc123 --force
    """
    # Load from /etc/runpod.env as fallback
    runpod_env = _load_runpod_env_file()

    # Resolve pod_id: CLI arg -> file -> env var
    if not pod_id:
        pod_id = runpod_env.get("RUNPOD_POD_ID")
    if not pod_id:
        pod_id = os.environ.get("RUNPOD_POD_ID")

    # Resolve api_key: CLI arg -> file -> env var
    if not api_key:
        api_key = runpod_env.get("RUNPOD_API_KEY")
    if not api_key:
        api_key = os.environ.get("RUNPOD_API_KEY")

    if not pod_id:
        console.print("[red]Error: RUNPOD_POD_ID not found[/red]")
        console.print("[dim]Checked: --pod-id arg, /etc/runpod.env, RUNPOD_POD_ID env var[/dim]")
        console.print("[dim]Use --pod-id to specify explicitly[/dim]")
        raise typer.Exit(1)

    if not api_key:
        console.print("[red]Error: RUNPOD_API_KEY not found[/red]")
        console.print("[dim]Checked: --api-key arg, /etc/runpod.env, RUNPOD_API_KEY env var[/dim]")
        console.print("[dim]Ensure the runpod_api_key secret is configured in RunPod console[/dim]")
        raise typer.Exit(1)

    console.print(f"Pod ID: [cyan]{pod_id}[/cyan]")

    if not force:
        confirm = typer.confirm("Terminate this pod?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        import runpod
        runpod.api_key = api_key
        runpod.terminate_pod(pod_id)
        console.print("[green]Pod termination initiated[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def pod_logs(
    pod_id: str = typer.Argument(
        ...,
        help="Pod ID",
    ),
    tail: int = typer.Option(
        50,
        "--tail",
        "-n",
        help="Number of lines to show",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Follow log output (like tail -f)",
    ),
    log_path: Optional[str] = typer.Option(
        None,
        "--path",
        "-p",
        help="Explicit log file path (auto-discovers if not set)",
    ),
):
    """Show logs from pod.

    Auto-discovers the most recent log file or falls back to /workspace/outputs.
    Use --path for explicit log file path.
    """
    client = get_runpod_client()

    try:
        pod = client.get_pod(pod_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    show_pod_logs(pod, log_path=log_path, tail=tail, follow=follow)
