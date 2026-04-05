"""Cloud CLI — RunPod compute, storage, and vLLM serving."""
import typer

from runpod_toolkit.cli.config import app as config_app
from runpod_toolkit.cli.pod import app as pod_app
from runpod_toolkit.cli.volume import app as volume_app
from runpod_toolkit.cli.s3 import app as s3_app
from runpod_toolkit.cli.serve import app as serve_app

app = typer.Typer(help="RunPod toolkit — compute, storage, and model serving")
app.add_typer(config_app, name="config")
app.add_typer(pod_app, name="pod")
app.add_typer(volume_app, name="volume")
app.add_typer(s3_app, name="s3")
app.add_typer(serve_app, name="serve")

if __name__ == "__main__":
    app()
