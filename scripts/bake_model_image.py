#!/usr/bin/env python3
"""Bake a vLLM model image with one Docker layer per safetensors shard.

Why one-layer-per-shard?
------------------------
Pushing a single 75-100 GB layer to Docker Hub is unreliable: pushes hang
in the "Preparing" phase for 20+ minutes and rarely complete (see project
memory ``feedback_dockerd_push_build_contention.md``). Splitting the model
into one COPY per shard creates ~30-40 small (~2 GB) layers that can be
uploaded in parallel and individually retried by ``docker push``.

How it works
------------
1. Resolves the HF cache snapshot dir for ``--model``.
2. Creates a build staging dir on the SAME filesystem as the model
   (so hardlinks are free), then hardlinks each shard / config file from
   its blob into the staging dir.
3. Generates a ``Dockerfile.shards`` with:
       FROM leosiriusdawn/runpod-vllm:latest
       ENV MODEL_NAME=/models/<short-name>
       <one COPY for all small files>
       <one COPY per safetensors shard>
4. Optionally runs ``docker build`` and ``docker push``.

Usage
-----
    bake_model_image.py --model Qwen/Qwen3-Coder-Next-FP8 \
        --tag leosiriusdawn/runpod-vllm:qwen3-coder-next-fp8-baked \
        --build --push

The script is intentionally side-effect-light when ``--build/--push`` are
omitted: it only writes the staging dir and the Dockerfile, then prints a
``docker build`` invocation the operator can run by hand.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HF cache lives here. The orchestrator side keeps it on Leo_X10p_4TB_00 to
# avoid filling root or Internal-2nd. Override with HF_HOME for testing.
DEFAULT_HF_HOME = Path(
    os.environ.get("HF_HOME", "/media/leo/Leo_X10p_4TB_00/leo/models")
)

# Build staging lives next to the model so hardlinks work without copying.
DEFAULT_STAGING_ROOT = Path(
    "/media/leo/Leo_X10p_4TB_00/leo/build-staging"
)

BASE_IMAGE = "leosiriusdawn/runpod-vllm:latest"

# Files we treat as "small" — bundled into one COPY layer for cleanliness.
SMALL_FILE_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
}

# Extra files we silently include if present (parser plugins, READMEs).
EXTRA_FILE_GLOBS = ("*.py", "README*", "LICENSE*", "*.md")


# ---------------------------------------------------------------------------
# HF cache resolution
# ---------------------------------------------------------------------------


def hf_model_dir(hf_home: Path, model_name: str) -> Path:
    """Map ``Org/Name`` to ``hf_home / models--Org--Name``."""
    safe = model_name.replace("/", "--")
    return hf_home / f"models--{safe}"


def latest_snapshot(model_dir: Path) -> Path:
    """Return the most-recently-modified snapshot dir under ``model_dir``."""
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"No snapshots dir under {model_dir}")
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No snapshots in {snapshots}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_snapshot_files(snapshot: Path) -> tuple[list[Path], list[Path]]:
    """Return (shard_files, small_files) under a snapshot dir.

    Both lists hold absolute paths to the symlinks (not the underlying blobs).
    Symlinks are kept so the caller can resolve them via ``Path.resolve``.
    """
    shards: list[Path] = []
    smalls: list[Path] = []
    for entry in sorted(snapshot.iterdir()):
        name = entry.name
        if name.startswith("."):
            continue
        if name.endswith(".safetensors") and "model-" in name:
            shards.append(entry)
        elif name in SMALL_FILE_NAMES:
            smalls.append(entry)
        else:
            for pattern in EXTRA_FILE_GLOBS:
                if entry.match(pattern):
                    smalls.append(entry)
                    break
    if not shards:
        raise FileNotFoundError(f"No model-*.safetensors shards in {snapshot}")
    return shards, smalls


# ---------------------------------------------------------------------------
# Build staging
# ---------------------------------------------------------------------------


def short_model_name(model_name: str) -> str:
    """Drop the org prefix; keep the bare model name for use as a path."""
    return model_name.split("/", 1)[-1]


def hf_cache_subpath(model_name: str, snapshot_hash: str) -> str:
    """Return the HF cache subpath under ``$HF_HOME/hub`` for a snapshot.

    Example:
        ``Qwen/Qwen3-Coder-Next-FP8`` →
        ``models--Qwen--Qwen3-Coder-Next-FP8/snapshots/<hash>``
    """
    safe = model_name.replace("/", "--")
    return f"models--{safe}/snapshots/{snapshot_hash}"


def populate_staging(
    snapshot: Path,
    shards: list[Path],
    smalls: list[Path],
    staging: Path,
) -> None:
    """Hardlink shards + small files from the HF cache into ``staging``.

    Hardlinks work because staging is on the same filesystem as the model
    cache. They cost 0 bytes and ``docker build`` reads through them
    transparently. Files land flat at the top of ``staging``; the Dockerfile
    rewrites the path during ``COPY`` so the in-image layout matches the
    standard HF cache shape.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for src in shards + smalls:
        target = src.resolve()  # follow the symlink to the blob
        if not target.is_file():
            raise FileNotFoundError(
                f"Symlink {src} resolves to {target}, which is not a file"
            )
        dst = staging / src.name
        os.link(target, dst)


# ---------------------------------------------------------------------------
# Dockerfile generation
# ---------------------------------------------------------------------------


def generate_dockerfile(
    model_name: str,
    snapshot_hash: str,
    shards: list[Path],
    smalls: list[Path],
    *,
    base_image: str = BASE_IMAGE,
    max_layers: int = 80,
) -> str:
    """Build the Dockerfile contents for one model.

    Files are placed in the standard HuggingFace hub cache layout under
    ``/models/hub/models--Org--Name/snapshots/<hash>/``, plus a
    ``refs/main`` pointer file. The image sets ``HF_HOME=/models`` so that
    later passing the HF model name (e.g. ``Qwen/Qwen3-Coder-Next-FP8``)
    via ``MODEL_NAME`` resolves to the baked snapshot without any HF
    download. ``MODEL_NAME`` is intentionally NOT set in this image — the
    eval launcher's ``env_overrides`` provides it from configs.py, keeping
    result-file model identifiers on their HF names.

    When the shard count exceeds ``max_layers`` (default 80, chosen to stay
    safely under Docker legacy builder's ~127 overlay layer limit after
    accounting for base-image layers), shards are grouped into multi-file
    COPY instructions. Each group still creates one layer, but contains
    multiple shards. Push reliability degrades slightly (bigger layers)
    but avoids the "max depth exceeded" build failure.
    """
    snapshot_dir = f"/models/hub/{hf_cache_subpath(model_name, snapshot_hash)}"
    refs_dir = f"/models/hub/models--{model_name.replace('/', '--')}/refs"

    lines: list[str] = []
    lines.append("# Auto-generated by bake_model_image.py — do not edit by hand.")
    lines.append("# Files land in the standard HF hub cache layout so the eval")
    lines.append("# launcher can keep using the HF model name as MODEL_NAME.")
    lines.append(f"FROM {base_image}")
    lines.append("")
    lines.append("ENV HF_HOME=/models")
    lines.append("ENV TRANSFORMERS_OFFLINE=1")
    lines.append("ENV HF_HUB_OFFLINE=1")
    lines.append(f"RUN mkdir -p {snapshot_dir} {refs_dir} && \\")
    lines.append(f"    echo -n '{snapshot_hash}' > {refs_dir}/main")
    lines.append("")

    if smalls:
        small_names = " ".join(sorted(p.name for p in smalls))
        lines.append("# Small files (config, tokenizer, parser plugins) in one layer")
        lines.append(f"COPY {small_names} {snapshot_dir}/")
        lines.append("")

    sorted_shards = sorted(shards, key=lambda p: p.name)
    if len(sorted_shards) <= max_layers:
        lines.append(
            f"# {len(sorted_shards)} safetensors shards, "
            f"one layer each (~2 GB / layer)"
        )
        for shard in sorted_shards:
            lines.append(f"COPY {shard.name} {snapshot_dir}/")
    else:
        # Group shards into batches to stay under the overlay layer limit.
        import math

        batch_size = math.ceil(len(sorted_shards) / max_layers)
        batches = [
            sorted_shards[i : i + batch_size]
            for i in range(0, len(sorted_shards), batch_size)
        ]
        lines.append(
            f"# {len(sorted_shards)} safetensors shards in "
            f"{len(batches)} batched layers ({batch_size} shards/layer) "
            f"to stay under overlay2's ~127-layer limit"
        )
        for batch in batches:
            names = " ".join(s.name for s in batch)
            lines.append(f"COPY {names} {snapshot_dir}/")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build / push
# ---------------------------------------------------------------------------


def docker_build(staging: Path, tag: str, *, log_path: Path | None = None) -> None:
    """Run ``docker build`` against the staging dir.

    Forces the legacy (non-BuildKit) builder via ``DOCKER_BUILDKIT=0``: the
    host's docker has BuildKit enabled by default but the buildx component
    is missing, which causes BuildKit to error out immediately. The legacy
    builder is sufficient for the COPY-only Dockerfile this script
    generates.
    """
    cmd = [
        "docker",
        "build",
        "-f",
        str(staging / "Dockerfile.shards"),
        "-t",
        tag,
        str(staging),
    ]
    print(f"+ {' '.join(cmd)}")
    env = {**os.environ, "DOCKER_BUILDKIT": "0"}
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    else:
        result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(f"docker build failed with exit code {result.returncode}")


def docker_push(tag: str, *, log_path: Path | None = None) -> None:
    """Run ``docker push`` for the tag."""
    cmd = ["docker", "push", tag]
    print(f"+ {' '.join(cmd)}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"docker push failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model name, e.g. Qwen/Qwen3-Coder-Next-FP8",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Docker tag to build, e.g. leosiriusdawn/runpod-vllm:qwen3-coder-next-fp8-baked",
    )
    p.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"HF cache root (default: {DEFAULT_HF_HOME})",
    )
    p.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help=f"Build staging dir parent (default: {DEFAULT_STAGING_ROOT})",
    )
    p.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Override snapshot dir (default: latest under --hf-home)",
    )
    p.add_argument("--build", action="store_true", help="Run docker build")
    p.add_argument("--push", action="store_true", help="Run docker push (implies --build)")
    p.add_argument(
        "--build-log",
        type=Path,
        default=None,
        help="Write docker build output to this file instead of stdout",
    )
    p.add_argument(
        "--push-log",
        type=Path,
        default=None,
        help="Write docker push output to this file instead of stdout",
    )
    args = p.parse_args()

    if args.snapshot is not None:
        snapshot = args.snapshot
    else:
        model_dir = hf_model_dir(args.hf_home, args.model)
        if not model_dir.is_dir():
            print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
            print(
                f"  hint: run `huggingface-cli download {args.model}` first",
                file=sys.stderr,
            )
            return 2
        snapshot = latest_snapshot(model_dir)

    print(f"snapshot: {snapshot}")
    shards, smalls = resolve_snapshot_files(snapshot)
    print(f"  {len(shards)} safetensors shards")
    print(f"  {len(smalls)} small files: {', '.join(p.name for p in smalls)}")

    short = short_model_name(args.model)
    staging = args.staging_root / short
    print(f"staging: {staging}")
    populate_staging(snapshot, shards, smalls, staging)

    snapshot_hash = snapshot.name
    dockerfile = generate_dockerfile(args.model, snapshot_hash, shards, smalls)
    dockerfile_path = staging / "Dockerfile.shards"
    dockerfile_path.write_text(dockerfile)
    print(f"wrote: {dockerfile_path} ({len(shards) + 1} layers + base)")

    if args.build or args.push:
        docker_build(staging, args.tag, log_path=args.build_log)
    else:
        print()
        print("Build it manually with:")
        print(f"  docker build -f {dockerfile_path} -t {args.tag} {staging}")

    if args.push:
        docker_push(args.tag, log_path=args.push_log)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
