"""Health and readiness probes for vLLM endpoints."""
from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def check_health(host: str, ssh_port: int, api_port: int = 8000) -> tuple[bool, str]:
    """Check vLLM health via SSH tunnel to localhost:{api_port}/health.

    Returns (is_healthy, detail_message).
    """
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-p", str(ssh_port), f"root@{host}",
        f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{api_port}/health",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        code = result.stdout.strip().strip("'")
        if code == "200":
            return True, "healthy"
        return False, f"HTTP {code}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def list_models(host: str, ssh_port: int, api_port: int = 8000) -> list[str]:
    """Query /v1/models via SSH and return loaded model IDs."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-p", str(ssh_port), f"root@{host}",
        f"curl -s http://localhost:{api_port}/v1/models",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
