"""
Configuration management for cloud infrastructure.

Loads configuration from:
1. Environment variables
2. ~/.secrets/b2.env and ~/.secrets/runpod.env files
3. Optional config override

Provides validated config objects for B2, RunPod, and cloud settings.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


def _load_dotenv(env_file: Path) -> Dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_file: Path to .env file

    Returns:
        Dictionary of environment variables

    Raises:
        ConfigurationError: If file doesn't exist
    """
    if not env_file.exists():
        raise ConfigurationError(
            f"Credentials file not found: {env_file}\n"
            f"Please create this file with the required credentials."
        )

    env_vars = {}
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                env_vars[key] = value

    return env_vars


@dataclass
class B2Config:
    """Backblaze B2 configuration (cold storage)."""

    key_id: str
    application_key: str
    endpoint: str
    bucket: str
    region: str

    @property
    def s3_endpoint_url(self) -> str:
        """Full S3-compatible endpoint URL."""
        return f"https://{self.endpoint}"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "B2Config":
        """Load B2 configuration from environment or .env file.

        Args:
            env_file: Path to .env file (default: ~/.secrets/b2.env)

        Returns:
            B2Config instance

        Raises:
            ConfigurationError: If required credentials are missing
        """
        if env_file is None:
            env_file = Path.home() / ".secrets" / "b2.env"

        env_vars = {}
        if env_file.exists():
            env_vars = _load_dotenv(env_file)
            logger.debug(f"Loaded B2 config from {env_file}")
        else:
            logger.debug(f"B2 env file not found at {env_file}, using environment")

        def get_var(name: str, required: bool = True) -> Optional[str]:
            value = env_vars.get(name) or os.environ.get(name)
            if required and not value:
                raise ConfigurationError(
                    f"Missing required B2 configuration: {name}\n"
                    f"Set in {env_file} or as environment variable."
                )
            return value

        return cls(
            key_id=get_var("B2_KEY_ID"),
            application_key=get_var("B2_APPLICATION_KEY"),
            endpoint=get_var("B2_ENDPOINT"),
            bucket=get_var("B2_BUCKET"),
            region=get_var("B2_REGION"),
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary (for serialization)."""
        return asdict(self)

    def to_redacted_dict(self) -> Dict[str, str]:
        """Convert to dictionary with secrets redacted."""
        return {
            "key_id": self.key_id[:4] + "****" if self.key_id else None,
            "application_key": "****" if self.application_key else None,
            "endpoint": self.endpoint,
            "bucket": self.bucket,
            "region": self.region,
        }


@dataclass
class RunPodS3Config:
    """RunPod S3 API configuration for network volume access."""

    access_key: str
    secret_key: str

    def get_endpoint(self, datacenter: str) -> str:
        """Get S3 endpoint URL for datacenter."""
        return f"https://s3api-{datacenter.lower()}.runpod.io"

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "RunPodS3Config":
        """Load RunPod S3 configuration from environment or .env file.

        Args:
            env_file: Path to .env file (default: ~/.secrets/runpod_s3.env)

        Returns:
            RunPodS3Config instance

        Raises:
            ConfigurationError: If required credentials are missing
        """
        if env_file is None:
            env_file = Path.home() / ".secrets" / "runpod_s3.env"

        env_vars = {}
        if env_file.exists():
            env_vars = _load_dotenv(env_file)
            logger.debug(f"Loaded RunPod S3 config from {env_file}")
        else:
            logger.debug(
                f"RunPod S3 env file not found at {env_file}, using environment"
            )

        def get_var(name: str, required: bool = True) -> Optional[str]:
            value = env_vars.get(name) or os.environ.get(name)
            if required and not value:
                raise ConfigurationError(
                    f"Missing required RunPod S3 configuration: {name}\n"
                    f"Set in {env_file} or as environment variable.\n"
                    f"Create S3 API keys at: RunPod Console → Settings → S3 API Keys"
                )
            return value

        return cls(
            access_key=get_var("RUNPOD_S3_ACCESS_KEY"),
            secret_key=get_var("RUNPOD_S3_SECRET_KEY"),
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary (for serialization)."""
        return asdict(self)

    def to_redacted_dict(self) -> Dict[str, str]:
        """Convert to dictionary with secrets redacted."""
        return {
            "access_key": self.access_key[:8] + "****" if self.access_key else None,
            "secret_key": "****" if self.secret_key else None,
        }


@dataclass
class RunPodConfig:
    """RunPod configuration."""

    api_key: str
    template_id: Optional[str] = None
    default_gpu_type: str = "NVIDIA RTX A5000"
    default_datacenter: str = "EU-RO-1"
    cloud_type: str = "SECURE"  # SECURE or COMMUNITY
    container_disk_gb: int = 20
    default_network_volume_id: Optional[str] = None  # Network volume to attach by default

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "RunPodConfig":
        """Load RunPod configuration from environment or .env file.

        Args:
            env_file: Path to .env file (default: ~/.secrets/runpod.env)

        Returns:
            RunPodConfig instance

        Raises:
            ConfigurationError: If required credentials are missing
        """
        if env_file is None:
            env_file = Path.home() / ".secrets" / "runpod.env"

        env_vars = {}
        if env_file.exists():
            env_vars = _load_dotenv(env_file)
            logger.debug(f"Loaded RunPod config from {env_file}")
        else:
            logger.debug(f"RunPod env file not found at {env_file}, using environment")

        def get_var(name: str, required: bool = True, default: Any = None) -> Any:
            value = env_vars.get(name) or os.environ.get(name)
            if required and not value and default is None:
                raise ConfigurationError(
                    f"Missing required RunPod configuration: {name}\n"
                    f"Set in {env_file} or as environment variable."
                )
            return value if value else default

        return cls(
            api_key=get_var("RUNPOD_API_KEY"),
            template_id=get_var("RUNPOD_TEMPLATE_ID", required=False),
            default_gpu_type=get_var(
                "RUNPOD_DEFAULT_GPU", required=False, default="NVIDIA RTX A5000"
            ),
            default_datacenter=get_var(
                "RUNPOD_DEFAULT_DATACENTER", required=False, default="EU-RO-1"
            ),
            cloud_type=get_var("RUNPOD_CLOUD_TYPE", required=False, default="SECURE"),
            container_disk_gb=int(
                get_var("RUNPOD_CONTAINER_DISK_GB", required=False, default="20")
            ),
            default_network_volume_id=get_var("RUNPOD_DEFAULT_VOLUME_ID", required=False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return asdict(self)

    def to_redacted_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with secrets redacted."""
        return {
            "api_key": self.api_key[:8] + "****" if self.api_key else None,
            "template_id": self.template_id,
            "default_gpu_type": self.default_gpu_type,
            "default_datacenter": self.default_datacenter,
            "cloud_type": self.cloud_type,
            "container_disk_gb": self.container_disk_gb,
            "default_network_volume_id": self.default_network_volume_id,
        }


@dataclass
class CloudConfig:
    """Combined cloud configuration."""

    b2: B2Config
    runpod: RunPodConfig

    # Local paths
    local_cache_dir: Path = field(default_factory=lambda: Path("/tmp/runpod_toolkit_cache"))
    local_backup_dir: Optional[Path] = None

    # B2 prefixes
    b2_cache_prefix: str = "cache"
    b2_results_prefix: str = "results"
    b2_experiments_prefix: str = "experiments"
    b2_code_prefix: str = "code"

    # Pod paths
    pod_workspace: str = "/workspace"
    pod_cache_dir: str = "/workspace/cache"
    pod_code_dir: str = "/root/code"
    pod_outputs_dir: str = "/workspace/outputs"
    pod_checkpoints_dir: str = "/workspace/checkpoints"

    # State file for tracking active volume
    state_file: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "runpod_toolkit" / "cloud_state.json"
    )

    @classmethod
    def load(
        cls,
        b2_env_file: Optional[Path] = None,
        runpod_env_file: Optional[Path] = None,
    ) -> "CloudConfig":
        """Load complete cloud configuration.

        Args:
            b2_env_file: Path to B2 credentials file
            runpod_env_file: Path to RunPod credentials file

        Returns:
            CloudConfig instance

        Raises:
            ConfigurationError: If required credentials are missing
        """
        b2_config = B2Config.from_env(b2_env_file)
        runpod_config = RunPodConfig.from_env(runpod_env_file)

        return cls(b2=b2_config, runpod=runpod_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        result = asdict(self)
        # Convert Path objects to strings
        for key in ["local_cache_dir", "local_backup_dir", "state_file"]:
            if key in result and result[key]:
                result[key] = str(result[key])
        return result

    def to_redacted_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with secrets redacted."""
        return {
            "b2": self.b2.to_redacted_dict(),
            "runpod": self.runpod.to_redacted_dict(),
            "local_cache_dir": str(self.local_cache_dir),
            "local_backup_dir": str(self.local_backup_dir) if self.local_backup_dir else None,
            "b2_cache_prefix": self.b2_cache_prefix,
            "b2_results_prefix": self.b2_results_prefix,
            "pod_workspace": self.pod_workspace,
        }

    # === State Management ===

    def get_active_volume(self) -> Optional[Dict[str, str]]:
        """Get currently active Network Volume, if any.

        Returns:
            Dict with 'volume_id' and 'datacenter_id', or None
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            return state.get("active_volume")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read state file: {e}")
            return None

    def set_active_volume(self, volume_id: str, datacenter_id: str) -> None:
        """Set the active Network Volume for this burst period.

        Args:
            volume_id: RunPod Network Volume ID
            datacenter_id: Datacenter where volume is located
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {}
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        state["active_volume"] = {
            "volume_id": volume_id,
            "datacenter_id": datacenter_id,
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Set active volume: {volume_id} in {datacenter_id}")

    def clear_active_volume(self) -> None:
        """Clear active volume (end of burst)."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        if "active_volume" in state:
            del state["active_volume"]
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("Cleared active volume")


# Module-level config cache
_config_cache: Optional[CloudConfig] = None


def load_cloud_config(
    b2_env_file: Optional[Path] = None,
    runpod_env_file: Optional[Path] = None,
    force_reload: bool = False,
) -> CloudConfig:
    """Load cloud configuration (with caching).

    Args:
        b2_env_file: Path to B2 credentials file
        runpod_env_file: Path to RunPod credentials file
        force_reload: Force reload even if cached

    Returns:
        CloudConfig instance

    Raises:
        ConfigurationError: If required credentials are missing
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    _config_cache = CloudConfig.load(b2_env_file, runpod_env_file)
    return _config_cache


def validate_b2_config(config: B2Config) -> bool:
    """Validate B2 configuration by attempting to list buckets.

    Args:
        config: B2 configuration to validate

    Returns:
        True if valid

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.key_id,
            aws_secret_access_key=config.application_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 1},
            ),
        )

        # Try to head the bucket (lightweight check)
        client.head_bucket(Bucket=config.bucket)
        return True

    except ImportError:
        raise ConfigurationError("boto3 is not installed. Run: pip install boto3")
    except Exception as e:
        raise ConfigurationError(f"B2 validation failed: {e}")


def validate_runpod_config(config: RunPodConfig) -> bool:
    """Validate RunPod configuration by querying the API.

    Args:
        config: RunPod configuration to validate

    Returns:
        True if valid

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        import runpod

        runpod.api_key = config.api_key

        # Try to list pods (lightweight check)
        pods = runpod.get_pods()
        # If we get here without exception, the API key is valid
        return True

    except ImportError:
        raise ConfigurationError("runpod is not installed. Run: pip install runpod")
    except Exception as e:
        raise ConfigurationError(f"RunPod validation failed: {e}")
