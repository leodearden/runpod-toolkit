"""
RunPod S3-compatible API client for network volume access.

Provides direct file access to RunPod Network Volumes via S3 API:
- Upload/download files directly to volumes
- List and manage objects in volumes
- No need for SSH/rsync - works from anywhere

Endpoint format: https://s3api-{datacenter}.runpod.io/
Bucket = Network Volume ID
"""

import logging
from pathlib import Path
from typing import Iterator, Optional, Callable

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from runpod_toolkit.config import RunPodS3Config, ConfigurationError
from runpod_toolkit.storage.b2_client import ObjectInfo, UploadResult

logger = logging.getLogger(__name__)


class RunPodS3Client:
    """
    S3-compatible client for RunPod Network Volumes.

    Uses boto3 with S3-compatible API to access network volumes.
    Each volume is accessed as a "bucket" with its volume ID.
    """

    def __init__(self, config: RunPodS3Config, datacenter: str):
        """
        Initialize RunPod S3 client.

        Args:
            config: RunPod S3 configuration with credentials
            datacenter: Datacenter ID (e.g., 'EU-RO-1')

        Raises:
            ConfigurationError: If boto3 is not installed
        """
        if not HAS_BOTO3:
            raise ConfigurationError(
                "boto3 is not installed. Run: pip install boto3"
            )

        self._config = config
        self._datacenter = datacenter
        self._endpoint = config.get_endpoint(datacenter)
        self._client = self._create_client()

    def _create_client(self):
        """Create boto3 client with RunPod S3 settings."""
        return boto3.client(
            's3',
            endpoint_url=self._endpoint,
            aws_access_key_id=self._config.access_key,
            aws_secret_access_key=self._config.secret_key,
            region_name=self._datacenter.lower(),
            config=Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive',
                },
                max_pool_connections=50,
            )
        )

    @property
    def endpoint(self) -> str:
        """S3 endpoint URL."""
        return self._endpoint

    @property
    def datacenter(self) -> str:
        """Datacenter ID."""
        return self._datacenter

    def upload_file(
        self,
        local_path: Path,
        volume_id: str,
        remote_key: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> UploadResult:
        """
        Upload single file to a network volume.

        Args:
            local_path: Path to local file
            volume_id: Network volume ID (used as bucket)
            remote_key: S3 key (path in volume). Defaults to filename.
            progress_callback: Called with bytes uploaded

        Returns:
            UploadResult with key, size, etag, s3_uri

        Raises:
            FileNotFoundError: If local file doesn't exist
            ClientError: On upload failure
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        # Default remote key to filename
        if remote_key is None:
            remote_key = local_path.name

        file_size = local_path.stat().st_size

        logger.info(f"Uploading {local_path} ({file_size / 1024 / 1024 / 1024:.2f} GB) to {volume_id}/{remote_key}")

        # Progress callback wrapper
        class ProgressCallback:
            def __init__(self, callback):
                self.callback = callback
                self.bytes_transferred = 0

            def __call__(self, bytes_amount):
                self.bytes_transferred += bytes_amount
                if self.callback:
                    self.callback(self.bytes_transferred)

        callback = ProgressCallback(progress_callback) if progress_callback else None

        # Upload with optional progress
        if callback:
            self._client.upload_file(
                str(local_path),
                volume_id,
                remote_key,
                Callback=callback,
            )
        else:
            self._client.upload_file(
                str(local_path),
                volume_id,
                remote_key,
            )

        # Get uploaded object info
        response = self._client.head_object(Bucket=volume_id, Key=remote_key)

        return UploadResult(
            key=remote_key,
            size=file_size,
            etag=response['ETag'].strip('"'),
            s3_uri=f"s3://{volume_id}/{remote_key}",
        )

    def download_file(
        self,
        volume_id: str,
        remote_key: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Path:
        """
        Download single file from a network volume.

        Args:
            volume_id: Network volume ID (used as bucket)
            remote_key: S3 key (path in volume)
            local_path: Path to save file
            progress_callback: Called with bytes downloaded

        Returns:
            Path to downloaded file

        Raises:
            ClientError: On download failure (e.g., key not found)
        """
        local_path = Path(local_path)

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {volume_id}/{remote_key} to {local_path}")

        # Progress callback wrapper
        class ProgressCallback:
            def __init__(self, callback):
                self.callback = callback
                self.bytes_transferred = 0

            def __call__(self, bytes_amount):
                self.bytes_transferred += bytes_amount
                if self.callback:
                    self.callback(self.bytes_transferred)

        callback = ProgressCallback(progress_callback) if progress_callback else None

        # Download with optional progress
        if callback:
            self._client.download_file(
                volume_id,
                remote_key,
                str(local_path),
                Callback=callback,
            )
        else:
            self._client.download_file(
                volume_id,
                remote_key,
                str(local_path),
            )

        return local_path

    def list_objects(
        self,
        volume_id: str,
        prefix: str = '',
        max_keys: int = 1000
    ) -> Iterator[ObjectInfo]:
        """
        List objects in a network volume.

        Args:
            volume_id: Network volume ID (used as bucket)
            prefix: Prefix to filter by
            max_keys: Maximum keys per request

        Yields:
            ObjectInfo for each object

        Note:
            RunPod's S3-compatible API has a pagination bug where continuation
            tokens don't work correctly, causing pages to overlap. This method
            uses client-side deduplication with StartAfter pagination as a
            workaround.
        """
        seen_keys: set[str] = set()
        start_after: str | None = None

        while True:
            kwargs = {
                'Bucket': volume_id,
                'MaxKeys': max_keys,
            }
            if prefix:
                kwargs['Prefix'] = prefix
            if start_after:
                kwargs['StartAfter'] = start_after

            response = self._client.list_objects_v2(**kwargs)
            contents = response.get('Contents', [])

            if not contents:
                break

            new_objects_in_page = 0
            for obj in contents:
                key = obj['Key']
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_objects_in_page += 1
                    yield ObjectInfo(
                        key=key,
                        size=obj['Size'],
                        etag=obj['ETag'].strip('"'),
                        last_modified=str(obj['LastModified']),
                    )

            # If no new objects were found, we've seen everything
            if new_objects_in_page == 0:
                break

            # If not truncated, we're done
            if not response.get('IsTruncated', False):
                break

            # Use last key as StartAfter for next page
            start_after = contents[-1]['Key']

    def get_object_metadata(self, volume_id: str, key: str) -> Optional[ObjectInfo]:
        """
        Get object metadata (size, etag, modified time).

        Args:
            volume_id: Network volume ID
            key: S3 key

        Returns:
            ObjectInfo if exists, None otherwise
        """
        try:
            response = self._client.head_object(Bucket=volume_id, Key=key)
            return ObjectInfo(
                key=key,
                size=response['ContentLength'],
                etag=response['ETag'].strip('"'),
                last_modified=str(response['LastModified']),
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise

    def object_exists(self, volume_id: str, key: str) -> bool:
        """
        Check if object exists in volume.

        Args:
            volume_id: Network volume ID
            key: S3 key

        Returns:
            True if object exists
        """
        return self.get_object_metadata(volume_id, key) is not None

    def delete_object(self, volume_id: str, key: str) -> bool:
        """
        Delete an object from volume.

        Args:
            volume_id: Network volume ID
            key: S3 key

        Returns:
            True if deleted successfully
        """
        try:
            self._client.delete_object(Bucket=volume_id, Key=key)
            logger.info(f"Deleted {volume_id}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {volume_id}/{key}: {e}")
            raise

    def get_volume_size(self, volume_id: str, prefix: str = '') -> dict:
        """
        Calculate total size of objects in volume.

        Args:
            volume_id: Network volume ID
            prefix: Prefix to calculate size for

        Returns:
            Dict with 'total_bytes', 'total_mb', 'total_gb', 'object_count'
        """
        total_bytes = 0
        object_count = 0

        for obj in self.list_objects(volume_id, prefix):
            total_bytes += obj.size
            object_count += 1

        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'total_gb': total_bytes / (1024 * 1024 * 1024),
            'object_count': object_count,
        }

    def test_connection(self, volume_id: str) -> bool:
        """
        Test connection to volume by listing objects.

        Args:
            volume_id: Network volume ID to test

        Returns:
            True if connection successful

        Raises:
            ClientError: On connection failure
        """
        try:
            # Try to list objects (even empty volume works)
            self._client.list_objects_v2(Bucket=volume_id, MaxKeys=1)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchBucket':
                raise ConfigurationError(
                    f"Volume '{volume_id}' not found in datacenter '{self._datacenter}'.\n"
                    f"Ensure the volume exists and is in the correct datacenter."
                )
            raise
