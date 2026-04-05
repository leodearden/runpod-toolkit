"""
Backblaze B2 client using S3-compatible API.

Provides high-level operations for:
- Uploading/downloading files and directories
- Listing and managing objects
- Generating presigned URLs for direct pod access
"""

import logging
import hashlib
import os
from pathlib import Path
from typing import Iterator, Optional, List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from runpod_toolkit.config import B2Config, ConfigurationError

logger = logging.getLogger(__name__)


# Multipart upload threshold (100MB)
MULTIPART_THRESHOLD = 100 * 1024 * 1024

# Default chunk size for multipart uploads (8MB)
MULTIPART_CHUNK_SIZE = 8 * 1024 * 1024

# Default number of parallel transfers
DEFAULT_PARALLEL_TRANSFERS = 8


@dataclass
class ObjectInfo:
    """Information about a B2 object."""
    key: str
    size: int
    etag: str
    last_modified: str

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size / (1024 * 1024)


@dataclass
class UploadResult:
    """Result of an upload operation."""
    key: str
    size: int
    etag: str
    s3_uri: str


class B2Client:
    """
    Thread-safe B2 client with connection pooling.

    Uses boto3 with S3-compatible API for B2 operations.
    """

    def __init__(self, config: B2Config):
        """
        Initialize B2 client with validated config.

        Args:
            config: B2 configuration

        Raises:
            ConfigurationError: If boto3 is not installed
        """
        if not HAS_BOTO3:
            raise ConfigurationError(
                "boto3 is not installed. Run: pip install boto3"
            )

        self._config = config
        self._client = self._create_client()
        self._bucket = config.bucket

    def _create_client(self):
        """Create boto3 client with B2-optimized settings."""
        return boto3.client(
            's3',
            endpoint_url=self._config.s3_endpoint_url,
            aws_access_key_id=self._config.key_id,
            aws_secret_access_key=self._config.application_key,
            config=Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive',
                },
                max_pool_connections=50,
            )
        )

    def upload_file(
        self,
        local_path: Path,
        remote_key: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> UploadResult:
        """
        Upload single file to B2.

        Uses multipart upload for files >100MB.

        Args:
            local_path: Path to local file
            remote_key: S3 key (path in bucket)
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

        file_size = local_path.stat().st_size

        logger.debug(f"Uploading {local_path} ({file_size / 1024 / 1024:.1f} MB) to {remote_key}")

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
        extra_args = {}
        if callback:
            self._client.upload_file(
                str(local_path),
                self._bucket,
                remote_key,
                Callback=callback,
                ExtraArgs=extra_args,
            )
        else:
            self._client.upload_file(
                str(local_path),
                self._bucket,
                remote_key,
                ExtraArgs=extra_args,
            )

        # Get uploaded object info
        response = self._client.head_object(Bucket=self._bucket, Key=remote_key)

        return UploadResult(
            key=remote_key,
            size=file_size,
            etag=response['ETag'].strip('"'),
            s3_uri=f"s3://{self._bucket}/{remote_key}",
        )

    def download_file(
        self,
        remote_key: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Path:
        """
        Download single file from B2.

        Args:
            remote_key: S3 key (path in bucket)
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

        logger.debug(f"Downloading {remote_key} to {local_path}")

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
                self._bucket,
                remote_key,
                str(local_path),
                Callback=callback,
            )
        else:
            self._client.download_file(
                self._bucket,
                remote_key,
                str(local_path),
            )

        return local_path

    def upload_directory(
        self,
        local_dir: Path,
        remote_prefix: str,
        exclude_patterns: Optional[List[str]] = None,
        parallel: int = DEFAULT_PARALLEL_TRANSFERS,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> int:
        """
        Upload directory recursively to B2.

        Args:
            local_dir: Local directory to upload
            remote_prefix: Prefix (folder) in B2
            exclude_patterns: Glob patterns to exclude (e.g., ['*.pyc', '__pycache__/*'])
            parallel: Number of parallel uploads
            progress_callback: Called with (filename, file_index, total_files)

        Returns:
            Number of files uploaded

        Raises:
            FileNotFoundError: If local directory doesn't exist
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        exclude_patterns = exclude_patterns or []

        # Collect files to upload
        files_to_upload = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Check exclusions
                relative_path = file_path.relative_to(local_dir)
                excluded = False
                for pattern in exclude_patterns:
                    if relative_path.match(pattern):
                        excluded = True
                        break
                if not excluded:
                    remote_key = f"{remote_prefix}/{relative_path}".replace('\\', '/')
                    files_to_upload.append((file_path, remote_key))

        total_files = len(files_to_upload)
        logger.info(f"Uploading {total_files} files from {local_dir} to {remote_prefix}/")

        if total_files == 0:
            return 0

        uploaded = 0

        # Upload in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(self.upload_file, local_path, remote_key): (local_path, remote_key)
                for local_path, remote_key in files_to_upload
            }

            for i, future in enumerate(as_completed(futures)):
                local_path, remote_key = futures[future]
                try:
                    future.result()
                    uploaded += 1
                    if progress_callback:
                        progress_callback(str(local_path.name), i + 1, total_files)
                except Exception as e:
                    logger.error(f"Failed to upload {local_path}: {e}")
                    raise

        return uploaded

    def download_directory(
        self,
        remote_prefix: str,
        local_dir: Path,
        parallel: int = DEFAULT_PARALLEL_TRANSFERS,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> int:
        """
        Download directory from B2 with parallel transfers.

        Args:
            remote_prefix: Prefix (folder) in B2
            local_dir: Local directory to download to
            parallel: Number of parallel downloads
            progress_callback: Called with (filename, file_index, total_files)

        Returns:
            Number of files downloaded
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # List objects under prefix
        objects = list(self.list_objects(remote_prefix))
        total_files = len(objects)

        logger.info(f"Downloading {total_files} files from {remote_prefix}/ to {local_dir}")

        if total_files == 0:
            return 0

        downloaded = 0

        # Prepare download tasks
        download_tasks = []
        for obj in objects:
            # Compute local path relative to prefix
            relative_path = obj.key[len(remote_prefix):].lstrip('/')
            local_path = local_dir / relative_path
            download_tasks.append((obj.key, local_path))

        # Download in parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(self.download_file, remote_key, local_path): (remote_key, local_path)
                for remote_key, local_path in download_tasks
            }

            for i, future in enumerate(as_completed(futures)):
                remote_key, local_path = futures[future]
                try:
                    future.result()
                    downloaded += 1
                    if progress_callback:
                        progress_callback(local_path.name, i + 1, total_files)
                except Exception as e:
                    logger.error(f"Failed to download {remote_key}: {e}")
                    raise

        return downloaded

    def list_objects(
        self,
        prefix: str = '',
        max_keys: int = 1000
    ) -> Iterator[ObjectInfo]:
        """
        List objects under prefix with pagination.

        Args:
            prefix: Prefix to filter by
            max_keys: Maximum keys per request

        Yields:
            ObjectInfo for each object
        """
        paginator = self._client.get_paginator('list_objects_v2')

        for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=prefix,
            PaginationConfig={'PageSize': max_keys}
        ):
            for obj in page.get('Contents', []):
                yield ObjectInfo(
                    key=obj['Key'],
                    size=obj['Size'],
                    etag=obj['ETag'].strip('"'),
                    last_modified=str(obj['LastModified']),
                )

    def get_presigned_url(
        self,
        key: str,
        expires_in: int = 3600,
        operation: str = 'get_object'
    ) -> str:
        """
        Generate presigned URL for direct access.

        Args:
            key: S3 key
            expires_in: URL expiration time in seconds (default: 1 hour)
            operation: 'get_object' for download, 'put_object' for upload

        Returns:
            Presigned URL
        """
        return self._client.generate_presigned_url(
            operation,
            Params={'Bucket': self._bucket, 'Key': key},
            ExpiresIn=expires_in,
        )

    def delete_objects(
        self,
        prefix: str,
        dry_run: bool = True
    ) -> int:
        """
        Delete all objects under prefix.

        Args:
            prefix: Prefix to delete
            dry_run: If True, only count objects without deleting

        Returns:
            Number of objects deleted (or would be deleted)
        """
        objects = list(self.list_objects(prefix))
        count = len(objects)

        if dry_run:
            logger.info(f"Would delete {count} objects under {prefix}/")
            return count

        if count == 0:
            return 0

        # Delete in batches of 1000 (S3 limit)
        for i in range(0, count, 1000):
            batch = objects[i:i + 1000]
            delete_objects = {'Objects': [{'Key': obj.key} for obj in batch]}

            self._client.delete_objects(
                Bucket=self._bucket,
                Delete=delete_objects,
            )

        logger.info(f"Deleted {count} objects under {prefix}/")
        return count

    def get_object_metadata(self, key: str) -> Optional[ObjectInfo]:
        """
        Get object metadata (size, etag, modified time).

        Args:
            key: S3 key

        Returns:
            ObjectInfo if exists, None otherwise
        """
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
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

    def object_exists(self, key: str) -> bool:
        """
        Check if object exists.

        Args:
            key: S3 key

        Returns:
            True if object exists
        """
        return self.get_object_metadata(key) is not None

    def get_bucket_size(self, prefix: str = '') -> dict:
        """
        Calculate total size of objects under prefix.

        Args:
            prefix: Prefix to calculate size for

        Returns:
            Dict with 'total_bytes', 'total_mb', 'object_count'
        """
        total_bytes = 0
        object_count = 0

        for obj in self.list_objects(prefix):
            total_bytes += obj.size
            object_count += 1

        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'total_gb': total_bytes / (1024 * 1024 * 1024),
            'object_count': object_count,
        }
