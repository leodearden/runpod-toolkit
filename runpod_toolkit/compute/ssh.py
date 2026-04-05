"""SSH operations for RunPod pods.

Provides SSH connectivity and command execution on remote pods.
Uses paramiko for the underlying SSH implementation.

Example:
    >>> from runpod_toolkit.compute import SSHManager
    >>>
    >>> ssh = SSHManager()
    >>> with ssh.connect("123.45.67.89", port=12345) as conn:
    ...     exit_code, stdout, stderr = ssh.run_command(conn, "nvidia-smi")
    ...     print(stdout)
"""

import logging
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, BinaryIO, Union

import paramiko

logger = logging.getLogger(__name__)


class SSHError(Exception):
    """SSH operation error."""
    pass


class SSHConnectionError(SSHError):
    """SSH connection error."""
    pass


class SSHCommandError(SSHError):
    """SSH command execution error."""

    def __init__(self, message: str, exit_code: int = -1, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class CommandResult:
    """Result of an SSH command execution."""
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0


class SSHConnection:
    """Wrapper around paramiko SSHClient for context management."""

    def __init__(self, client: paramiko.SSHClient, host: str, port: int):
        """Initialize SSH connection.

        Args:
            client: Connected paramiko SSHClient.
            host: Remote host address.
            port: SSH port.
        """
        self.client = client
        self.host = host
        self.port = port

    def close(self):
        """Close the SSH connection."""
        try:
            self.client.close()
            logger.debug(f"Closed SSH connection to {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Error closing SSH connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SSHManager:
    """Manager for SSH operations on remote pods.

    Uses paramiko for SSH connections. Supports:
    - Command execution with timeout
    - File upload/download via SFTP
    - Automatic host key handling (auto-add)
    """

    # Default timeout for SSH operations (seconds)
    DEFAULT_TIMEOUT = 60

    # Default connection timeout
    CONNECT_TIMEOUT = 30

    def __init__(
        self,
        username: str = "root",
        private_key_path: Optional[Path] = None,
        password: Optional[str] = None,
    ):
        """Initialize SSH manager.

        Args:
            username: SSH username (default: root for RunPod).
            private_key_path: Path to private key file (default: ~/.ssh/id_rsa).
            password: Optional password for key or password auth.
        """
        self.username = username
        self.password = password

        # Resolve private key path
        if private_key_path:
            self.private_key_path = Path(private_key_path)
        else:
            self.private_key_path = Path.home() / ".ssh" / "id_rsa"

        # Load private key if it exists
        self._private_key = None
        if self.private_key_path.exists():
            try:
                self._private_key = paramiko.RSAKey.from_private_key_file(
                    str(self.private_key_path),
                    password=password,
                )
                logger.debug(f"Loaded private key from {self.private_key_path}")
            except Exception as e:
                logger.warning(f"Failed to load private key: {e}")

    def connect(
        self,
        host: str,
        port: int = 22,
        timeout: Optional[int] = None,
    ) -> SSHConnection:
        """Connect to remote host.

        Args:
            host: Remote host address.
            port: SSH port.
            timeout: Connection timeout in seconds.

        Returns:
            SSHConnection context manager.

        Raises:
            SSHConnectionError: If connection fails.
        """
        timeout = timeout or self.CONNECT_TIMEOUT

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            logger.info(f"Connecting to {host}:{port}")

            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": self.username,
                "timeout": timeout,
            }

            # Use private key if available
            if self._private_key:
                connect_kwargs["pkey"] = self._private_key
            elif self.password:
                connect_kwargs["password"] = self.password
            else:
                # Try default SSH agent
                connect_kwargs["allow_agent"] = True
                connect_kwargs["look_for_keys"] = True

            client.connect(**connect_kwargs)
            logger.debug(f"SSH connection established to {host}:{port}")

            return SSHConnection(client, host, port)

        except paramiko.AuthenticationException as e:
            raise SSHConnectionError(f"Authentication failed: {e}") from e
        except paramiko.SSHException as e:
            raise SSHConnectionError(f"SSH error: {e}") from e
        except TimeoutError as e:
            raise SSHConnectionError(f"Connection timeout: {e}") from e
        except Exception as e:
            raise SSHConnectionError(f"Failed to connect to {host}:{port}: {e}") from e

    def run_command(
        self,
        connection: SSHConnection,
        command: str,
        timeout: Optional[int] = None,
        get_pty: bool = False,
    ) -> CommandResult:
        """Execute command on remote host.

        Args:
            connection: Active SSH connection.
            command: Command to execute.
            timeout: Command timeout in seconds.
            get_pty: Request a pseudo-terminal (useful for interactive commands).

        Returns:
            CommandResult with exit code, stdout, and stderr.

        Raises:
            SSHCommandError: If command execution fails.
        """
        timeout = timeout or self.DEFAULT_TIMEOUT

        try:
            logger.debug(f"Executing: {command}")

            stdin, stdout, stderr = connection.client.exec_command(
                command,
                timeout=timeout,
                get_pty=get_pty,
            )

            # Wait for command to complete
            exit_code = stdout.channel.recv_exit_status()

            # Read output
            stdout_text = stdout.read().decode("utf-8", errors="replace")
            stderr_text = stderr.read().decode("utf-8", errors="replace")

            result = CommandResult(
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

            if exit_code != 0:
                logger.debug(f"Command exited with code {exit_code}: {stderr_text[:200]}")

            return result

        except Exception as e:
            raise SSHCommandError(f"Command execution failed: {e}") from e

    def upload_file(
        self,
        connection: SSHConnection,
        local_path: Union[Path, str],
        remote_path: str,
    ) -> None:
        """Upload file to remote host via SFTP.

        Args:
            connection: Active SSH connection.
            local_path: Local file path.
            remote_path: Remote destination path.

        Raises:
            SSHError: If upload fails.
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise SSHError(f"Local file not found: {local_path}")

        try:
            sftp = connection.client.open_sftp()
            try:
                logger.debug(f"Uploading {local_path} to {remote_path}")
                sftp.put(str(local_path), remote_path)
                logger.info(f"Uploaded {local_path} to {remote_path}")
            finally:
                sftp.close()

        except Exception as e:
            raise SSHError(f"Failed to upload {local_path}: {e}") from e

    def download_file(
        self,
        connection: SSHConnection,
        remote_path: str,
        local_path: Union[Path, str],
    ) -> Path:
        """Download file from remote host via SFTP.

        Args:
            connection: Active SSH connection.
            remote_path: Remote file path.
            local_path: Local destination path.

        Returns:
            Path to downloaded file.

        Raises:
            SSHError: If download fails.
        """
        local_path = Path(local_path)

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            sftp = connection.client.open_sftp()
            try:
                logger.debug(f"Downloading {remote_path} to {local_path}")
                sftp.get(remote_path, str(local_path))
                logger.info(f"Downloaded {remote_path} to {local_path}")
                return local_path
            finally:
                sftp.close()

        except Exception as e:
            raise SSHError(f"Failed to download {remote_path}: {e}") from e

    def upload_bytes(
        self,
        connection: SSHConnection,
        data: bytes,
        remote_path: str,
    ) -> None:
        """Upload bytes to remote file via SFTP.

        Args:
            connection: Active SSH connection.
            data: Bytes to upload.
            remote_path: Remote destination path.

        Raises:
            SSHError: If upload fails.
        """
        try:
            sftp = connection.client.open_sftp()
            try:
                logger.debug(f"Uploading {len(data)} bytes to {remote_path}")
                with sftp.file(remote_path, "wb") as f:
                    f.write(data)
                logger.info(f"Uploaded {len(data)} bytes to {remote_path}")
            finally:
                sftp.close()

        except Exception as e:
            raise SSHError(f"Failed to upload to {remote_path}: {e}") from e

    def download_bytes(
        self,
        connection: SSHConnection,
        remote_path: str,
    ) -> bytes:
        """Download remote file as bytes via SFTP.

        Args:
            connection: Active SSH connection.
            remote_path: Remote file path.

        Returns:
            File contents as bytes.

        Raises:
            SSHError: If download fails.
        """
        try:
            sftp = connection.client.open_sftp()
            try:
                logger.debug(f"Downloading {remote_path}")
                with sftp.file(remote_path, "rb") as f:
                    data = f.read()
                logger.info(f"Downloaded {len(data)} bytes from {remote_path}")
                return data
            finally:
                sftp.close()

        except Exception as e:
            raise SSHError(f"Failed to download {remote_path}: {e}") from e

    def file_exists(
        self,
        connection: SSHConnection,
        remote_path: str,
    ) -> bool:
        """Check if remote file exists.

        Args:
            connection: Active SSH connection.
            remote_path: Remote file path.

        Returns:
            True if file exists.
        """
        try:
            sftp = connection.client.open_sftp()
            try:
                sftp.stat(remote_path)
                return True
            except FileNotFoundError:
                return False
            finally:
                sftp.close()

        except Exception:
            return False
