"""
Optional Google Drive integration adapter.

This module provides Google Drive upload/download functionality using the driada library.
If driada is not installed, functions raise helpful ImportError messages.

Installation:
    pip install iabs-synchronizer[gdrive]
"""

from typing import Optional
import os

# Try to import driada - gracefully handle if not installed
try:
    from driada.gdrive.download import download_file_from_gdrive
    from driada.gdrive.upload import upload_file_to_gdrive
    from driada.gdrive.auth import authenticate
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False


class GDriveNotAvailableError(ImportError):
    """Raised when Google Drive functionality is used but driada is not installed."""

    def __init__(self, message: Optional[str] = None):
        if message is None:
            message = (
                "Google Drive support requires the 'driada' package.\n"
                "Install with: pip install iabs-synchronizer[gdrive]"
            )
        super().__init__(message)


def check_gdrive_available():
    """
    Check if Google Drive functionality is available.

    Raises:
        GDriveNotAvailableError: If driada is not installed
    """
    if not GDRIVE_AVAILABLE:
        raise GDriveNotAvailableError()


def save_to_gdrive(local_path: str,
                   remote_folder: str,
                   filename: Optional[str] = None,
                   gauth=None) -> str:
    """
    Upload file to Google Drive.

    Args:
        local_path: Path to local file to upload
        remote_folder: Google Drive folder name
        filename: Optional remote filename (default: use local filename)
        gauth: Optional authenticated GoogleAuth object (if None, will authenticate)

    Returns:
        str: Google Drive file ID

    Raises:
        GDriveNotAvailableError: If driada is not installed
        FileNotFoundError: If local file doesn't exist

    Example:
        >>> save_to_gdrive('experiment_001.npz', 'Aligned Data')
    """
    check_gdrive_available()

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Use local filename if remote filename not specified
    if filename is None:
        filename = os.path.basename(local_path)

    # Authenticate if needed
    if gauth is None:
        gauth = authenticate()

    # Upload file
    file_id = upload_file_to_gdrive(
        local_path=local_path,
        remote_folder=remote_folder,
        remote_filename=filename,
        gauth=gauth
    )

    return file_id


def load_from_gdrive(file_id: str,
                     local_path: str,
                     gauth=None) -> None:
    """
    Download file from Google Drive.

    Args:
        file_id: Google Drive file ID
        local_path: Local path to save downloaded file
        gauth: Optional authenticated GoogleAuth object (if None, will authenticate)

    Raises:
        GDriveNotAvailableError: If driada is not installed

    Example:
        >>> load_from_gdrive('1ABC...XYZ', 'experiment_001.npz')
    """
    check_gdrive_available()

    # Authenticate if needed
    if gauth is None:
        gauth = authenticate()

    # Download file
    download_file_from_gdrive(
        file_id=file_id,
        local_path=local_path,
        gauth=gauth
    )


def is_gdrive_available() -> bool:
    """
    Check if Google Drive functionality is available.

    Returns:
        bool: True if driada is installed, False otherwise

    Example:
        >>> if is_gdrive_available():
        ...     save_to_gdrive('data.npz', 'My Folder')
        ... else:
        ...     print("Install gdrive support: pip install iabs-synchronizer[gdrive]")
    """
    return GDRIVE_AVAILABLE


# Provide helpful error message if gdrive module is imported but not available
if not GDRIVE_AVAILABLE:
    import warnings
    warnings.warn(
        "Google Drive support is not available. "
        "Install with: pip install iabs-synchronizer[gdrive]",
        ImportWarning
    )
