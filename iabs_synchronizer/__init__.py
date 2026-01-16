"""
IABS Data Synchronizer - Neuroscience Data Synchronization Library

A Python package for synchronizing multi-modal neuroscience recordings
(calcium imaging, behavior tracking, coordinates) to a common temporal reference.

Quick Start:
    >>> from iabs_synchronizer import Synchronizer
    >>> sync = Synchronizer(root_path='/data/neuroscience')
    >>> result = sync.synchronize_experiment('RT_A04_S01')
    >>> result.save('output.npz')

Main Components:
    - Synchronizer: Main orchestration class
    - SyncConfig: Configuration object for custom parameters
    - SyncResult: Result object with aligned data and logs

For backward compatibility with notebook code:
    >>> from iabs_synchronizer import read_all_data, filter_data, align_all_data
"""

__version__ = '1.1.0'
__author__ = 'IABS Lab'

# Main public API
from .pipeline.synchronizer import Synchronizer
from .config import SyncConfig
from .models.data_structures import SyncResult

# Core functions (for backward compatibility with notebook)
from .core.io import read_all_data
from .core.filtering import filter_data
from .core.alignment import align_all_data
from .core.postprocessing import (
    rename_attributes,
    save_aligned_data,
    load_aligned_data,
    validate_alignment,
    print_alignment_summary
)

# Optional GDrive support
try:
    from .gdrive.adapter import (
        save_to_gdrive,
        load_from_gdrive,
        is_gdrive_available
    )
    _has_gdrive = True
except ImportError:
    _has_gdrive = False


__all__ = [
    # Main classes
    'Synchronizer',
    'SyncConfig',
    'SyncResult',

    # Core functions (backward compatibility)
    'read_all_data',
    'filter_data',
    'align_all_data',

    # Post-processing
    'rename_attributes',
    'save_aligned_data',
    'load_aligned_data',
    'validate_alignment',
    'print_alignment_summary',

    # Metadata
    '__version__',
]

# Add GDrive functions if available
if _has_gdrive:
    __all__.extend([
        'save_to_gdrive',
        'load_from_gdrive',
        'is_gdrive_available'
    ])


def get_version():
    """Get package version string."""
    return __version__


def check_gdrive_support():
    """
    Check if Google Drive support is available.

    Returns:
        bool: True if gdrive module is installed and available
    """
    return _has_gdrive
