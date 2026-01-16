"""
Experiment discovery functionality.

This module provides classes for discovering experiments in various formats
from the file system.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import os

from ..config import NEW_FORMAT_SUFFIXES


def is_valid_session_id(name: str) -> bool:
    """
    Check if a string is a valid session ID.

    Valid session IDs must have at least 2 underscores (experiment_animal-id_session).
    Examples: LNOF_J01_1D, RT_A04_S01, NOF_H01_1D_baseline

    Args:
        name: Candidate session ID string

    Returns:
        bool: True if valid session ID format
    """
    return name.count('_') >= 2


def get_file_suffix(filename: str) -> Optional[str]:
    """
    Extract suffix (everything after first 3 underscore-separated parts).

    Example: '3DM_J19_5D_VT_TS.csv' → 'VT_TS.csv'

    Args:
        filename: Full filename

    Returns:
        Suffix string if filename has 4+ parts, None otherwise
    """
    parts = filename.split('_', 3)  # Split into max 4 parts
    if len(parts) == 4:
        return parts[3]
    return None


def map_suffix_to_file_type(suffix: str, suffixes: dict) -> Optional[str]:
    """
    Map a file suffix to its file type.

    Args:
        suffix: File suffix (e.g., 'VT_TS.csv')
        suffixes: Dict mapping suffixes to file types

    Returns:
        File type if suffix recognized, None otherwise
    """
    return suffixes.get(suffix)


def extract_session_id(filename: str) -> Optional[str]:
    """
    Extract session ID from first three underscore-separated parts.

    Args:
        filename: Full filename (e.g., 'RT_A04_S01_data.npz')

    Returns:
        Session ID (e.g., 'RT_A04_S01') if filename has 3+ parts, None otherwise
    """
    parts = filename.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return None


class ExperimentDiscovery:
    """
    Discovers experiments in root directory based on file naming patterns.

    Supports two formats:
    - 'new': Modern format with _data.npz, _Features.csv, etc.
    - 'legacy': Traditional nested directory structure
    """

    def __init__(self, root_path: str):
        """
        Initialize experiment discovery.

        Args:
            root_path: Root directory containing experiment data (or base path when using path_config)
        """
        self.root = root_path

    def list_experiments(self) -> List[str]:
        """
        List all experiment directories in root path (legacy format).

        Returns:
            list: Names of all subdirectories in root path

        Example:
            >>> discovery = ExperimentDiscovery(root_path='/data')
            >>> experiments = discovery.list_experiments()
            >>> print(f"Found {len(experiments)} experiments")
        """
        try:
            experiments = [
                d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            ]
            return sorted(experiments)
        except Exception as e:
            print(f"Error listing experiments: {e}")
            return []

    def discover_experiments(self, format: str = 'auto', path_config: Dict[str, str] = None) -> Dict[str, Dict[str, any]]:
        """
        Auto-discover experiments based on file naming patterns.

        Supports two formats:
        - 'new': Modern format with _data.npz, _Features.csv, etc.
        - 'legacy': Traditional nested directory structure
        - 'auto': Try both (new format first)

        Naming convention: experiment_animal-id_session
        - experiment: e.g., LNOF, RT, NOF
        - animal-id: e.g., J01, A04, H01
        - session: e.g., 1D, S01, S02_part1 (can contain underscores)

        Examples:
        - LNOF_J01_1D
        - RT_A04_S01
        - NOF_H01_1D
        - RT_A04_S01_baseline

        Args:
            format: Which format to check ('new', 'legacy', or 'auto')
            path_config: Optional dict specifying custom paths for scattered files (new format only).
                        Keys: 'activity_data', 'behavior_features', 'activity_timeline',
                              'behavior_timeline', 'metadata'
                        Example:
                            {
                                'activity_data': '/mnt/imaging',
                                'behavior_features': '/mnt/behavior',
                                'activity_timeline': '/mnt/timestamps',
                                'behavior_timeline': '/mnt/timestamps',
                                'metadata': '/mnt/metadata'
                            }

        Returns:
            dict: Mapping experiment names to metadata:
                {
                    'experiment_name': {
                        'format': 'new' or 'legacy',
                        'files': dict of available files,
                        'complete': bool (has minimum required files)
                    }
                }

        Example:
            >>> discovery = ExperimentDiscovery(root_path='/data/neuroscience')
            >>> experiments = discovery.discover_experiments()
            >>> for name, info in experiments.items():
            ...     print(f"{name}: {info['format']}, complete={info['complete']}")

            >>> # With scattered files
            >>> path_config = {
            ...     'activity_data': '/mnt/imaging',
            ...     'behavior_features': '/mnt/behavior'
            ... }
            >>> experiments = discovery.discover_experiments(path_config=path_config)
        """
        discovered = {}

        if format in ('new', 'auto'):
            # Discover new format experiments
            new_format_exps = self._discover_new_format(path_config=path_config)
            discovered.update(new_format_exps)

        if format in ('legacy', 'auto') and path_config is None:
            # Discover legacy format experiments (only if not already found in new format)
            # Legacy format doesn't support path_config (all files must be in nested structure)
            legacy_format_exps = self._discover_legacy_format()
            for exp_name, exp_info in legacy_format_exps.items():
                if exp_name not in discovered:  # Don't override new format
                    discovered[exp_name] = exp_info

        return discovered

    def _discover_new_format(self, path_config: Dict[str, str] = None) -> Dict[str, Dict[str, any]]:
        """
        Discover experiments in new format (_data.npz, _Features.csv, etc.).

        Uses first-three-parts approach:
        1. Scan files and extract session IDs from first 3 underscore-separated parts
        2. Extract suffix and map to file type
        3. Detect duplicate file types per session (raises ValueError)
        4. Build experiment dict

        Args:
            path_config: Optional dict specifying custom paths for each file type

        Returns:
            dict: Mapping experiment names to file information

        Raises:
            ValueError: If duplicate file types found for same session
        """
        experiments = {}

        # Use suffixes from config
        suffixes = NEW_FORMAT_SUFFIXES

        if path_config is None:
            # Standard discovery: all files in root directory
            all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]

            # Step 1: Build mapping of session_id -> {file_type: filename}
            # with conflict detection
            session_files = defaultdict(dict)  # session_id -> {file_type: filename}

            for filename in all_files:
                session_id = extract_session_id(filename)
                if not session_id or not is_valid_session_id(session_id):
                    continue

                suffix = get_file_suffix(filename)
                if not suffix:
                    continue

                file_type = map_suffix_to_file_type(suffix, suffixes)
                if not file_type:
                    continue

                # Check for duplicate file type
                if file_type in session_files[session_id]:
                    existing = session_files[session_id][file_type]
                    raise ValueError(
                        f"Duplicate file type '{file_type}' for session '{session_id}': "
                        f"'{existing}' and '{filename}'"
                    )

                session_files[session_id][file_type] = filename

            # Step 2: Build experiment dict from collected files
            for session_id, files in session_files.items():
                complete = 'activity_data' in files and 'behavior_features' in files

                experiments[session_id] = {
                    'format': 'new',
                    'files': files,
                    'complete': complete,
                    'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                    'has_metadata': ('metadata' in files),
                    'path': self.root
                }

        else:
            # Scattered files discovery: folder determines file type
            experiments = self._discover_scattered(path_config)

        return experiments

    def _discover_scattered(self, path_config: Dict[str, str]) -> Dict[str, Dict[str, any]]:
        """
        Discover experiments with files scattered across multiple directories.

        Folder determines file type. Each folder in path_config contains ONE type of data.
        Ambiguity = two files with same 3-part session ID in same folder.

        Args:
            path_config: Dict specifying custom paths for each file type.
                        Keys are file types, values are directory paths.

        Returns:
            dict: Mapping experiment names to file information

        Raises:
            ValueError: If two files with same session ID found in same folder
        """
        experiments = {}
        session_files = defaultdict(dict)  # session_id -> {file_type: filename}

        for file_type, directory in path_config.items():
            if not os.path.isdir(directory):
                continue

            # Track session IDs seen in THIS directory for ambiguity detection
            seen_in_dir = {}  # session_id -> filename

            for filename in os.listdir(directory):
                if not os.path.isfile(os.path.join(directory, filename)):
                    continue

                session_id = extract_session_id(filename)
                if not session_id or not is_valid_session_id(session_id):
                    continue

                # Ambiguity: two files with same session ID in same folder
                if session_id in seen_in_dir:
                    raise ValueError(
                        f"Ambiguous files for session '{session_id}' in {directory}: "
                        f"'{seen_in_dir[session_id]}' and '{filename}'"
                    )

                seen_in_dir[session_id] = filename
                session_files[session_id][file_type] = filename

        # Step 2: Build experiment dict from collected files
        for session_id, files in session_files.items():
            if not files:
                continue

            complete = 'activity_data' in files and 'behavior_features' in files

            experiments[session_id] = {
                'format': 'new',
                'files': files,
                'complete': complete,
                'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                'has_metadata': ('metadata' in files),
                'path_config': path_config
            }

        return experiments

    def _discover_legacy_format(self) -> Dict[str, Dict[str, any]]:
        """
        Discover experiments in legacy format (nested directories).

        Returns:
            dict: Mapping experiment names to directory information
        """
        experiments = {}

        try:
            # Get all subdirectories
            subdirs = [
                d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            ]

            for exp_name in subdirs:
                exp_path = os.path.join(self.root, exp_name)

                # Check for required subdirectories
                try:
                    contents = os.listdir(exp_path)
                    has_calcium = 'Calcium' in contents
                    has_behavior = 'Behavior_auto' in contents or 'Behavior' in contents

                    experiments[exp_name] = {
                        'format': 'legacy',
                        'subdirs': contents,
                        'complete': has_calcium,  # Calcium is minimum requirement
                        'has_behavior': has_behavior,
                        'path': exp_path
                    }
                except Exception as e:
                    print(f"Error reading directory {exp_name}: {e}")

        except Exception as e:
            print(f"Error discovering legacy format experiments: {e}")

        return experiments
