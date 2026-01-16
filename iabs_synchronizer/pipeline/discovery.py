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


def extract_session_id(filename: str, suffixes: dict) -> Optional[str]:
    """
    Extract session ID from filename by removing known suffix.

    Args:
        filename: Full filename (e.g., 'RT_A04_S01_data.npz')
        suffixes: Dict mapping suffixes to file types (e.g., {'data.npz': 'activity_data'})

    Returns:
        Session ID if suffix matched (e.g., 'RT_A04_S01'), None otherwise
    """
    for suffix in suffixes.keys():
        full_suffix = f'_{suffix}'
        if filename.endswith(full_suffix):
            return filename[:-len(full_suffix)]
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

        Uses session-ID-first approach:
        1. Scan files and extract candidate session IDs
        2. Validate each candidate (2+ underscores)
        3. For each valid session, check which expected files exist

        Args:
            path_config: Optional dict specifying custom paths for each file type

        Returns:
            dict: Mapping experiment names to file information
        """
        experiments = {}

        try:
            # Use suffixes from config
            suffixes = NEW_FORMAT_SUFFIXES

            if path_config is None:
                # Standard discovery: all files in root directory
                all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
                all_files_set = set(all_files)

                # Step 1: Extract unique candidate session IDs from all files
                candidates = set()
                for filename in all_files:
                    session_id = extract_session_id(filename, suffixes)
                    if session_id:
                        candidates.add(session_id)

                # Step 2: Validate session IDs (2+ underscores)
                valid_sessions = {s for s in candidates if is_valid_session_id(s)}

                # Step 3: For each valid session, find its files
                for session_id in valid_sessions:
                    files = {}
                    for suffix, file_type in suffixes.items():
                        expected = f"{session_id}_{suffix}"
                        if expected in all_files_set:
                            files[file_type] = expected

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
                # Scattered files discovery: scan each directory separately
                experiments = self._discover_scattered(path_config, suffixes)

        except Exception as e:
            print(f"Error discovering new format experiments: {e}")

        return experiments

    def _discover_scattered(self, path_config: Dict[str, str], suffixes: dict) -> Dict[str, Dict[str, any]]:
        """
        Discover experiments with files scattered across multiple directories.

        Uses session-ID-first approach:
        1. Scan all configured directories to collect candidate session IDs
        2. Validate each candidate (2+ underscores)
        3. For each valid session, check which expected files exist

        Args:
            path_config: Dict specifying custom paths for each file type
            suffixes: Dict mapping file suffixes to their types

        Returns:
            dict: Mapping experiment names to file information
        """
        experiments = {}

        try:
            # Step 1: Collect candidate session IDs from all configured directories
            candidates = set()
            dir_files = {}  # Cache: directory -> set of filenames

            # Get unique directories to scan
            directories_to_scan = set(path_config.values())
            directories_to_scan.add(self.root)  # Include root as fallback

            for directory in directories_to_scan:
                try:
                    if os.path.isdir(directory):
                        files = {f for f in os.listdir(directory)
                                if os.path.isfile(os.path.join(directory, f))}
                        dir_files[directory] = files

                        # Extract session IDs from files in this directory
                        for filename in files:
                            session_id = extract_session_id(filename, suffixes)
                            if session_id:
                                candidates.add(session_id)
                except Exception as e:
                    print(f"Error scanning directory {directory}: {e}")

            # Step 2: Validate session IDs (2+ underscores)
            valid_sessions = {s for s in candidates if is_valid_session_id(s)}

            # Step 3: For each valid session, check which files exist
            for session_id in valid_sessions:
                files = {}

                for suffix, file_type in suffixes.items():
                    search_dir = path_config.get(file_type, self.root)
                    expected_filename = f"{session_id}_{suffix}"

                    # Use cached file list if available, otherwise check directly
                    if search_dir in dir_files:
                        if expected_filename in dir_files[search_dir]:
                            files[file_type] = expected_filename
                    else:
                        file_path = os.path.join(search_dir, expected_filename)
                        if os.path.exists(file_path):
                            files[file_type] = expected_filename

                # Only include sessions that have at least one file
                if files:
                    complete = 'activity_data' in files and 'behavior_features' in files

                    experiments[session_id] = {
                        'format': 'new',
                        'files': files,
                        'complete': complete,
                        'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                        'has_metadata': ('metadata' in files),
                        'path_config': path_config
                    }

        except Exception as e:
            print(f"Error discovering scattered experiments: {e}")

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
