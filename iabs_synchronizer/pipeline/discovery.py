"""
Experiment discovery functionality.

This module provides classes for discovering experiments in various formats
from the file system.
"""

from typing import Dict, List
from collections import defaultdict
import os

from ..config import NEW_FORMAT_SUFFIXES


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

                # Group files by experiment name
                exp_files = defaultdict(dict)

                for filename in all_files:
                    # Check each known suffix
                    for suffix, file_type in suffixes.items():
                        if filename.endswith(f'_{suffix}'):
                            # Extract experiment name
                            exp_name = filename[:-len(f'_{suffix}')]

                            # Validate naming convention (at least 2 underscores)
                            if exp_name.count('_') >= 2:
                                exp_files[exp_name][file_type] = filename

                # Build experiment info
                for exp_name, files in exp_files.items():
                    complete = ('activity_data' in files and 'behavior_features' in files)

                    experiments[exp_name] = {
                        'format': 'new',
                        'files': files,
                        'complete': complete,
                        'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                        'has_metadata': ('metadata' in files),
                        'path': self.root
                    }

            else:
                # Scattered files discovery: scan each directory separately
                # Step 1: Get experiment names from activity_data directory (required files)
                activity_dir = path_config.get('activity_data', self.root)
                activity_suffix = [s for s, t in suffixes.items() if t == 'activity_data'][0]

                try:
                    activity_files = [f for f in os.listdir(activity_dir)
                                     if os.path.isfile(os.path.join(activity_dir, f))
                                     and f.endswith(f'_{activity_suffix}')]

                    # Extract experiment names
                    exp_names = []
                    for filename in activity_files:
                        exp_name = filename[:-len(f'_{activity_suffix}')]
                        if exp_name.count('_') >= 2:  # Validate naming convention
                            exp_names.append(exp_name)

                    # Step 2: For each experiment name, check which files exist
                    for exp_name in exp_names:
                        files = {}

                        # Check each file type in its respective directory
                        for suffix, file_type in suffixes.items():
                            search_dir = path_config.get(file_type, self.root)
                            expected_filename = f"{exp_name}_{suffix}"
                            file_path = os.path.join(search_dir, expected_filename)

                            if os.path.exists(file_path):
                                files[file_type] = expected_filename

                        # Build experiment info
                        complete = ('activity_data' in files and 'behavior_features' in files)

                        experiments[exp_name] = {
                            'format': 'new',
                            'files': files,
                            'complete': complete,
                            'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                            'has_metadata': ('metadata' in files),
                            'path_config': path_config  # Store path_config for later use
                        }

                except Exception as e:
                    print(f"Error scanning activity_data directory {activity_dir}: {e}")

        except Exception as e:
            print(f"Error discovering new format experiments: {e}")

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
