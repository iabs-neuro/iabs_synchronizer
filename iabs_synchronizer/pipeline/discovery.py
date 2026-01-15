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
            root_path: Root directory containing experiment data
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

    def discover_experiments(self, format: str = 'auto') -> Dict[str, Dict[str, any]]:
        """
        Auto-discover experiments in root directory based on file naming patterns.

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
        """
        discovered = {}

        if format in ('new', 'auto'):
            # Discover new format experiments
            new_format_exps = self._discover_new_format()
            discovered.update(new_format_exps)

        if format in ('legacy', 'auto'):
            # Discover legacy format experiments (only if not already found in new format)
            legacy_format_exps = self._discover_legacy_format()
            for exp_name, exp_info in legacy_format_exps.items():
                if exp_name not in discovered:  # Don't override new format
                    discovered[exp_name] = exp_info

        return discovered

    def _discover_new_format(self) -> Dict[str, Dict[str, any]]:
        """
        Discover experiments in new format (_data.npz, _Features.csv, etc.).

        Returns:
            dict: Mapping experiment names to file information
        """
        experiments = {}

        try:
            # Get all files in root directory
            all_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]

            # Group files by experiment name
            # Pattern: {expname}_{suffix}.{ext}
            # Extract experiment name by removing suffix
            exp_files = defaultdict(dict)

            # Use suffixes from config (can be customized if needed)
            suffixes = NEW_FORMAT_SUFFIXES

            for filename in all_files:
                # Check each known suffix
                for suffix, file_type in suffixes.items():
                    if filename.endswith(f'_{suffix}'):
                        # Extract experiment name
                        exp_name = filename[:-len(f'_{suffix}')]

                        # Validate naming convention (at least 2 underscores for experiment_animal_session)
                        if exp_name.count('_') >= 2:
                            exp_files[exp_name][file_type] = filename

            # Build experiment info
            for exp_name, files in exp_files.items():
                # Check if experiment is complete (minimum: activity_data + behavior_features)
                complete = ('activity_data' in files and 'behavior_features' in files)

                experiments[exp_name] = {
                    'format': 'new',
                    'files': files,
                    'complete': complete,
                    'has_timelines': ('activity_timeline' in files and 'behavior_timeline' in files),
                    'has_metadata': ('metadata' in files),
                    'path': self.root
                }

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
