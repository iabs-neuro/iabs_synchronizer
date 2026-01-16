"""
Experiment validation functionality.

This module provides classes for validating experiment data before synchronization.
"""

from typing import Dict, List, Tuple
import os

from ..config import NEW_FORMAT_SUFFIXES
from .discovery import is_valid_session_id


class ExperimentValidator:
    """
    Validates experiment data without synchronizing.

    Checks for required files, data pieces, and basic structure.
    """

    def __init__(self, root_path: str, data_pieces: List[str]):
        """
        Initialize experiment validator.

        Args:
            root_path: Root directory containing experiment data
            data_pieces: List of expected data piece names (e.g., ['Calcium', 'Behavior_auto'])
        """
        self.root = root_path
        self.data_pieces = data_pieces

    def experiment_exists(self, experiment_name: str, path_config: Dict[str, str] = None) -> Tuple[bool, str]:
        """
        Check if experiment exists in either legacy or new format.

        Args:
            experiment_name: Name of experiment to check
            path_config: Optional path configuration for scattered files

        Returns:
            tuple: (exists, format) where:
                - exists: True if experiment found
                - format: 'legacy', 'new', or '' if not found
        """
        # Validate session ID format early (new format requires 2+ underscores)
        valid_session_format = is_valid_session_id(experiment_name)

        # Check new format first (only if valid session ID format)
        if valid_session_format:
            activity_suffix = [suffix for suffix, file_type in NEW_FORMAT_SUFFIXES.items()
                              if file_type == 'activity_data'][0]  # 'data.npz'

            if path_config and 'activity_data' in path_config:
                exp_path_new = os.path.join(path_config['activity_data'], f"{experiment_name}_{activity_suffix}")
            else:
                exp_path_new = os.path.join(self.root, f"{experiment_name}_{activity_suffix}")

            if os.path.exists(exp_path_new):
                return True, 'new'

        # Check legacy format (directory-based, no session ID format requirement)
        exp_path_legacy = os.path.join(self.root, experiment_name)
        if os.path.exists(exp_path_legacy):
            return True, 'legacy'

        return False, ''

    def validate_experiment(self, experiment_name: str) -> Dict[str, any]:
        """
        Validate experiment data without synchronizing.

        Checks:
        - Experiment directory exists
        - Calcium data is present (required)
        - Which data pieces are available
        - Data can be loaded successfully

        Args:
            experiment_name: Name of experiment to validate

        Returns:
            dict: Validation report with keys:
                - 'valid': bool
                - 'has_calcium': bool
                - 'available_pieces': list
                - 'errors': list
                - 'timepoints': int (if successfully loaded)

        Example:
            >>> validator = ExperimentValidator(root_path='/data', data_pieces=['Calcium', 'Behavior_auto'])
            >>> report = validator.validate_experiment('RT_A04_S01')
            >>> if report['valid']:
            ...     print("Experiment is ready for synchronization")
        """
        report = {
            'valid': True,
            'has_calcium': False,
            'available_pieces': [],
            'errors': []
        }

        # Check experiment path
        exp_path = os.path.join(self.root, experiment_name)
        if not os.path.exists(exp_path):
            report['valid'] = False
            report['errors'].append(f"Experiment path does not exist: {exp_path}")
            return report

        # Check for data pieces
        try:
            subdirs = os.listdir(exp_path)
            available = [dp for dp in self.data_pieces if dp in subdirs]
            report['available_pieces'] = available

            # Check for calcium (required)
            if 'Calcium' in available:
                report['has_calcium'] = True
            else:
                report['valid'] = False
                report['errors'].append("Calcium data is missing (required)")

        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"Error reading experiment directory: {e}")

        return report
