"""
Main synchronization pipeline orchestration.

This module provides the high-level Synchronizer class that coordinates
data loading, filtering, alignment, and postprocessing.

Source: Lines 851-2199 from IABS_data_synchronizer_1_3.ipynb
"""

from typing import Dict, List, Optional
from pathlib import Path
import os

from ..config import SyncConfig, DEFAULT_DATA_PIECES, NEW_FORMAT_SUFFIXES
from ..models.data_structures import SyncResult
from ..core.io import read_all_data
from ..core.filtering import filter_data
from ..core.alignment import align_all_data
from ..core.postprocessing import (
    rename_attributes,
    exclude_features,
    save_aligned_data,
    validate_alignment
)
from .discovery import ExperimentDiscovery
from .validation import ExperimentValidator


class Synchronizer:
    """
    Main interface for neuroscience data synchronization.

    The Synchronizer coordinates the entire synchronization pipeline:
    1. Data loading (read_all_data)
    2. Quality filtering (filter_data)
    3. Temporal alignment (align_all_data)
    4. Post-processing (rename, exclude, validate)

    Attributes:
        root: Root directory path containing experiment data
        config: SyncConfig object with synchronization parameters
        data_pieces: List of data piece types to load

    Example:
        >>> # Basic usage
        >>> sync = Synchronizer(root_path='/data/neuroscience')
        >>> result = sync.synchronize_experiment('RT_A04_S01')
        >>> result.save('output.npz')

        >>> # Advanced usage with custom config
        >>> config = SyncConfig(align_precision=0.01)  # Stricter alignment
        >>> sync = Synchronizer(root_path='/data', config=config)
        >>> result = sync.synchronize_experiment(
        ...     'RT_A04_S01',
        ...     force_mode='cast_to_ca',
        ...     rename_dict={'Speed': 'locomotion_speed'},
        ...     exclude_features=['x_green', 'y_green']
        ... )
    """

    def __init__(self,
                 root_path: str,
                 config: Optional[SyncConfig] = None,
                 data_pieces: Optional[List[str]] = None):
        """
        Initialize synchronizer.

        Args:
            root_path: Root directory containing experiment folders
            config: Optional SyncConfig with custom parameters
            data_pieces: Optional list of data piece types to load
                        (default: config.default_data_pieces)
        """
        self.root = str(Path(root_path).absolute())
        self.config = config if config is not None else SyncConfig()
        self.data_pieces = data_pieces if data_pieces is not None else self.config.default_data_pieces

        # Validate root path exists
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Root path does not exist: {self.root}")

        # Initialize discovery and validation components
        self._discovery = ExperimentDiscovery(self.root)
        self._validator = ExperimentValidator(self.root, self.data_pieces)

    def synchronize_experiment(self,
                               experiment_name: str,
                               force_mode: Optional[str] = None,
                               rename_dict: Optional[Dict[str, str]] = None,
                               exclude_list: Optional[List[str]] = None,
                               validate: bool = True,
                               path_config: Optional[Dict[str, str]] = None,
                               files: Optional[Dict[str, str]] = None) -> SyncResult:
        """
        Synchronize all data for a single experiment.

        Complete pipeline:
        1. Load all data pieces from experiment directory
        2. Filter data based on quality thresholds
        3. Align all data to calcium reference timeline
        4. Apply renaming and exclusions
        5. Validate alignment consistency
        6. Return SyncResult with aligned data and logs

        Args:
            experiment_name: Name of experiment (subdirectory under root)
            force_mode: Optional alignment mode to force
                       (None = auto-select, or one of ['2 timelines', 'simple',
                        'cast_to_ca', 'crop'])
            rename_dict: Optional dict mapping old names to new names
            exclude_list: Optional list of features to exclude from output
            validate: If True, validate alignment consistency (default: True)
            path_config: Optional dict specifying custom paths for each file type
                        (new format only). Allows files to be in different directories.
                        Keys: 'activity_data', 'behavior_features', 'activity_timeline',
                              'behavior_timeline', 'metadata'
                        Values: Base directory paths for each file type
                        Example:
                            {
                                'activity_data': '/data/imaging',
                                'behavior_features': '/data/behavior',
                                'activity_timeline': '/data/timestamps',
                                'behavior_timeline': '/data/timestamps',
                                'metadata': '/data/metadata'
                            }
            files: Optional dict of discovered filenames for each file type.
                   Used in scattered mode where folder determines file type.
                   Example:
                       {
                           'activity_data': '3DM_J19_5D_data.npz',
                           'behavior_features': '3DM_J19_5D_features.csv',
                           ...
                       }

        Returns:
            SyncResult: Object containing aligned_data dict and logs

        Raises:
            FileNotFoundError: If experiment directory doesn't exist
            KeyError: If Calcium data is missing (required)
            ValueError: If alignment validation fails

        Example:
            >>> result = sync.synchronize_experiment(
            ...     'RT_A04_S01',
            ...     force_mode='cast_to_ca',
            ...     rename_dict={'X': 'x_position', 'Y': 'y_position'},
            ...     exclude_list=['x_green', 'y_green']
            ... )
            >>> print(f"Synchronized {len(result.aligned_data)} features")
            >>> result.save('RT_A04_S01_aligned.npz')

            >>> # Using custom paths for scattered files
            >>> path_config = {
            ...     'activity_data': '/mnt/imaging',
            ...     'behavior_features': '/mnt/behavior',
            ...     'activity_timeline': '/mnt/timestamps',
            ...     'behavior_timeline': '/mnt/timestamps',
            ...     'metadata': '/mnt/metadata'
            ... }
            >>> result = sync.synchronize_experiment(
            ...     'RT_A04_S01',
            ...     path_config=path_config
            ... )
        """
        # Validate experiment exists (delegate to validator)
        exists, exp_format = self._validator.experiment_exists(experiment_name, path_config)
        if not exists:
            raise FileNotFoundError(
                f"Experiment '{experiment_name}' not found in either legacy or new format at: {self.root}"
            )

        # Phase 1: Load data (with metadata support)
        active_data_pieces, extracted_info, read_log, source_metadata = read_all_data(
            experiment_name,
            root=self.root,
            data_pieces=self.data_pieces,
            prefer_new_format=True,
            path_config=path_config,
            files=files
        )

        # Phase 2: Filter data
        filtered_info, filter_log = filter_data(
            active_data_pieces,
            extracted_info,
            config=self.config
        )

        # Phase 3: Align data
        aligned_data, align_log, mode_stats = align_all_data(
            filtered_info,
            force_pathway=force_mode,
            config=self.config
        )

        # Phase 4: Post-processing
        if rename_dict:
            aligned_data = rename_attributes(aligned_data, rename_dict)

        if exclude_list:
            aligned_data = exclude_features(aligned_data, exclude_list)

        # Phase 5: Validation
        if validate:
            validation = validate_alignment(aligned_data)
            if not validation['valid']:
                error_msg = "Alignment validation failed:\n" + "\n".join(validation['errors'])
                raise ValueError(error_msg)

        # Build synchronization info
        from datetime import datetime
        from .. import __version__

        sync_info = {
            'synchronizer_version': __version__,
            'sync_timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'source_format': 'new' if source_metadata else 'legacy',
            'alignment_mode': force_mode if force_mode else 'auto',
            'n_features': len(aligned_data),
            'validation_passed': validation['valid'] if validate else None,
            'mode_stats': mode_stats  # Track how many features used each alignment mode
        }

        # Add timepoints if available
        try:
            sync_info['n_timepoints'] = validation['timepoints']
        except (KeyError, TypeError):
            # Validation dict doesn't have timepoints key or validation is not a dict
            pass

        # Create result object
        result = SyncResult(
            aligned_data=aligned_data,
            read_log=read_log,
            filter_log=filter_log,
            align_log=align_log,
            metadata={  # Legacy field for backward compatibility
                'experiment_name': experiment_name,
                'root_path': self.root,
                'force_mode': force_mode,
                'config': self.config
            },
            source_metadata=source_metadata,  # New: from data loading
            sync_info=sync_info  # New: synchronization tracking
        )

        return result

    def synchronize_batch(self,
                         experiment_list: List[str],
                         output_dir: Optional[str] = None,
                         **kwargs) -> Dict[str, SyncResult]:
        """
        Synchronize multiple experiments in batch.

        Args:
            experiment_list: List of experiment names to process
            output_dir: Optional directory to save results
                       (if provided, saves each result as {expname}_aligned.npz)
            **kwargs: Additional arguments passed to synchronize_experiment()
                     (e.g., force_mode, rename_dict, exclude_list, validate, path_config)

        Returns:
            dict: Mapping experiment names to SyncResult objects

        Example:
            >>> experiments = ['RT_A04_S01', 'RT_A04_S02', 'RT_A04_S03']
            >>> results = sync.synchronize_batch(
            ...     experiments,
            ...     output_dir='/data/aligned',
            ...     force_mode='cast_to_ca'
            ... )
            >>> print(f"Successfully processed {len(results)} experiments")

            >>> # Using custom paths for scattered files
            >>> path_config = {
            ...     'activity_data': '/mnt/imaging',
            ...     'behavior_features': '/mnt/behavior',
            ...     'activity_timeline': '/mnt/timestamps',
            ...     'behavior_timeline': '/mnt/timestamps',
            ...     'metadata': '/mnt/metadata'
            ... }
            >>> results = sync.synchronize_batch(
            ...     experiments,
            ...     output_dir='/data/aligned',
            ...     path_config=path_config
            ... )
        """
        results = {}
        failed = []

        # Extract discovered_files from kwargs (used in scattered mode)
        discovered_files = kwargs.pop('discovered_files', None)

        try:
            from tqdm import tqdm
            iterator = tqdm(experiment_list, desc="Synchronizing experiments")
        except ImportError:
            iterator = experiment_list
            print(f"Processing {len(experiment_list)} experiments...")

        for exp_name in iterator:
            try:
                # Get discovered files for this experiment (if available)
                exp_files = discovered_files.get(exp_name) if discovered_files else None
                result = self.synchronize_experiment(exp_name, files=exp_files, **kwargs)
                results[exp_name] = result

                # Save if output directory specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f'{exp_name}_aligned.npz')
                    result.save(output_path)
                    print(f"Saved {exp_name} to {output_path}")

            except Exception as e:
                print(f"Failed to process {exp_name}: {e}")
                failed.append((exp_name, str(e)))

        # Aggregate mode statistics across experiments
        aggregated_mode_stats = {}  # mode -> count of experiments using this mode
        for exp_name, result in results.items():
            exp_modes = result.sync_info.get('mode_stats', {})
            for mode in exp_modes.keys():
                if mode not in aggregated_mode_stats:
                    aggregated_mode_stats[mode] = 0
                aggregated_mode_stats[mode] += 1

        # Summary
        print(f"\nBatch processing complete:")
        print(f"  Success: {len(results)}/{len(experiment_list)}")
        if failed:
            print(f"  Failed: {len(failed)}")
            for exp_name, error in failed:
                print(f"    - {exp_name}: {error}")

        # Report alignment mode statistics
        if aggregated_mode_stats:
            print(f"\n  Alignment methods used:")
            for mode, count in sorted(aggregated_mode_stats.items(), key=lambda x: -x[1]):
                print(f"    - {mode}: {count} experiment(s)")

        return results

    def list_experiments(self) -> List[str]:
        """
        List all experiment directories in root path (legacy format).

        Returns:
            list: Names of all subdirectories in root path

        Example:
            >>> sync = Synchronizer(root_path='/data')
            >>> experiments = sync.list_experiments()
            >>> print(f"Found {len(experiments)} experiments")
        """
        return self._discovery.list_experiments()

    def discover_experiments(self, format: str = 'auto', path_config: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, any]]:
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
            path_config: Optional dict specifying custom paths for scattered files (new format only)

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
            >>> sync = Synchronizer(root_path='/data/neuroscience')
            >>> experiments = sync.discover_experiments()
            >>> for name, info in experiments.items():
            ...     print(f"{name}: {info['format']}, complete={info['complete']}")

            >>> # With scattered files
            >>> path_config = {'activity_data': '/mnt/imaging', 'behavior_features': '/mnt/behavior'}
            >>> experiments = sync.discover_experiments(path_config=path_config)
        """
        return self._discovery.discover_experiments(format=format, path_config=path_config)

    def synchronize_all(self,
                       output_dir: str,
                       format: str = 'auto',
                       only_complete: bool = True,
                       **kwargs) -> Dict[str, SyncResult]:
        """
        Auto-discover and synchronize all experiments in root directory.

        This method automatically discovers experiments based on file naming patterns,
        validates they have required files, and synchronizes all (or only complete) experiments.

        Naming convention support:
        - experiment_animal-id_session (e.g., LNOF_J01_1D, RT_A04_S01)
        - Session can have multiple underscores (e.g., RT_A04_S01_baseline)

        Args:
            output_dir: Directory where synchronized files will be saved
            format: Which format to discover ('new', 'legacy', or 'auto')
            only_complete: If True, only process experiments with all required files
            **kwargs: Additional arguments passed to synchronize_experiment()
                     (e.g., force_mode, rename_dict, exclude_list, validate, path_config)

        Returns:
            dict: Mapping experiment names to SyncResult objects (only successful ones)

        Raises:
            FileNotFoundError: If output_dir parent doesn't exist

        Example:
            >>> # Basic usage: Auto-discover and synchronize all complete experiments
            >>> sync = Synchronizer(root_path='/data/neuroscience')
            >>> results = sync.synchronize_all(output_dir='/data/synchronized')
            >>> print(f"Synchronized {len(results)} experiments")

            >>> # Process all experiments (including incomplete)
            >>> results = sync.synchronize_all(
            ...     output_dir='/data/synchronized',
            ...     only_complete=False,
            ...     force_mode='cast_to_ca'
            ... )

            >>> # Process only new format
            >>> results = sync.synchronize_all(
            ...     output_dir='/data/synchronized',
            ...     format='new'
            ... )

            >>> # Using custom paths for scattered files
            >>> path_config = {
            ...     'activity_data': '/mnt/imaging',
            ...     'behavior_features': '/mnt/behavior',
            ...     'activity_timeline': '/mnt/timestamps',
            ...     'behavior_timeline': '/mnt/timestamps',
            ...     'metadata': '/mnt/metadata'
            ... }
            >>> results = sync.synchronize_all(
            ...     output_dir='/data/synchronized',
            ...     path_config=path_config
            ... )
        """
        # Validate output directory
        output_path = Path(output_dir)
        if not output_path.parent.exists():
            raise FileNotFoundError(f"Output directory parent does not exist: {output_path.parent}")

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Extract path_config from kwargs if present (for discovery)
        path_config = kwargs.get('path_config', None)

        # Discover experiments
        if path_config:
            print(f"Discovering experiments in scattered directories...")
        else:
            print(f"Discovering experiments in: {self.root}")
        discovered = self.discover_experiments(format=format, path_config=path_config)

        if not discovered:
            print("No experiments found.")
            return {}

        # Filter experiments
        if only_complete:
            experiments_to_process = {
                name: info for name, info in discovered.items()
                if info['complete']
            }
            incomplete_exps = {
                name: info for name, info in discovered.items()
                if not info['complete']
            }
            if incomplete_exps:
                print(f"Found {len(discovered)} experiments, {len(incomplete_exps)} incomplete (skipping)")
                # Summarize missing reasons
                missing_summary = {}  # reason -> count
                for name, info in incomplete_exps.items():
                    for reason in info.get('missing', ['unknown reason']):
                        # Normalize reason (remove specific paths for grouping)
                        if 'directory not found' in reason:
                            key = reason.split('(')[0].strip() + ' (directory not found)'
                        elif 'not found in' in reason:
                            key = reason.split('(')[0].strip() + ' (not found in configured directory)'
                        else:
                            key = reason
                        missing_summary[key] = missing_summary.get(key, 0) + 1
                print("  Incomplete reasons:")
                for reason, count in sorted(missing_summary.items(), key=lambda x: -x[1]):
                    print(f"    - {reason}: {count} experiment(s)")
        else:
            experiments_to_process = discovered

        if not experiments_to_process:
            print("No complete experiments found to process.")
            return {}

        # Print summary
        print(f"\nExperiments to process: {len(experiments_to_process)}")
        for name, info in sorted(experiments_to_process.items()):
            status_icons = []
            if info['format'] == 'new':
                if info.get('has_timelines'):
                    status_icons.append("T")  # Timelines
                if info.get('has_metadata'):
                    status_icons.append("M")  # Metadata
            status = f" [{','.join(status_icons)}]" if status_icons else ""
            print(f"  - {name} ({info['format']}){status}")

        print()

        # Process experiments using synchronize_batch
        # Pass discovered files for each experiment (used in scattered mode)
        experiment_names = list(experiments_to_process.keys())
        discovered_files = {
            name: info.get('files', {})
            for name, info in experiments_to_process.items()
        }
        results = self.synchronize_batch(
            experiment_names,
            output_dir=output_dir,
            discovered_files=discovered_files,
            **kwargs
        )

        return results

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

        Example:
            >>> report = sync.validate_experiment('RT_A04_S01')
            >>> if report['valid']:
            ...     print("Experiment is ready for synchronization")
        """
        return self._validator.validate_experiment(experiment_name)

    def __repr__(self) -> str:
        return f"Synchronizer(root='{self.root}', config={self.config})"
