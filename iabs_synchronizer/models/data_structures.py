"""
Data structures for neuroscience data synchronization.

This module defines type-safe dataclasses used throughout the synchronization pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np


@dataclass
class DataPiece:
    """
    Single data piece (e.g., one behavioral feature, calcium trace, etc.).

    Attributes:
        name: Identifier for this data piece (e.g., 'speed', 'calcium')
        data: Numpy array containing the time series data
        timeline: Optional timeline (timestamps) for this data
        fps: Optional frames-per-second value for this data
    """
    name: str
    data: np.ndarray
    timeline: Optional[np.ndarray] = None
    fps: Optional[int] = None

    def __repr__(self) -> str:
        shape_str = f"shape={self.data.shape}"
        timeline_str = f", timeline_len={len(self.timeline)}" if self.timeline is not None else ""
        fps_str = f", fps={self.fps}" if self.fps is not None else ""
        return f"DataPiece(name='{self.name}', {shape_str}{timeline_str}{fps_str})"


@dataclass
class ExtractedInfo:
    """
    Raw extracted data from file loading phase.

    Contains all data pieces loaded from disk, organized by data piece type.
    Each data piece type can have multiple records (e.g., multiple behavior files).

    Attributes:
        pieces: Dictionary mapping data piece names to lists of tuples
                Format: {data_piece_name: [(data, names, timeline, fps), ...]}

    Example:
        >>> extracted = ExtractedInfo(pieces={
        ...     'Calcium': [(calcium_array, None, timeline, 30)],
        ...     'Behavior_auto': [(behavior_data, feature_names, timeline, 30)]
        ... })
    """
    pieces: Dict[str, List[Tuple[np.ndarray, Optional[List[str]], Optional[np.ndarray], Optional[int]]]] = field(default_factory=dict)

    def get_active_pieces(self) -> List[str]:
        """Return list of data pieces that have data."""
        return [name for name, data_list in self.pieces.items() if len(data_list) > 0]


@dataclass
class FilteredInfo:
    """
    Filtered data ready for alignment phase.

    Data has been quality-filtered and structured for synchronization.
    Neuronal data (Calcium, Spikes) are in dict format, behavioral data in list format.

    Attributes:
        pieces: Dictionary of filtered data pieces
                Neuronal format: {data_piece: array, timeline: array, fps: int}
                Behavioral format: [{feature_name: array, timeline: array, fps: int}, ...]
        target_length: Target length for alignment (from calcium data)
        target_timeline: Target timeline for alignment (from calcium data)
        target_fps: Target FPS for alignment (from calcium data)

    Example:
        >>> filtered = FilteredInfo(
        ...     pieces={
        ...         'Calcium': {'Calcium': array, 'timeline': array, 'fps': 30},
        ...         'Behavior_auto': [
        ...             {'speed': array, 'timeline': array, 'fps': 30},
        ...             {'x': array, 'timeline': array, 'fps': 30}
        ...         ]
        ...     },
        ...     target_length=10000,
        ...     target_timeline=array([0, 0.033, 0.066, ...]),
        ...     target_fps=30
        ... )
    """
    pieces: Dict[str, Any] = field(default_factory=dict)
    target_length: Optional[int] = None
    target_timeline: Optional[np.ndarray] = None
    target_fps: Optional[int] = None


@dataclass
class SyncResult:
    """
    Complete synchronization result with aligned data and logs.

    This is the final output of the synchronization pipeline, containing
    all aligned arrays and detailed logs from each processing phase.

    Attributes:
        aligned_data: Dictionary of aligned arrays, all with same timepoint dimension
                      Format: {feature_name: array, ...}
        read_log: List of log messages from data loading phase
        filter_log: List of log messages from filtering phase
        align_log: List of log messages from alignment phase
        metadata: Optional dictionary for storing additional metadata (legacy, for backward compat)
        source_metadata: Optional source metadata from new data format (FPS, CNMF params, etc.)
        sync_info: Optional synchronization metadata (alignment mode, warnings, etc.)

    Example:
        >>> result = SyncResult(
        ...     aligned_data={
        ...         'calcium': np.array((251, 17855)),  # 251 neurons × 17855 timepoints
        ...         'speed': np.array((17855,)),        # synchronized to same length
        ...         'x': np.array((17855,))
        ...     },
        ...     read_log=['Loaded calcium...', 'Loaded behavior...'],
        ...     filter_log=['Filtered 2 NaN indices...'],
        ...     align_log=['Aligned using cast_to_ca mode...'],
        ...     source_metadata={'fps': 30.0, 'session_name': 'exp_001'},
        ...     sync_info={'alignment_mode': '2 timelines', 'n_frames': 17855}
        ... )
        >>> result.save('output.npz')  # Save to file with metadata
    """
    aligned_data: Dict[str, np.ndarray] = field(default_factory=dict)
    read_log: List[str] = field(default_factory=list)
    filter_log: List[str] = field(default_factory=list)
    align_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Legacy field for backward compat
    source_metadata: Dict[str, Any] = field(default_factory=dict)  # New: from data loading
    sync_info: Dict[str, Any] = field(default_factory=dict)  # New: synchronization tracking

    def save(self, output_path: str) -> None:
        """
        Save aligned data to .npz file with metadata.

        Metadata (source_metadata and sync_info) will be saved with keys
        '_metadata' and '_sync_info' and require allow_pickle=True when loading.

        Args:
            output_path: Path to output file (should end with .npz)

        Raises:
            ValueError: If no aligned data available to save
        """
        from ..core.postprocessing import save_aligned_data

        save_aligned_data(
            self.aligned_data,
            output_path,
            compressed=False,
            metadata=self.source_metadata if self.source_metadata else None,
            sync_info=self.sync_info if self.sync_info else None
        )

    def save_compressed(self, output_path: str) -> None:
        """
        Save aligned data to compressed .npz file with metadata.

        Metadata (source_metadata and sync_info) will be saved with keys
        '_metadata' and '_sync_info' and require allow_pickle=True when loading.

        Args:
            output_path: Path to output file (should end with .npz)

        Raises:
            ValueError: If no aligned data available to save
        """
        from ..core.postprocessing import save_aligned_data

        save_aligned_data(
            self.aligned_data,
            output_path,
            compressed=True,
            metadata=self.source_metadata if self.source_metadata else None,
            sync_info=self.sync_info if self.sync_info else None
        )

    def get_timepoints(self) -> int:
        """
        Get the number of synchronized timepoints.

        Returns:
            Number of timepoints (should be same for all arrays)

        Raises:
            ValueError: If no aligned data available or inconsistent lengths
        """
        if not self.aligned_data:
            raise ValueError("No aligned data available")

        lengths = set()
        for key, arr in self.aligned_data.items():
            if arr.ndim == 1:
                lengths.add(len(arr))
            elif arr.ndim == 2:
                lengths.add(arr.shape[1])  # Assume (neurons, timepoints) format

        if len(lengths) > 1:
            raise ValueError(f"Inconsistent timepoint lengths found: {lengths}")

        return lengths.pop()

    def get_full_log(self) -> str:
        """
        Get combined log from all processing phases.

        Returns:
            Formatted string with all logs
        """
        log_parts = []

        if self.read_log:
            log_parts.append("=== DATA LOADING ===")
            log_parts.extend(self.read_log)

        if self.filter_log:
            log_parts.append("\n=== FILTERING ===")
            log_parts.extend(self.filter_log)

        if self.align_log:
            log_parts.append("\n=== ALIGNMENT ===")
            log_parts.extend(self.align_log)

        return '\n'.join(log_parts)

    def summary(self, print_output: bool = True) -> str:
        """
        Generate a human-readable summary of the synchronization result.

        Provides an overview of:
        - Experiment name and format
        - Number of timepoints and features
        - Shape of each aligned array with alignment mode used
        - Aggregated alignment mode statistics
        - Any warnings from the logs

        Args:
            print_output: If True, print the summary. If False, only return it.

        Returns:
            Formatted summary string

        Example:
            >>> result.summary()
            ═══════════════════════════════════════════════════════
            SYNCHRONIZATION SUMMARY
            ═══════════════════════════════════════════════════════
            Experiment: RT_A04_S01
            Format: new
            Timepoints: 17855
            Features: 12

            Initial Data:
              Activity: 17855 frames @ 30.0 FPS
              Behavior: 53568 frames @ 90.0 FPS

            Aligned Data:
              Calcium          (251, 17855)    [neuronal]
              Spikes           (251, 17855)    [neuronal]
              Speed            (17855,)        [2 timelines]
              X                (17855,)        [2 timelines]
              ...

            Alignment Modes:
              2 timelines: 8 features
              cast_to_ca: 2 features

            Warnings: None
            ═══════════════════════════════════════════════════════
        """
        lines = []
        sep = "═" * 55

        lines.append(sep)
        lines.append("SYNCHRONIZATION SUMMARY")
        lines.append(sep)

        # Experiment info
        exp_name = self.sync_info.get('experiment_name', 'Unknown')
        src_format = self.sync_info.get('source_format', 'Unknown')
        lines.append(f"Experiment: {exp_name}")
        lines.append(f"Format: {src_format}")

        # Timepoints
        try:
            n_timepoints = self.get_timepoints()
            lines.append(f"Timepoints: {n_timepoints}")
        except ValueError as e:
            lines.append(f"Timepoints: ERROR - {e}")

        # Feature count
        lines.append(f"Features: {len(self.aligned_data)}")

        # Parse initial data lengths from read_log
        activity_frames = None
        activity_fps = None
        behavior_frames = None
        behavior_fps = None

        # Flatten read_log (may be list of strings or list with one multi-line string)
        read_log_lines = []
        for item in self.read_log:
            read_log_lines.extend(item.split('\n'))

        for log_line in read_log_lines:
            # Parse: "Activity: {n_neurons} neurons x {n_frames} frames"
            if 'Activity:' in log_line and 'neurons x' in log_line:
                try:
                    activity_frames = int(log_line.split('x')[1].split('frames')[0].strip())
                except (IndexError, ValueError):
                    pass
            # Parse: "Behavior: {n_frames} frames, {n_features} features"
            elif 'Behavior:' in log_line and 'frames,' in log_line:
                try:
                    behavior_frames = int(log_line.split('Behavior:')[1].split('frames')[0].strip())
                except (IndexError, ValueError):
                    pass
            # Parse: "Activity estimated FPS: {fps}"
            elif 'Activity estimated FPS:' in log_line:
                try:
                    activity_fps = float(log_line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
            # Parse: "Behavior estimated FPS: {fps}"
            elif 'Behavior estimated FPS:' in log_line:
                try:
                    behavior_fps = float(log_line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass

        # Show initial data lengths
        if activity_frames or behavior_frames:
            lines.append("")
            lines.append("Initial Data:")
            if activity_frames:
                fps_str = f" @ {activity_fps:.1f} FPS" if activity_fps else ""
                lines.append(f"  Activity: {activity_frames} frames{fps_str}")
            if behavior_frames:
                fps_str = f" @ {behavior_fps:.1f} FPS" if behavior_fps else ""
                lines.append(f"  Behavior: {behavior_frames} frames{fps_str}")

        # Parse alignment modes from align_log
        feature_modes = {}
        for log_line in self.align_log:
            if 'using "' in log_line and 'Feature "' in log_line:
                try:
                    # Parse: Feature "name": reason, using "mode" mode
                    feature_name = log_line.split('Feature "')[1].split('"')[0]
                    mode = log_line.split('using "')[1].split('"')[0]
                    feature_modes[feature_name] = mode
                except (IndexError, ValueError):
                    pass

        # Aligned data shapes
        lines.append("")
        lines.append("Aligned Data:")

        # Determine neuronal vs behavioral features
        neuro_keys = {'Calcium', 'Spikes', 'Reconstructions'}

        for key, arr in sorted(self.aligned_data.items()):
            shape_str = str(arr.shape)
            if key in neuro_keys:
                mode_str = "[neuronal]"
            else:
                mode = feature_modes.get(key, "unknown")
                mode_str = f"[{mode}]"

            lines.append(f"  {key:<20} {shape_str:<18} {mode_str}")

        # Mode statistics
        mode_stats = self.sync_info.get('mode_stats', {})
        if mode_stats:
            lines.append("")
            lines.append("Alignment Modes:")
            for mode, count in sorted(mode_stats.items(), key=lambda x: -x[1]):
                lines.append(f"  {mode}: {count} feature(s)")

        # Warnings
        warnings = []
        all_logs = self.read_log + self.filter_log + self.align_log
        for log_line in all_logs:
            log_lower = log_line.lower()
            if 'warning' in log_lower or 'error' in log_lower:
                # Clean up the warning text
                clean_line = log_line.strip()
                if clean_line and clean_line not in warnings:
                    warnings.append(clean_line)

        lines.append("")
        if warnings:
            lines.append(f"Warnings ({len(warnings)}):")
            for w in warnings[:5]:  # Show first 5 warnings
                lines.append(f"  - {w[:70]}{'...' if len(w) > 70 else ''}")
            if len(warnings) > 5:
                lines.append(f"  ... and {len(warnings) - 5} more")
        else:
            lines.append("Warnings: None")

        lines.append(sep)

        summary_str = '\n'.join(lines)

        if print_output:
            print(summary_str)

        return summary_str

    def __repr__(self) -> str:
        n_features = len(self.aligned_data)
        try:
            n_timepoints = self.get_timepoints()
            return f"SyncResult(features={n_features}, timepoints={n_timepoints})"
        except ValueError:
            # Data has inconsistent timepoint lengths across features
            return f"SyncResult(features={n_features}, timepoints=inconsistent)"
