"""
Global configuration constants for data synchronization.

These values control quality thresholds and synchronization behavior.
Modify with caution - changing these affects synchronization quality.
"""

from typing import List, Optional


# Allowed frames-per-second values for automatic FPS detection
ALLOWED_FPS = [10, 20, 30, 40, 50]

# Quality thresholds (non-negotiable for maximum synchronization quality)
TOO_MANY_NANS_THR = 5  # Reject time series with >5% NaN values
TOO_SHORT_TS_THR = 10  # Reject series <10% of calcium length
MAX_UNIQUE_VALS = 0.02  # Flag if >2% timestamp differences are unique (strange data)
ALIGN_PRECISION = 0.02  # Max 2% length difference for auto-alignment

# Data piece classifications
NEURO_DATA_PARTS = ['Calcium', 'Spikes', 'Reconstructions']
DEFAULT_DATA_PIECES = ['Calcium', 'Spikes', 'Reconstructions', 'Behavior_auto', 'Behavior_manual', 'Coordinates']

# New format file suffixes
# Maps file suffixes to their logical data type
# Format: {expname}_{suffix} -> data type
NEW_FORMAT_SUFFIXES = {
    'data.npz': 'activity_data',
    'Features.csv': 'behavior_features',
    'Mini_TS.csv': 'activity_timeline',
    'VT_TS.csv': 'behavior_timeline',
    'metadata.json': 'metadata'
}


class SyncConfig:
    """
    Configuration object for customizing synchronization parameters.

    All threshold values can be adjusted to suit specific experimental requirements.
    Default values are optimized for maximum synchronization quality.

    Attributes:
        allowed_fps: List of acceptable FPS values for automatic detection
        too_many_nans_thr: Percentage threshold for rejecting data with too many NaNs
        too_short_ts_thr: Percentage threshold for rejecting data that's too short
        max_unique_vals: Threshold for detecting unexpected unique values in categorical data
        align_precision: Maximum allowed percentage difference for automatic alignment
        neuro_data_parts: List of data piece types considered as neuronal data
        default_data_pieces: Default list of data pieces to load
        detect_gaps: Enable automatic gap detection during filtering
        warn_on_gaps: Print warnings when gaps are detected
        gap_threshold_multiplier: Gap threshold as multiple of typical interval (default: 3.0)
        exclude_low_coverage: Automatically exclude features with low data coverage
        min_coverage_threshold: Minimum data coverage percentage to keep feature (default: 50.0%)
        file_suffixes: New format file suffixes mapping (default: NEW_FORMAT_SUFFIXES)

    Example:
        >>> # Create custom configuration with stricter alignment
        >>> config = SyncConfig(align_precision=0.01)  # 1% tolerance instead of 2%
        >>>
        >>> # Use different allowed FPS values
        >>> config = SyncConfig(allowed_fps=[20, 30, 40])
    """

    def __init__(self,
                 allowed_fps: Optional[List[int]] = None,
                 too_many_nans_thr: Optional[float] = None,
                 too_short_ts_thr: Optional[float] = None,
                 max_unique_vals: Optional[float] = None,
                 align_precision: Optional[float] = None,
                 neuro_data_parts: Optional[List[str]] = None,
                 default_data_pieces: Optional[List[str]] = None,
                 detect_gaps: bool = True,
                 warn_on_gaps: bool = True,
                 gap_threshold_multiplier: float = 3.0,
                 exclude_low_coverage: bool = False,
                 min_coverage_threshold: float = 50.0,
                 file_suffixes: Optional[dict] = None):
        """
        Initialize synchronization configuration.

        Args:
            allowed_fps: Allowed FPS values (default: [10, 20, 30, 40, 50])
            too_many_nans_thr: Max NaN percentage (default: 5%)
            too_short_ts_thr: Min length percentage (default: 10%)
            max_unique_vals: Max unique value ratio (default: 0.02)
            align_precision: Max length difference ratio (default: 0.02)
            neuro_data_parts: Neuronal data identifiers (default: ['Calcium', 'Spikes', 'Reconstructions'])
            default_data_pieces: Data pieces to load (default: full list)
            detect_gaps: Enable gap detection during filtering (default: True)
            warn_on_gaps: Print warnings for detected gaps (default: True)
            gap_threshold_multiplier: Gap = N × typical interval (default: 3.0)
            exclude_low_coverage: Auto-exclude features with low coverage (default: False)
            min_coverage_threshold: Min coverage % to keep feature (default: 50.0)
            file_suffixes: Custom file suffix mapping for new format (default: NEW_FORMAT_SUFFIXES)
        """
        self.allowed_fps = allowed_fps if allowed_fps is not None else ALLOWED_FPS.copy()
        self.too_many_nans_thr = too_many_nans_thr if too_many_nans_thr is not None else TOO_MANY_NANS_THR
        self.too_short_ts_thr = too_short_ts_thr if too_short_ts_thr is not None else TOO_SHORT_TS_THR
        self.max_unique_vals = max_unique_vals if max_unique_vals is not None else MAX_UNIQUE_VALS
        self.align_precision = align_precision if align_precision is not None else ALIGN_PRECISION
        self.neuro_data_parts = neuro_data_parts if neuro_data_parts is not None else NEURO_DATA_PARTS.copy()
        self.default_data_pieces = default_data_pieces if default_data_pieces is not None else DEFAULT_DATA_PIECES.copy()

        # Gap detection settings
        self.detect_gaps = detect_gaps
        self.warn_on_gaps = warn_on_gaps
        self.gap_threshold_multiplier = gap_threshold_multiplier
        self.exclude_low_coverage = exclude_low_coverage
        self.min_coverage_threshold = min_coverage_threshold

        # New format file suffixes
        self.file_suffixes = file_suffixes if file_suffixes is not None else NEW_FORMAT_SUFFIXES.copy()

    def __repr__(self) -> str:
        return (f"SyncConfig(allowed_fps={self.allowed_fps}, "
                f"too_many_nans_thr={self.too_many_nans_thr}%, "
                f"align_precision={self.align_precision})")
