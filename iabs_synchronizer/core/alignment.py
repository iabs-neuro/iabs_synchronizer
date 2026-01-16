"""
Core synchronization and alignment logic for neuroscience data.

This module implements the 5 alignment regimes for synchronizing multi-modal
neuroscience data to a common temporal reference (calcium imaging timeline).

⚠️ CRITICAL MODULE - DO NOT MODIFY INTERPOLATION LOGIC
The _interpolate_to_timeline() function is the core quality guarantee.
Any changes to the interpolation algorithm will affect synchronization quality.

Source: Lines 599-750 from IABS_data_synchronizer_1_3.ipynb
"""

from typing import Tuple, Dict, Optional, Any, List
from fractions import Fraction
from itertools import permutations
import numpy as np
from scipy.interpolate import interp1d

from ..config import (
    ALLOWED_FPS,
    ALIGN_PRECISION,
    NEURO_DATA_PARTS,
    SyncConfig
)
from ..utils.logging import LogCapture


def _is_close(num1: float, num2: float, threshold: Optional[float] = None) -> bool:
    """
    Check if two numbers are within precision threshold of each other.

    Uses relative difference check against ALIGN_PRECISION threshold (default 2%).
    This allows fuzzy matching for lengths that should be equal but have minor differences.

    Args:
        num1: First number to compare
        num2: Second number to compare
        threshold: Optional custom threshold (default: ALIGN_PRECISION from config)

    Returns:
        bool: True if numbers are within threshold, False otherwise
    """
    if threshold is None:
        threshold = ALIGN_PRECISION

    delta = np.abs(num1 - num2)
    if 1.0 * delta / num1 > threshold or 1.0 * delta / num2 > threshold:
        return False
    else:
        return True


def _calculate_temporal_overlap(timeline1: np.ndarray, timeline2: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate the temporal intersection of two timelines.

    This enforces the Golden Rule: "We keep ONLY timeline parts where BOTH activity and behavior exist"

    Args:
        timeline1: First timeline array (e.g., behavioral timestamps)
        timeline2: Second timeline array (e.g., calcium timestamps)

    Returns:
        tuple: (min_overlap, max_overlap) if overlap exists, None otherwise
    """
    min_overlap = max(np.min(timeline1), np.min(timeline2))
    max_overlap = min(np.max(timeline1), np.max(timeline2))

    if min_overlap >= max_overlap:
        return None  # No temporal overlap

    return (min_overlap, max_overlap)


def _trim_to_temporal_overlap(data: np.ndarray,
                                timeline: np.ndarray,
                                overlap_bounds: Tuple[float, float],
                                feature_name: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim time series data and timeline to overlap region.

    Enforces the Golden Rule by discarding data outside the temporal overlap.
    Issues warnings when significant data loss occurs (>5% by default).

    Args:
        data: Time series array to trim
        timeline: Timeline array for the data
        overlap_bounds: (min_time, max_time) tuple defining the overlap region
        feature_name: Optional name for logging purposes

    Returns:
        tuple: (trimmed_data, trimmed_timeline)
    """
    min_overlap, max_overlap = overlap_bounds

    # Create mask for points within overlap region
    mask = (timeline >= min_overlap) & (timeline <= max_overlap)

    # Trim data and timeline
    trimmed_data = data[mask]
    trimmed_timeline = timeline[mask]

    # Calculate data loss percentage
    original_duration = np.max(timeline) - np.min(timeline)
    overlap_duration = max_overlap - min_overlap
    data_loss_pct = 100 * (1 - overlap_duration / original_duration)

    # Log warning if significant data loss (>5%)
    if data_loss_pct > 5.0:
        lost_duration = original_duration - overlap_duration
        name_str = f"{feature_name} " if feature_name else ""
        print(f"WARNING: {name_str}trimmed {data_loss_pct:.1f}% "
              f"({lost_duration:.2f}s) outside overlap range "
              f"[{min_overlap:.2f}s, {max_overlap:.2f}s]")

    return trimmed_data, trimmed_timeline


def _calculate_global_temporal_overlap(
    behavioral_features: List[Dict[str, Any]],
    target_timeline: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate global temporal overlap across all behavioral features and calcium.

    Returns the intersection of all timeline ranges - the time period where
    ALL features AND calcium have data.

    Args:
        behavioral_features: List of feature dicts with 'timeline' keys
        target_timeline: Calcium timeline array

    Returns:
        (global_min, global_max): Tuple defining common overlap region

    Raises:
        ValueError: If no global overlap exists (no common time period)

    Example:
        >>> # Calcium: [0, 100], Feature A: [10, 90], Feature B: [20, 95]
        >>> global_overlap = _calculate_global_temporal_overlap(...)
        >>> print(global_overlap)
        (20.0, 90.0)  # Intersection of all ranges
    """
    # Start with calcium timeline bounds
    global_min = float(np.min(target_timeline))
    global_max = float(np.max(target_timeline))

    # Narrow to intersection with each feature's timeline
    for feature_dict in behavioral_features:
        if feature_dict.get('timeline') is not None:
            feature_min = float(np.min(feature_dict['timeline']))
            feature_max = float(np.max(feature_dict['timeline']))

            # Update global bounds to intersection
            global_min = max(global_min, feature_min)
            global_max = min(global_max, feature_max)

    # Check if any overlap exists
    if global_min >= global_max:
        raise ValueError(
            f"No global temporal overlap found. "
            f"Calcium and behavioral features have no common time period."
        )

    return (global_min, global_max)


def length_match_different_fps(num1: int, num2: int, target_fps: Optional[int] = None, allowed_fps: Optional[list] = None) -> Tuple[Fraction, bool]:
    """
    Detect if two different lengths are related by an FPS ratio.

    Checks all permutations of ALLOWED_FPS values to see if num1 * (fps1/fps2) ≈ num2.
    If target_fps is specified, only checks ratios involving that FPS.

    Args:
        num1: First length (e.g., behavioral data length)
        num2: Second length (e.g., calcium data length)
        target_fps: Optional target FPS to constrain search
        allowed_fps: Optional list of allowed FPS values (default: ALLOWED_FPS from config)

    Returns:
        tuple: (ratio, match_found) where:
            - ratio: Fraction representing the best FPS ratio found
            - match_found: True if a matching ratio was found, False otherwise
    """
    if allowed_fps is None:
        allowed_fps = ALLOWED_FPS

    best_coeff, match_found = 0, False

    if target_fps is None:
        coeffs = set([1.0 * pair[0] / pair[1] for pair in permutations(allowed_fps, 2)])
    else:
        coeffs = set([1.0 * pair[0] / pair[1] for pair in permutations(allowed_fps, 2) if target_fps in pair])

    for coeff in coeffs:
        if _is_close(int(num1 * coeff), num2):
            best_coeff = coeff
            match_found = True

    return Fraction(best_coeff).limit_denominator(), match_found


def _interpolate_to_timeline(ts: np.ndarray, timeline: np.ndarray, target_timeline: np.ndarray) -> np.ndarray:
    """
    Resample time series from source timeline to target timeline using linear interpolation.

    ⚠️ CRITICAL FUNCTION - MODIFIED TO WORK WITH TRIMMED DATA
    This function assumes timeline and target_timeline have already been trimmed to
    their temporal overlap region by _trim_to_temporal_overlap().

    The algorithm:
    1. Validates that timelines are within compatible bounds
    2. Applies scipy linear interpolation directly on the timelines
    3. Evaluates interpolated function at target timeline points

    The direct interpolation (without rescaling) ensures:
    - No distortion of temporal relationships
    - Proper handling of temporal overlap (trimming happens before interpolation)
    - Errors are raised if extrapolation is attempted (bounds_error=True)

    Args:
        ts: Time series array to interpolate (1D array)
        timeline: Source timeline (timestamps for ts) - MUST be trimmed to overlap
        target_timeline: Target timeline (desired output timestamps) - MUST be trimmed to overlap

    Returns:
        np.ndarray: Interpolated time series matching target_timeline length

    Raises:
        ValueError: If target_timeline requests extrapolation beyond timeline bounds
    """
    if target_timeline is None:
        return ts

    # Direct interpolation without rescaling
    # bounds_error=True ensures we catch any extrapolation attempts
    f = interp1d(timeline, ts, kind='linear', bounds_error=True)
    ats = f(target_timeline)

    return ats


def _align_with_two_timelines(ts: np.ndarray,
                              target_length: int,
                              timeline: np.ndarray,
                              target_timeline: np.ndarray) -> np.ndarray:
    """
    Strategy: Align using both timelines with temporal overlap.

    Implements the Golden Rule: Keep ONLY timeline parts where BOTH data sources exist.

    Args:
        ts: Source time series data
        target_length: Target length (not used in this mode, kept for interface consistency)
        timeline: Source timeline
        target_timeline: Target timeline

    Returns:
        Aligned time series matching the overlapping portion of target_timeline

    Raises:
        ValueError: If no temporal overlap exists between timelines
    """
    # Check if timelines are already perfectly aligned
    if len(timeline) == len(target_timeline) and np.allclose(timeline, target_timeline):
        return ts  # Perfect match, no interpolation needed

    # Calculate temporal overlap (Golden Rule: keep ONLY where BOTH exist)
    overlap_bounds = _calculate_temporal_overlap(timeline, target_timeline)

    if overlap_bounds is None:
        raise ValueError(
            f"No temporal overlap between data timeline "
            f"[{np.min(timeline):.2f}s, {np.max(timeline):.2f}s] and "
            f"target timeline [{np.min(target_timeline):.2f}s, {np.max(target_timeline):.2f}s]"
        )

    # Trim source data and timeline to overlap region
    trimmed_ts, trimmed_timeline = _trim_to_temporal_overlap(
        ts, timeline, overlap_bounds, feature_name="source data"
    )

    # Trim target timeline to overlap region
    min_overlap, max_overlap = overlap_bounds
    target_mask = (target_timeline >= min_overlap) & (target_timeline <= max_overlap)
    trimmed_target_timeline = target_timeline[target_mask]

    # Ensure target timeline is strictly within source timeline bounds
    # (handle floating-point precision issues by clipping to source bounds)
    trimmed_target_timeline = np.clip(
        trimmed_target_timeline,
        np.min(trimmed_timeline),
        np.max(trimmed_timeline)
    )

    # Now interpolate within the safe overlap region
    return _interpolate_to_timeline(trimmed_ts, trimmed_timeline, trimmed_target_timeline)


def _align_simple(ts: np.ndarray,
                  target_length: int,
                  timeline: Optional[np.ndarray],
                  target_timeline: Optional[np.ndarray]) -> np.ndarray:
    """
    Strategy: Simple pass-through (requires exact length match).

    Use when data is already same length and aligned with target.

    Args:
        ts: Source time series data
        target_length: Expected length
        timeline: Not used (kept for interface consistency)
        target_timeline: Not used (kept for interface consistency)

    Returns:
        Original time series unchanged

    Raises:
        ValueError: If lengths don't match
    """
    if len(ts) != target_length:
        raise ValueError(f"'simple' mode requires matching lengths: got {len(ts)}, expected {target_length}")
    return ts


def _align_cast_to_ca(ts: np.ndarray,
                      target_length: int,
                      timeline: Optional[np.ndarray],
                      target_timeline: np.ndarray) -> np.ndarray:
    """
    Strategy: Cast/interpolate to calcium timeline (MOST COMMON mode).

    Creates synthetic timeline for behavior data matching calcium's time range,
    then interpolates to calcium's timeline.

    Args:
        ts: Source time series data
        target_length: Not used directly (uses target_timeline length)
        timeline: Not used (synthetic timeline created)
        target_timeline: Target timeline to interpolate to

    Returns:
        Interpolated time series matching target_timeline length
    """
    # Create synthetic timeline for behavior data matching calcium's time range
    mint, maxt = min(target_timeline), max(target_timeline)
    synthetic_timeline = np.linspace(mint, maxt, num=len(ts))
    return _interpolate_to_timeline(ts, synthetic_timeline, target_timeline)


def _align_crop(ts: np.ndarray,
                target_length: int,
                timeline: Optional[np.ndarray],
                target_timeline: Optional[np.ndarray]) -> np.ndarray:
    """
    Strategy: Crop to target length.

    Trims excess data from the end to match target length.

    Args:
        ts: Source time series data
        target_length: Target length to crop to
        timeline: Not used (kept for interface consistency)
        target_timeline: Not used (kept for interface consistency)

    Returns:
        Cropped time series of target_length

    Raises:
        ValueError: If source data is shorter than target
    """
    if len(ts) < target_length:
        raise ValueError('behaviour data shorter than calcium')

    if len(ts) > target_length:
        print(f'cropped {len(ts) - target_length} values from the end')

    return ts[:target_length]


def _align_factor(ts: np.ndarray,
                  target_length: int,
                  timeline: Optional[np.ndarray],
                  target_timeline: Optional[np.ndarray]) -> np.ndarray:
    """
    Strategy: Factor-based alignment (NOT IMPLEMENTED).

    Reserved for future FPS-based scaling.

    Args:
        ts: Source time series data
        target_length: Target length
        timeline: Source timeline
        target_timeline: Target timeline

    Raises:
        NotImplementedError: Always (not yet implemented)
    """
    raise NotImplementedError('Merging data with known fps difference factor is not implemented yet')


def align_data(ts: np.ndarray,
               target_length: int,
               timeline: Optional[np.ndarray],
               target_timeline: Optional[np.ndarray],
               factor: float = 1,
               mode: str = 'simple') -> np.ndarray:
    """
    Align time series to target using specified synchronization mode.

    Uses Strategy pattern to delegate to appropriate alignment algorithm.

    Implements 5 different alignment regimes:

    **Mode 1: '2 timelines'**
    - Use when: Data has its own timeline that may differ from calcium's timeline
    - Algorithm: If timelines match exactly → pass through; otherwise → interpolate to target
    - Use case: Behavioral data with explicit timestamps

    **Mode 2: 'simple'**
    - Use when: Data is already same length and aligned with calcium
    - Algorithm: Pass through unchanged
    - Use case: Pre-aligned data streams

    **Mode 3: 'cast_to_ca'** (MOST COMMON)
    - Use when: Behavioral data has different sampling rate but is temporally synchronized
    - Algorithm: Create synthetic timeline spanning calcium's time range, then interpolate
    - Use case: Primary mode for behavioral-to-calcium synchronization

    **Mode 4: 'crop'**
    - Use when: Behavioral data is longer than calcium and needs trimming
    - Algorithm: Trim to target length (crop excess from end)
    - Use case: Longer recordings that need to match calcium duration

    **Mode 5: 'factor'**
    - Use when: Data has known FPS ratio difference from calcium
    - Algorithm: Not yet implemented (raises NotImplementedError)
    - Use case: Reserved for future FPS-based scaling

    Args:
        ts: Source time series array to align
        target_length: Target length (from calcium data)
        timeline: Source timeline (may be None)
        target_timeline: Target timeline (from calcium)
        factor: FPS conversion factor (for 'factor' mode, not yet implemented)
        mode: Alignment mode - one of ['2 timelines', 'simple', 'cast_to_ca', 'crop', 'factor']

    Returns:
        np.ndarray: Aligned time series matching target_length

    Raises:
        ValueError: If mode is unknown or alignment fails
        NotImplementedError: If using 'factor' mode (not yet implemented)
    """
    # Default target timeline if None provided
    if target_timeline is None:
        target_timeline = np.linspace(0, target_length, target_length)

    # Strategy pattern: map mode names to implementation functions
    strategies = {
        '2 timelines': _align_with_two_timelines,
        'simple': _align_simple,
        'cast_to_ca': _align_cast_to_ca,
        'crop': _align_crop,
        'factor': _align_factor
    }

    if mode not in strategies:
        raise ValueError(
            f"Unknown alignment mode: {mode}. "
            f"Must be one of {list(strategies.keys())}"
        )

    # Call the selected strategy
    strategy_func = strategies[mode]
    return strategy_func(ts, target_length, timeline, target_timeline)


def _select_alignment_mode(
    ts_length: int,
    target_length: int,
    timeline: Optional[np.ndarray],
    target_timeline: Optional[np.ndarray],
    force_pathway: Optional[str],
    feature_name: str
) -> Tuple[str, str]:
    """
    Select appropriate alignment mode based on data characteristics.

    Decision tree (in priority order):
    1. If force_pathway provided → use that mode
    2. If both timelines present → '2 timelines' mode (most accurate)
    3. If exact length match → 'simple' mode (no alignment needed)
    4. If approximately close length → 'crop' mode (trim excess)
    5. If both timelines missing → 'cast_to_ca' mode (fallback)
    6. If only one timeline present → ERROR (suspicious, likely data problem)

    Args:
        ts_length: Length of source time series
        target_length: Length of target (calcium) time series
        timeline: Source timeline (may be None)
        target_timeline: Target timeline (may be None)
        force_pathway: Optional forced mode (overrides automatic selection)
        feature_name: Name of feature being aligned (for error messages)

    Returns:
        tuple: (mode, reason) where:
            - mode: Selected alignment mode string
            - reason: Human-readable explanation of why this mode was chosen

    Raises:
        ValueError: If no suitable alignment mode can be determined, or if only one timeline is present

    Examples:
        >>> mode, reason = _select_alignment_mode(100, 100, None, None, None, "speed")
        >>> print(f"{mode}: {reason}")
        'simple': Exact length match (100 == 100)

        >>> mode, reason = _select_alignment_mode(1500, 1000, tl1, tl2, None, "position")
        >>> print(f"{mode}: {reason}")
        '2 timelines': Both timelines present (most accurate)

        >>> mode, reason = _select_alignment_mode(3000, 1000, None, None, None, "speed")
        >>> print(f"{mode}: {reason}")
        'cast_to_ca': Fallback - no timelines, assuming same time span (3000 -> 1000)
    """
    # Priority 1: Forced mode overrides all automatic selection
    if force_pathway is not None:
        return force_pathway, f"Forced mode: {force_pathway}"

    # Priority 2: Both timelines present - use temporal overlap alignment
    if target_timeline is not None and timeline is not None:
        return '2 timelines', "Both timelines present (most accurate)"

    # Priority 3: Exact length match - no alignment needed (regardless of timeline availability)
    if ts_length == target_length:
        return 'simple', f"Exact length match ({ts_length} == {target_length})"

    # Priority 4: Approximately close length - crop to match
    if _is_close(ts_length, target_length):
        return 'crop', f"Approximately close length ({ts_length} ≈ {target_length}, will crop)"

    # Priority 5: Both timelines missing - use cast_to_ca as fallback
    # ASSUMPTION: Data spans the same time period (common in simultaneous recordings)
    # align_data() will create synthetic timelines for interpolation
    if target_timeline is None and timeline is None:
        return 'cast_to_ca', f"Fallback - no timelines, assuming same time span ({ts_length} -> {target_length} samples) [WARNING: Verify recordings started/ended simultaneously!]"

    # Priority 6: Exactly one timeline present - ERROR (suspicious configuration)
    # Having only one timeline suggests a data problem or incomplete information
    # User should either provide both timelines (ideal) or neither (acceptable with assumptions)
    if target_timeline is not None and timeline is None:
        raise ValueError(
            f"Cannot automatically align feature '{feature_name}': "
            f"Only target timeline present (source timeline missing). "
            f"This is suspicious - either provide both timelines or neither. "
            f"To force alignment anyway, use force_mode='cast_to_ca'."
        )

    if target_timeline is None and timeline is not None:
        raise ValueError(
            f"Cannot automatically align feature '{feature_name}': "
            f"Only source timeline present (target timeline missing). "
            f"This is suspicious - either provide both timelines or neither. "
            f"To force alignment anyway, use force_mode='cast_to_ca'."
        )


def align_all_data(filtered_data: Dict[str, Any],
                   force_pathway: Optional[str] = None,
                   config: Optional[SyncConfig] = None) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Orchestrate synchronization of all data pieces to calcium reference timeline.

    This is the main alignment function that:
    1. Extracts reference parameters from Calcium data (timeline, fps, length)
    2. For each data piece:
       - If neuronal (Calcium/Spikes): pass through unchanged
       - If behavioral: automatically select alignment mode or use forced mode
    3. Collects all aligned arrays into a flat dictionary

    **Automatic Mode Selection** (when force_pathway=None):
    - Priority 1: If both source and target have timelines → '2 timelines' mode
    - Priority 2: If lengths exactly match → 'simple' mode
    - Priority 3: If lengths approximately match (within ALIGN_PRECISION) → 'crop' mode
    - Priority 4: If lengths match FPS ratio from ALLOWED_FPS → 'factor' mode
    - Failure: If none of above, report error and skip

    Args:
        filtered_data: Dictionary of filtered data from filtering phase
                       Format: {data_piece: {data, timeline, fps} or [{feature: data, timeline, fps}, ...]}
        force_pathway: Optional mode to force for all data (overrides automatic selection)
        config: Optional SyncConfig object with custom settings

    Returns:
        tuple: (aligned_data, align_log) where:
            - aligned_data: Flat dictionary mapping feature names to aligned arrays
            - align_log: List of log messages from alignment process

    Raises:
        KeyError: If 'Calcium' data is not found in filtered_data
    """
    if config is None:
        neuro_data_parts = NEURO_DATA_PARTS
    else:
        neuro_data_parts = config.neuro_data_parts

    with LogCapture() as align_log:
        # Extract target parameters from Calcium data
        target_timeline = filtered_data['Calcium']['timeline']
        target_fps = filtered_data['Calcium']['fps']
        target_length = filtered_data['Calcium']['Calcium'].shape[1]

        aligned_data = {}

        if target_timeline is None:
            print('Processing without calcium data timeline...')

        # Get list of active data pieces (those with data)
        active_data_pieces = [dp for dp, data in filtered_data.items() if data is not None and len(data) > 0]

        # Calculate global temporal overlap for features with timelines
        if target_timeline is not None:
            # Collect all behavioral features that have timelines
            features_with_timelines = []
            for dp in active_data_pieces:
                if dp not in neuro_data_parts:
                    for tsdata in filtered_data[dp]:
                        if tsdata.get('timeline') is not None:
                            features_with_timelines.append(tsdata)

            # Calculate global overlap and create common target timeline
            if features_with_timelines:
                try:
                    global_min, global_max = _calculate_global_temporal_overlap(
                        features_with_timelines,
                        target_timeline
                    )

                    # Trim calcium timeline to global overlap
                    global_mask = (target_timeline >= global_min) & (target_timeline <= global_max)
                    common_target_timeline = target_timeline[global_mask]
                    common_target_length = len(common_target_timeline)

                    # Log data loss if significant
                    original_duration = np.max(target_timeline) - np.min(target_timeline)
                    overlap_duration = global_max - global_min
                    data_loss_pct = 100 * (1 - overlap_duration / original_duration)

                    if data_loss_pct > 0.1:  # Log even minor trimming for transparency
                        print(f'[GLOBAL OVERLAP] Using common time range [{global_min:.2f}s, {global_max:.2f}s]')
                        if data_loss_pct > 5.0:
                            print(f'[GLOBAL OVERLAP] Trimmed {data_loss_pct:.1f}% of calcium timeline '
                                  f'to ensure all features have data')

                except ValueError as e:
                    # No global overlap - this is a critical error
                    print(f'[ERROR] {str(e)}')
                    raise
            else:
                # No features with timelines - use full calcium timeline
                common_target_timeline = target_timeline
                common_target_length = target_length
        else:
            common_target_timeline = target_timeline
            common_target_length = target_length

        for dp in active_data_pieces:
            # Neuronal data: trim to global overlap if applied
            if dp in neuro_data_parts:
                if filtered_data[dp] is not None:
                    neuro_data = filtered_data[dp][dp]  # Shape: (neurons, timepoints)

                    # Trim to global overlap if calculated
                    if common_target_length != target_length:
                        # Trim neuronal data to match global overlap
                        aligned_data[dp] = neuro_data[:, :common_target_length]
                    else:
                        aligned_data[dp] = neuro_data
                continue  # Situation where spike series have different length/fps from calcium is not currently implemented

            print(f'Processing {dp}...')

            # Behavioral data: iterate through features
            for tsdata in filtered_data[dp]:
                # Extract feature name (the key that's not 'timeline' or 'fps')
                name = [_ for _ in list(tsdata.keys()) if _ != 'timeline' and _ != 'fps'][0]
                ts = tsdata[name]
                timeline = tsdata['timeline']

                # Select alignment mode (use COMMON target length and timeline)
                mode, reason = _select_alignment_mode(
                    ts_length=len(ts),
                    target_length=common_target_length,
                    timeline=timeline,
                    target_timeline=common_target_timeline,
                    force_pathway=force_pathway,
                    feature_name=name
                )

                # Log the decision
                print(f'Feature "{name}": {reason}, using "{mode}" mode')

                # Perform alignment (use COMMON target length and timeline)
                ats = align_data(ts, common_target_length, timeline, common_target_timeline, mode=mode)

                if ats is not None:
                    aligned_data[name] = ats

    return aligned_data, align_log.getvalue().split('\n')
