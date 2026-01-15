"""
Gap detection utilities for timeline analysis.

Detects recording gaps (interruptions) in time series data while being robust
to sampling noise and outliers.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np


def detect_timeline_gaps(timeline: np.ndarray,
                         gap_threshold_multiplier: float = 3.0,
                         min_gap_duration: Optional[float] = None,
                         outlier_percentile: float = 90.0) -> Tuple[bool, Dict]:
    """
    Detect gaps in timeline sampling while being robust to outliers.

    A "gap" is defined as an interval between consecutive timestamps that is
    significantly larger than the typical sampling interval. This indicates
    recording was paused/stopped and then resumed.

    Algorithm:
    1. Calculate all time differences between consecutive timestamps
    2. Estimate typical interval using robust statistics (ignoring outliers)
    3. Identify gaps where diff > gap_threshold * typical_interval
    4. Report gap locations and statistics

    Args:
        timeline: Array of timestamps (must be sorted ascending)
        gap_threshold_multiplier: A gap is detected when interval is this many
                                  times larger than typical interval (default: 3.0)
        min_gap_duration: Minimum duration (seconds) to consider a gap. If None,
                         uses gap_threshold_multiplier * typical_interval
        outlier_percentile: Percentile below which to calculate typical interval
                           (default: 90.0 means use 10th-90th percentile range)

    Returns:
        tuple: (has_gaps, gap_info) where:
            - has_gaps: bool indicating if gaps were found
            - gap_info: dict with keys:
                * 'n_gaps': Number of gaps found
                * 'gap_indices': List of indices where gaps start
                * 'gap_durations': List of gap durations (seconds)
                * 'total_gap_duration': Total time lost to gaps (seconds)
                * 'typical_interval': Estimated typical sampling interval
                * 'estimated_fps': Estimated FPS (robust to gaps)
                * 'gap_locations': List of (start_time, end_time) tuples
                * 'data_coverage': Percentage of time range with actual data
                * 'warnings': List of warning messages

    Raises:
        ValueError: If timeline is too short (< 2 points) or not sorted

    Example:
        >>> timeline = np.array([0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7])
        >>> has_gaps, info = detect_timeline_gaps(timeline)
        >>> print(f"Found {info['n_gaps']} gaps")
        Found 1 gaps
        >>> print(f"Gap duration: {info['gap_durations'][0]:.2f}s")
        Gap duration: 1.20s
    """
    if len(timeline) < 2:
        raise ValueError(f"Timeline too short for gap detection: {len(timeline)} points (need >= 2)")

    if not np.all(np.diff(timeline) >= 0):
        raise ValueError("Timeline must be sorted in ascending order")

    # Calculate all time differences
    diffs = np.diff(timeline)

    if len(diffs) == 0:
        return False, {
            'n_gaps': 0,
            'gap_indices': [],
            'gap_durations': [],
            'total_gap_duration': 0.0,
            'typical_interval': 0.0,
            'estimated_fps': 0.0,
            'gap_locations': [],
            'data_coverage': 100.0,
            'warnings': ['Timeline has only 1 point']
        }

    # Robust estimation of typical interval (ignore outliers/gaps)
    # Use percentile-based approach: typical interval is around the lower percentiles
    lower_percentile = 100.0 - outlier_percentile
    typical_interval = np.percentile(diffs, lower_percentile)

    # Alternative: use median of smaller diffs
    # This is more robust when there are many gaps
    median_diff = np.median(diffs)
    if median_diff < typical_interval * 2:
        # If median is reasonable, use it (more robust)
        typical_interval = median_diff

    # Handle edge case: all diffs are identical (perfect sampling)
    if np.allclose(diffs, diffs[0]):
        typical_interval = diffs[0]

    # Estimate FPS from typical interval
    estimated_fps = 1.0 / typical_interval if typical_interval > 0 else 0.0

    # Determine gap threshold
    if min_gap_duration is not None:
        gap_threshold = min_gap_duration
    else:
        gap_threshold = gap_threshold_multiplier * typical_interval

    # Detect gaps
    gap_mask = diffs > gap_threshold
    gap_indices = np.where(gap_mask)[0]

    has_gaps = len(gap_indices) > 0

    # Calculate gap statistics
    gap_durations = diffs[gap_mask].tolist() if has_gaps else []
    total_gap_duration = sum(gap_durations)

    # Gap locations (start_time, end_time)
    gap_locations = []
    for idx in gap_indices:
        start_time = timeline[idx]
        end_time = timeline[idx + 1]
        gap_locations.append((start_time, end_time))

    # Data coverage: what % of time range has actual data
    total_time_range = timeline[-1] - timeline[0]
    actual_data_time = total_time_range - total_gap_duration
    data_coverage = (actual_data_time / total_time_range * 100.0) if total_time_range > 0 else 100.0

    # Generate warnings
    warnings = []
    if typical_interval <= 0:
        warnings.append("Could not estimate typical sampling interval")
    if len(timeline) < 10:
        warnings.append(f"Timeline very short ({len(timeline)} points) - gap detection may be unreliable")
    if data_coverage < 50:
        warnings.append(f"Low data coverage ({data_coverage:.1f}%) - more than half the time range has gaps")

    gap_info = {
        'n_gaps': len(gap_indices),
        'gap_indices': gap_indices.tolist(),
        'gap_durations': gap_durations,
        'total_gap_duration': total_gap_duration,
        'typical_interval': typical_interval,
        'estimated_fps': estimated_fps,
        'gap_locations': gap_locations,
        'data_coverage': data_coverage,
        'warnings': warnings
    }

    return has_gaps, gap_info


def print_gap_report(timeline: np.ndarray,
                     gap_info: Dict,
                     feature_name: str = "Timeline") -> None:
    """
    Print a human-readable gap detection report.

    Args:
        timeline: The timeline array that was analyzed
        gap_info: Gap information dict from detect_timeline_gaps()
        feature_name: Name of the feature for display purposes
    """
    print(f"\n{'=' * 60}")
    print(f"Gap Detection Report: {feature_name}")
    print(f"{'=' * 60}")

    print(f"\nTimeline statistics:")
    print(f"  Total points: {len(timeline)}")
    print(f"  Time range: {timeline[0]:.3f}s to {timeline[-1]:.3f}s ({timeline[-1] - timeline[0]:.3f}s)")
    print(f"  Estimated FPS: {gap_info['estimated_fps']:.2f}")
    print(f"  Typical interval: {gap_info['typical_interval']:.4f}s")

    if gap_info['warnings']:
        print(f"\nWarnings:")
        for warning in gap_info['warnings']:
            print(f"  - {warning}")

    if gap_info['n_gaps'] == 0:
        print(f"\n[OK] No gaps detected - timeline appears continuous")
    else:
        print(f"\n[WARNING] {gap_info['n_gaps']} gap(s) detected:")
        print(f"  Total gap duration: {gap_info['total_gap_duration']:.3f}s")
        print(f"  Data coverage: {gap_info['data_coverage']:.1f}%")

        print(f"\n  Gap details:")
        for i, (start, end) in enumerate(gap_info['gap_locations']):
            duration = gap_info['gap_durations'][i]
            idx = gap_info['gap_indices'][i]
            print(f"    Gap #{i+1}: [{start:.3f}s -> {end:.3f}s] = {duration:.3f}s (after index {idx})")


def check_timeline_quality(timeline: np.ndarray,
                           feature_name: str = "data",
                           warn: bool = True,
                           raise_on_gaps: bool = False) -> Dict:
    """
    Check timeline quality and optionally warn or raise errors if gaps found.

    Convenience function that combines gap detection with configurable
    warning/error behavior for pipeline integration.

    Args:
        timeline: Array of timestamps
        feature_name: Name of feature for error messages
        warn: If True, print warnings when gaps are detected
        raise_on_gaps: If True, raise ValueError when gaps are detected

    Returns:
        dict: Gap information from detect_timeline_gaps()

    Raises:
        ValueError: If raise_on_gaps=True and gaps are detected

    Example:
        >>> # In data loading pipeline:
        >>> gap_info = check_timeline_quality(timeline, "Speed", warn=True)
        >>> if gap_info['n_gaps'] > 0:
        >>>     print(f"Warning: {feature_name} has recording gaps!")
    """
    has_gaps, gap_info = detect_timeline_gaps(timeline)

    if has_gaps:
        message = (
            f"Timeline quality issue: '{feature_name}' has {gap_info['n_gaps']} gap(s). "
            f"Total gap duration: {gap_info['total_gap_duration']:.3f}s. "
            f"Data coverage: {gap_info['data_coverage']:.1f}%. "
            f"This may indicate recording was paused/interrupted."
        )

        if raise_on_gaps:
            raise ValueError(message)
        elif warn:
            print(f"WARNING: {message}")

    return gap_info


def calculate_robust_fps(timeline: np.ndarray,
                        allowed_fps: Optional[List[int]] = None) -> Tuple[float, bool]:
    """
    Calculate FPS robustly, ignoring outliers and gaps.

    Uses gap detection to identify the typical sampling interval, then
    attempts to match it to known FPS values if provided.

    Args:
        timeline: Array of timestamps
        allowed_fps: Optional list of expected FPS values to match against

    Returns:
        tuple: (fps, is_confident) where:
            - fps: Estimated frames per second
            - is_confident: True if FPS matches a known value or sampling is very regular

    Example:
        >>> timeline = np.array([0, 0.05, 0.1, 0.15, 1.0, 1.05, 1.1])  # Gap at 0.15-1.0
        >>> fps, confident = calculate_robust_fps(timeline, allowed_fps=[20, 30, 40])
        >>> print(f"FPS: {fps}, Confident: {confident}")
        FPS: 20, Confident: True
    """
    if len(timeline) < 2:
        return 0.0, False

    # Use gap detection to get robust interval estimate
    has_gaps, gap_info = detect_timeline_gaps(timeline)

    estimated_fps = gap_info['estimated_fps']

    if estimated_fps <= 0:
        return 0.0, False

    # Check confidence
    diffs = np.diff(timeline)

    # If no gaps detected, check regularity of sampling
    if not has_gaps:
        # Calculate coefficient of variation
        cv = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else float('inf')
        is_regular = cv < 0.05  # Less than 5% variation
    else:
        # With gaps, less confident
        is_regular = False

    # Try to match to known FPS values
    if allowed_fps is not None:
        for fps in allowed_fps:
            if abs(estimated_fps - fps) / fps < 0.05:  # Within 5%
                return float(fps), True

    # Return estimated FPS with confidence based on regularity
    return estimated_fps, is_regular


# Utility for batch gap detection
def detect_gaps_in_dataset(filtered_data: Dict,
                           warn_on_gaps: bool = True,
                           return_summary: bool = True) -> Dict:
    """
    Check all timelines in filtered dataset for gaps.

    Convenience function for checking multiple features at once.

    Args:
        filtered_data: Filtered data dict (from filter_data())
        warn_on_gaps: Print warnings for features with gaps
        return_summary: Return summary statistics

    Returns:
        dict: Summary of gap detection results
            Keys: feature names
            Values: gap_info dicts from detect_timeline_gaps()

    Example:
        >>> gap_summary = detect_gaps_in_dataset(filtered_data)
        >>> for feature, info in gap_summary.items():
        >>>     if info['n_gaps'] > 0:
        >>>         print(f"{feature}: {info['n_gaps']} gaps")
    """
    results = {}

    for data_piece, data in filtered_data.items():
        if data is None:
            continue

        # Handle neuronal data (dict with single timeline)
        if isinstance(data, dict) and 'timeline' in data:
            timeline = data.get('timeline')
            if timeline is not None and len(timeline) > 0:
                has_gaps, gap_info = detect_timeline_gaps(timeline)
                results[data_piece] = gap_info

                if warn_on_gaps and has_gaps:
                    print(f"WARNING: {data_piece} has {gap_info['n_gaps']} gap(s) "
                          f"({gap_info['total_gap_duration']:.2f}s total)")

        # Handle behavioral data (list of feature dicts)
        elif isinstance(data, list):
            for feature_data in data:
                timeline = feature_data.get('timeline')
                if timeline is not None and len(timeline) > 0:
                    # Get feature name
                    feature_name = [k for k in feature_data.keys()
                                  if k not in ['timeline', 'fps']][0]
                    full_name = f"{data_piece}.{feature_name}"

                    has_gaps, gap_info = detect_timeline_gaps(timeline)
                    results[full_name] = gap_info

                    if warn_on_gaps and has_gaps:
                        print(f"WARNING: {full_name} has {gap_info['n_gaps']} gap(s) "
                              f"({gap_info['total_gap_duration']:.2f}s total)")

    return results
