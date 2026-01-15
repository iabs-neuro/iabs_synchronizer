"""
Quality filtering for neuroscience data synchronization.

This module handles filtering of time series data based on quality thresholds,
removing NaN values, validating data lengths, and structuring data for alignment.

Source: Lines 508-598 from IABS_data_synchronizer_1_3.ipynb
"""

from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd

from ..config import (
    TOO_MANY_NANS_THR,
    TOO_SHORT_TS_THR,
    NEURO_DATA_PARTS,
    DEFAULT_DATA_PIECES,
    SyncConfig
)
from ..utils.logging import LogCapture


def _filter_data_from_nans(ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify and filter NaN values from time series.

    Args:
        ts: Time series array (can contain NaNs)

    Returns:
        tuple: (invalid_indices, valid_vals) where:
            - invalid_indices: boolean array marking NaN positions
            - valid_vals: array with NaN values removed
    """
    series = pd.Series(ts)
    invalid_indices = series.isna().values
    return invalid_indices, series[~invalid_indices].values


def _rescale_timeline_if_needed(timeline: np.ndarray) -> np.ndarray:
    """
    Rescale timeline from milliseconds to seconds if needed.

    Args:
        timeline: Timeline array (may be in milliseconds or seconds)

    Returns:
        Timeline in seconds
    """
    if np.mean(np.diff(timeline)) > 1:
        return timeline / 1000
    return timeline


def _infer_data_orientation(data: np.ndarray, invalid_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """
    Infer and normalize data orientation.

    Determines whether data is in (neurons, timepoints) or (timepoints, neurons) format
    and filters NaN indices if provided.

    Heuristic: If first dimension < second dimension, assume (neurons, timepoints) format.
    This works because typically neurons << timepoints in neuroscience recordings.

    Args:
        data: 2D array to normalize
        invalid_indices: Optional boolean mask of invalid indices to filter

    Returns:
        tuple: (normalized_data, was_transposed) where:
            - normalized_data: Always in (neurons, timepoints) format
            - was_transposed: Whether data was transposed from original orientation

    Note:
        This heuristic may fail with square arrays (n_neurons ≈ n_timepoints),
        but such cases are rare in practice.
    """
    needs_transpose = data.shape[0] >= data.shape[1]

    if invalid_indices is not None:
        # Filter data based on orientation
        if needs_transpose:
            # Legacy (timepoints, neurons) format - filter along axis 0
            filtered_data = data[~invalid_indices, :]
        else:
            # Already in (neurons, timepoints) format - filter along axis 1
            filtered_data = data[:, ~invalid_indices]
    else:
        filtered_data = data

    # Ensure final format is (neurons, timepoints)
    normalized_data = filtered_data.T if needs_transpose else filtered_data

    return normalized_data, needs_transpose


def _check_1d_ts(ts: np.ndarray, target_length: int, fps: Optional[int] = None, config: Optional[SyncConfig] = None) -> Tuple[bool, str]:
    """
    Validate individual time series against quality thresholds.

    Checks for:
    - Too many NaN values (>TOO_MANY_NANS_THR%)
    - Non-numeric values
    - Too short time series (<TOO_SHORT_TS_THR% of target length)

    Args:
        ts: Time series array to validate
        target_length: Target length (from calcium data) for comparison
        fps: Optional frames-per-second value
        config: Optional SyncConfig object with custom thresholds

    Returns:
        tuple: (status_ok, reason) where:
            - status_ok: True if validation passed, False otherwise
            - reason: Empty string if passed, error description if failed
    """
    if config is None:
        too_many_nans_thr = TOO_MANY_NANS_THR
        too_short_ts_thr = TOO_SHORT_TS_THR
    else:
        too_many_nans_thr = config.too_many_nans_thr
        too_short_ts_thr = config.too_short_ts_thr

    status = True
    reason = ''

    # Check for too many NaNs
    invalid_indices, valid_vals = _filter_data_from_nans(ts)
    nan_percent = 100.0 * (1 - len(valid_vals) / len(ts))
    if nan_percent > too_many_nans_thr:
        status = False
        reason = f'too many NaNs: {int(nan_percent)}% of values (emergency threshold is {too_many_nans_thr}%)'
        return status, reason

    # Check for non-numeric values
    try:
        valid_vals = valid_vals.astype(float)
    except (ValueError, TypeError):
        # Data contains non-numeric values that can't be converted to float
        status = False
        reason = 'Non-numeric values'
        return status, reason

    # Check for too short time series
    if len(ts) < too_short_ts_thr / 100. * target_length:
        status = False
        reason = f'too short time series: {len(ts)} points with {target_length} in calcium data'

    return status, reason


def _filter_neuronal_data(data_piece: str,
                          extracted_data: List[Tuple],
                          config: Optional[SyncConfig]) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Filter neuronal data (Calcium, Spikes) for a single data piece.

    Args:
        data_piece: Name of data piece ('Calcium' or 'Spikes')
        extracted_data: List of extracted data tuples for this piece
        config: SyncConfig object with filtering parameters

    Returns:
        tuple: (filtered_dict, length) where:
            - filtered_dict: {data_piece: array, 'timeline': array, 'fps': int} or None if no data
            - length: Number of timepoints (for calcium length tracking) or None
    """
    if len(extracted_data) == 0:
        return None, None

    data, names, timeline, fps = extracted_data[0]

    if timeline is not None:
        # Remove NaN points in timeline and data
        invalid_time_indices, valid_time_vals = _filter_data_from_nans(timeline)
        timeline = timeline[~invalid_time_indices]

        # Infer data orientation and normalize to (neurons, timepoints) format
        final_data, was_transposed = _infer_data_orientation(data, invalid_time_indices)

        # Rescale timeline if values suggest milliseconds
        timeline = _rescale_timeline_if_needed(timeline)
    else:
        # No timeline - infer orientation only
        final_data, was_transposed = _infer_data_orientation(data)

    filtered_dict = {
        data_piece: final_data,
        'timeline': timeline,
        'fps': fps
    }

    # Return timepoint length for calcium length tracking
    length = final_data.shape[1]

    return filtered_dict, length


def _filter_behavioral_data(data_piece: str,
                            extracted_data: List[Tuple],
                            ca_length: Optional[int],
                            config: Optional[SyncConfig]) -> List[Dict[str, Any]]:
    """
    Filter behavioral data for a single data piece.

    Args:
        data_piece: Name of data piece (e.g., 'Behavior_auto')
        extracted_data: List of extracted data tuples for this piece
        ca_length: Target length from calcium data (for validation), or None
        config: SyncConfig object with filtering parameters

    Returns:
        List of filtered feature dictionaries, each containing:
            {feature_name: array, 'timeline': array, 'fps': int}
    """
    filtered_features = []

    for info in extracted_data:
        data, names, timeline, fps = info

        for i, name in enumerate(names):
            # Validate this feature
            if ca_length is not None:
                status_ok, reason = _check_1d_ts(data[:, i], fps=fps, target_length=ca_length, config=config)
            else:
                # If no calcium data yet, skip length validation
                status_ok = True
                reason = ''

            if status_ok:
                # Remove NaN points in data
                invalid_indices, valid_vals = _filter_data_from_nans(data[:, i])

                if timeline is not None:
                    # Remove NaN points in timeline as well
                    invalid_time_indices, valid_time_vals = _filter_data_from_nans(timeline)
                    # Remove points that are NaN in either data or timeline
                    combined_invalid = invalid_indices | invalid_time_indices
                    timeline_clean = timeline[~combined_invalid]
                    valid_data = data[:, i][~combined_invalid]

                    # Rescale timeline if values suggest milliseconds
                    timeline_clean = _rescale_timeline_if_needed(timeline_clean)
                else:
                    timeline_clean = None
                    valid_data = valid_vals

                # Fix numeric feature names by prepending data piece name
                if str(name).isnumeric():
                    name = data_piece + '_' + str(name)

                fdict = {
                    name: valid_data,
                    'timeline': timeline_clean,
                    'fps': fps
                }

                filtered_features.append(fdict)
            else:
                print(f'[INFO] Time series "{name}" discarded - reason: {reason}')

    return filtered_features


def _apply_gap_detection(filtered_info: Dict[str, Any],
                        active_data_pieces: List[str],
                        neuro_data_parts: List[str],
                        config: SyncConfig) -> Dict[str, Any]:
    """
    Apply gap detection to filtered data and optionally exclude low-coverage features.

    Args:
        filtered_info: Dictionary of filtered data
        active_data_pieces: List of active data pieces
        neuro_data_parts: List of neuronal data piece names
        config: SyncConfig with gap detection parameters

    Returns:
        Updated filtered_info with low-coverage features removed (if configured)
    """
    from ..utils.gap_detection import detect_timeline_gaps

    print('\n[Gap Detection] Checking timeline quality...')
    gap_report = {}
    features_with_gaps = []

    # Check neuronal data timelines
    for dp in active_data_pieces:
        if dp in neuro_data_parts and dp in filtered_info:
            if 'timeline' in filtered_info[dp] and filtered_info[dp]['timeline'] is not None:
                timeline = filtered_info[dp]['timeline']
                has_gaps, gap_info = detect_timeline_gaps(
                    timeline,
                    gap_threshold_multiplier=config.gap_threshold_multiplier
                )

                if has_gaps:
                    features_with_gaps.append(dp)
                    gap_report[dp] = gap_info

                    if config.warn_on_gaps:
                        print(f'  [WARNING] {dp} has {gap_info["n_gaps"]} gap(s), '
                              f'coverage: {gap_info["data_coverage"]:.1f}%')

    # Check behavioral data timelines
    for dp in active_data_pieces:
        if dp not in neuro_data_parts and dp in filtered_info:
            features_to_remove = []

            for i, feature_dict in enumerate(filtered_info[dp]):
                feature_name = [k for k in feature_dict.keys() if k not in ['timeline', 'fps']][0]

                if 'timeline' in feature_dict and feature_dict['timeline'] is not None:
                    timeline = feature_dict['timeline']
                    has_gaps, gap_info = detect_timeline_gaps(
                        timeline,
                        gap_threshold_multiplier=config.gap_threshold_multiplier
                    )

                    if has_gaps:
                        features_with_gaps.append(feature_name)
                        gap_report[feature_name] = gap_info

                        if config.warn_on_gaps:
                            print(f'  [WARNING] {feature_name} has {gap_info["n_gaps"]} gap(s), '
                                  f'coverage: {gap_info["data_coverage"]:.1f}%')

                        # Optionally exclude features with low coverage
                        if config.exclude_low_coverage:
                            if gap_info['data_coverage'] < config.min_coverage_threshold:
                                print(f'  [INFO] Excluding {feature_name} (coverage {gap_info["data_coverage"]:.1f}% '
                                      f'< threshold {config.min_coverage_threshold}%)')
                                features_to_remove.append(i)

            # Remove low-coverage features
            for i in sorted(features_to_remove, reverse=True):
                del filtered_info[dp][i]

    # Summary
    if features_with_gaps:
        print(f'[Gap Detection] Found gaps in {len(features_with_gaps)} feature(s)')
    else:
        print('[Gap Detection] No gaps detected in any timeline')

    return filtered_info


def filter_data(active_data_pieces: List[str],
                extracted_info: Dict[str, List[Tuple]],
                config: Optional[SyncConfig] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Filter and structure all data pieces for synchronization.

    Main filtering entry point that:
    1. Filters neuronal data (Calcium, Spikes) - removes NaN time indices, rescales timeline
    2. Filters behavioral data - validates each feature, removes NaNs
    3. Structures data appropriately for alignment phase

    Neuronal data structure (dict):
        {data_piece: array(n_neurons, n_timepoints),
         'timeline': array(n_timepoints),
         'fps': int}

    Behavioral data structure (list of dicts):
        [{feature_name: array(n_timepoints),
          'timeline': array(n_timepoints),
          'fps': int}, ...]

    Args:
        active_data_pieces: List of data pieces that have data for this experiment
        extracted_info: Dictionary mapping data piece names to extracted data tuples
        config: Optional SyncConfig object with custom filtering thresholds

    Returns:
        tuple: (filtered_info, filt_log) where:
            - filtered_info: Dictionary of filtered/structured data ready for alignment
            - filt_log: List of log messages from filtering process

    Raises:
        KeyError: If 'Calcium' data is not found (required for target length)
    """
    if config is None:
        neuro_data_parts = NEURO_DATA_PARTS
        data_pieces = DEFAULT_DATA_PIECES
    else:
        neuro_data_parts = config.neuro_data_parts
        data_pieces = config.default_data_pieces

    filtered_info = {dp: [] for dp in data_pieces}

    with LogCapture() as filt_log:
        ca_length = None  # Will be set from Calcium data

        for dp in active_data_pieces:
            print(f'Processing {dp}...')

            # Process neuronal data (Calcium, Spikes)
            if dp in neuro_data_parts:
                filtered_dict, length = _filter_neuronal_data(dp, extracted_info[dp], config)
                if filtered_dict is not None:
                    filtered_info[dp] = filtered_dict

                    # Get calcium length as target for behavioral data validation
                    if dp == 'Calcium':
                        ca_length = length

            # Process behavioral data
            else:
                filtered_features = _filter_behavioral_data(dp, extracted_info[dp], ca_length, config)
                filtered_info[dp] = filtered_features

            # Warn if no data passed filters for this data piece
            if len(filtered_info[dp]) == 0:
                print('-----------------------------------------------------------------------------------')
                print(f'No data from {dp} folder passed filters, please consider manual feature extraction')
                print('-----------------------------------------------------------------------------------')

    # Apply gap detection if enabled
    if config.detect_gaps:
        filtered_info = _apply_gap_detection(filtered_info, active_data_pieces, neuro_data_parts, config)

    return filtered_info, filt_log.getvalue().split('\n')
