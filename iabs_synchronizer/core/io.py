"""
Data loading and I/O operations for neuroscience synchronization.

This module handles reading CSV/XLSX files, detecting timelines, calculating FPS,
and extracting data arrays with automatic error correction.

Source: Lines 251-507 from IABS_data_synchronizer_1_3.ipynb
"""

import os
from os.path import join, splitext
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd

from ..config import (
    ALLOWED_FPS,
    MAX_UNIQUE_VALS,
    NEURO_DATA_PARTS,
    DEFAULT_DATA_PIECES
)
from ..utils.logging import LogCapture


def _convert_timeline_to_seconds(timeline: np.ndarray, name: str) -> Tuple[np.ndarray, str]:
    """
    Convert timeline from Windows ticks to seconds if detected.

    Windows ticks are 100-nanosecond intervals. This function detects timelines
    with values > 1e9 and median differences > 100000 (indicating tick format)
    and converts them to seconds by dividing by 1e7.

    Args:
        timeline: Timeline array (may be in ticks or seconds)
        name: Name for logging (e.g., "Activity timeline", "Behavior timeline")

    Returns:
        tuple: (converted_timeline, log_message)
            - converted_timeline: Timeline in seconds
            - log_message: Description of conversion performed
    """
    if timeline[0] > 1e9:  # Likely ticks or nanoseconds
        # Check if diffs suggest ticks (100ns intervals, ~10MHz)
        median_diff = np.median(np.diff(timeline))
        if median_diff > 100000:  # Likely Windows ticks (100ns units)
            timeline = timeline / 1e7  # Convert to seconds
            return timeline, f"{name}: {len(timeline)} points (converted from ticks)"
        else:
            return timeline, f"{name}: {len(timeline)} points"
    else:
        return timeline, f"{name}: {len(timeline)} points"


def _no_header(df: pd.DataFrame, dp: str) -> bool:
    """
    Check if DataFrame has a proper header or if the first row is data.

    Detects header errors by checking if column names are numeric values.
    For neuronal data (Calcium/Spikes), integer headers are allowed but floats indicate error.
    For other data types, any numeric headers indicate missing header row.

    Args:
        df: pandas DataFrame to check
        dp: data piece type (e.g., 'Calcium', 'Behavior_auto')

    Returns:
        bool: True if header is missing/invalid, False if header is valid
    """
    cols = df.columns.values
    float_in_headers = []
    number_in_headers = []

    for col in cols:
        fl = False
        number = False
        try:
            fl = (float(col) != int(float(col)))
            number = True
        except (ValueError, TypeError):
            # Column header is not numeric - this is expected for non-numeric headers
            pass
        float_in_headers.append(fl)
        number_in_headers.append(number)

    float_present = np.any(float_in_headers)  # No one in their right mind would name columns with fractional numbers - means the row has shifted to data
    numeric_present = np.any(number_in_headers)

    if dp in NEURO_DATA_PARTS:  # Neurons can have numerical headers in the form of integers, but not fractions
        no_header = float_present
    else:
        no_header = numeric_present or float_present  # All others cannot have integers either

    if no_header:
        print('[WARNING] Header error detected (float in header), auto-fixing...')

    return no_header


def read_table(path: str, fname: str, dp: str, skip_header: bool = False, header_checked: bool = False) -> pd.DataFrame:
    """
    Read CSV or XLSX file with automatic header detection.

    Supports both CSV and XLSX formats. Automatically detects and fixes
    missing headers by recursively re-reading the file without header inference.

    Args:
        path: Directory path containing the file
        fname: Filename to read
        dp: Data piece type (used for header validation)
        skip_header: If True, read without header row (internal use)
        header_checked: If True, skip header validation (internal use)

    Returns:
        pandas.DataFrame: Loaded data with proper headers

    Raises:
        ValueError: If file extension is not .csv or .xlsx
    """
    fpath = join(path, fname)
    if splitext(fname)[1] == '.csv':
        header = None if skip_header else 'infer'
        df = pd.read_csv(fpath, header=header)
    elif splitext(fname)[1] == '.xlsx':
        df = pd.read_excel(fpath)
    else:
        raise ValueError(f'Unknown extension of file {fname}')

    if not header_checked and _no_header(df, dp):
        df = read_table(path, fname, dp, skip_header=True, header_checked=True)

    return df


def _find_column_by_name(df: pd.DataFrame, name: str) -> Optional[str]:
    """
    Find column containing specified name (case-insensitive).

    Args:
        df: DataFrame to search
        name: Name substring to search for

    Returns:
        Column name if found, None otherwise
    """
    cols = df.columns.values
    matches = cols[np.array([name in str(c).lower() for c in cols])]
    if len(matches) > 0:
        return matches[0]
    return None


def _find_monotonic_columns(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """
    Find monotonically increasing columns (timeline candidates).

    Args:
        df: DataFrame to analyze

    Returns:
        tuple: (candidates, counters) where:
            - candidates: List of column indices that are monotonically increasing
            - counters: List of column indices that are simple integer counters (diff=1)
    """
    candidates = []
    counters = []

    for i, col in enumerate(df.columns):
        data = df[col].values
        try:
            if np.all(np.diff(data) > 0):
                candidates.append(i)
                # Check if it's a simple integer counter (1, 2, 3, ...)
                if set(np.diff(data)) == {1}:
                    counters.append(i)
        except (TypeError, ValueError):
            # Column contains non-numeric data or operation fails - skip this column
            pass

    return candidates, counters


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Automatically detect the time/timestamp column in a DataFrame.

    Detection strategy:
    1. First tries to find column with 'time' in name (case-insensitive)
    2. If not found, searches for monotonically increasing columns
    3. Filters out integer counters (columns with constant diff of 1)
    4. Returns the most likely timeline column

    Args:
        df: pandas DataFrame to analyze

    Returns:
        str or None: Column name of detected timeline, or None if no timeline found
    """
    cols = df.columns.values

    # Try to find column with 'time' in name
    time_col = _find_column_by_name(df, 'time')
    if time_col is not None:
        print(f'Assuming column "{time_col}" with index {list(cols).index(time_col)} as a timeline')
        print()
        return time_col

    # Search for monotonically increasing columns
    print('No specified time column found, searching for a timeline without name...')
    candidates, counters = _find_monotonic_columns(df)

    if len(candidates) == 0:
        print('No column in the data looks like a timeline, assuming data has no timestamps')
        print()
        return None

    # Report findings
    print(f'Columns with indices {candidates} can be interpreted as timelines')
    if len(counters) != 0:
        print(f'Columns with indices {counters} look suspiciously like integer counters')

    # Filter out counter columns
    final = list(set(candidates) - set(counters))

    if len(final) == 0:
        print('No column in the data looks like a timeline, assuming data has no timestamps')
        print()
        return None

    # Return first non-counter timeline candidate
    time_col = cols[final[0]]
    if len(final) == 1:
        print(f'Assuming column "{time_col}" with index {final[0]} as a timeline')
    else:
        print(f'[WARNING] Assuming the first column "{time_col}" with index {final[0]} out of possible index set {final} as a timeline - this may cause errors')

    print()
    return time_col


def _calc_fps(vals: np.ndarray) -> int:
    """
    Calculate FPS from timestamp values using robust gap-aware method.

    Uses median/percentile-based estimation that ignores gaps and outliers.
    This replaces the old naive averaging approach with the robust version
    from gap_detection.py.

    Args:
        vals: numpy array of timestamp values

    Returns:
        int: Detected frames-per-second value
    """
    from ..utils.gap_detection import calculate_robust_fps

    # Use robust FPS calculation that handles gaps and outliers
    fps_float, is_confident = calculate_robust_fps(vals, allowed_fps=ALLOWED_FPS)

    # Warn if estimation is not confident
    if not is_confident:
        print('[WARNING] FPS estimation not confident (possible gaps or irregular sampling)')

    fps = int(np.round(fps_float))

    print(f'Automatically determined fps: {fps}')
    if fps not in ALLOWED_FPS:
        print(f'[WARNING] Automatically determined fps {fps} is suspicious - consider manual check')

    return fps


def calc_fps(df: pd.DataFrame, time_col: Optional[str]) -> Optional[int]:
    """
    Calculate frames-per-second from DataFrame's time column.

    Wrapper around _calc_fps with error handling. Returns None if
    time_col is None or calculation fails.

    Args:
        df: pandas DataFrame containing time column
        time_col: Name of time column, or None if no timeline

    Returns:
        int or None: Frames-per-second value, or None if calculation failed
    """
    try:
        if time_col is not None:
            fps = _calc_fps(df[time_col].values)
            return fps
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        print(f'[ERROR] FPS computation failed: {e}')
        return None


def get_data(df: pd.DataFrame, time_col: Optional[str], data_type: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract data arrays and feature names from DataFrame.

    Removes time column and any columns to its left (assumed irrelevant).
    For neuronal data (Calcium/Spikes), returns data without feature names.
    For other data types, returns feature names from column headers.

    Args:
        df: pandas DataFrame with data
        time_col: Name of time column (or None if no timeline)
        data_type: Type of data piece (e.g., 'Calcium', 'Behavior_auto')

    Returns:
        tuple: (X, names) where:
            - X is numpy array of shape (n_samples, n_features)
            - names is array of feature names (or None for neuronal data)
    """
    cols = df.columns.values
    if time_col is not None:
        time_index = list(cols).index(time_col)
        if time_index > 0:
            print(f'All columns left to time column "{time_col}" disregarded as irrelevant')
    else:
        time_index = -1

    X = df.to_numpy()[:, time_index + 1:]

    print(data_type)
    if data_type not in NEURO_DATA_PARTS:
        names = cols[time_index + 1:]
        print(f'found {len(names)} features with names {cols[time_index + 1:]} with length {X.shape[0]}')
    else:
        names = None
        print(f'found {X.shape[1]} neuronal traces with length {X.shape[0]}')

    print()
    return X, names


def auto_analysis(df: pd.DataFrame, dp: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """
    Perform complete automated analysis of a DataFrame.

    Combines timeline detection, FPS calculation, and data extraction
    into a single pipeline. This is the main analysis function called
    for each loaded data file.

    Args:
        df: pandas DataFrame to analyze
        dp: Data piece type (e.g., 'Calcium', 'Behavior_auto')

    Returns:
        tuple: (data, names, timeline, fps) where:
            - data: numpy array of shape (n_samples, n_features)
            - names: array of feature names (or None for neuronal data)
            - timeline: numpy array of timestamps (or None if no timeline)
            - fps: frames-per-second value (or None if not calculated)
    """
    time_col = find_time_column(df)
    fps = calc_fps(df, time_col)
    data, names = get_data(df, time_col, dp)

    timeline = None
    if time_col is not None:
        timeline = df[time_col].values

    return data, names, timeline, fps


def load_data_part(dp: str, expname: str, root: str = '/content') -> Dict[str, pd.DataFrame]:
    """
    Load all data files for a specific data piece.

    Scans directory for data piece and loads all CSV/XLSX files found.
    Handles missing directories gracefully, with special handling for
    required Calcium data.

    Args:
        dp: Data piece name (e.g., 'Calcium', 'Behavior_auto')
        expname: Experiment name (subdirectory)
        root: Root directory path (default: '/content')

    Returns:
        dict: Dictionary mapping filenames to DataFrames

    Raises:
        FileNotFoundError: If Calcium data is missing (cannot proceed without it)
    """
    print(f'{dp}:')
    dfs = dict()
    path = join(root, expname, dp)

    try:
        for fname in os.listdir(path):
            df = read_table(path, fname, dp)
            print(f'Successfully loaded data from "{fname}"')
            dfs[fname] = df

    except (FileNotFoundError, PermissionError, OSError) as e:
        if dp == 'Calcium':
            raise FileNotFoundError('No calcium data found, cannot proceed further!') from e
        else:
            # Non-calcium data is optional - log the issue but continue
            print(f'[WARNING] Could not load {dp} data from {path}: {e}')
            pass

    print()
    return dfs


def _load_activity_data_new_format(expname: str, get_path_func, log_lines: List[str]) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Load and validate activity data from new format.

    Args:
        expname: Experiment name
        get_path_func: Function to get path for file type
        log_lines: List to append log messages

    Returns:
        tuple: (arrays_dict, n_neurons, n_frames)

    Raises:
        ValueError: If no data found or shapes mismatch
    """
    data_path = get_path_func('activity_data', f"{expname}_data.npz")
    activity_data = np.load(data_path)

    # Extract available arrays
    arrays = {}
    if 'C' in activity_data:
        arrays['C'] = activity_data['C']
    if 'spikes' in activity_data:
        arrays['spikes'] = activity_data['spikes']
    if 'reconstructions' in activity_data:
        arrays['reconstructions'] = activity_data['reconstructions']

    if not arrays:
        raise ValueError(f"No activity data found in {data_path}")

    # Validate shapes match
    shapes = {name: arr.shape for name, arr in arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(
            f"Activity arrays have mismatched shapes: {shapes}. "
            f"All activity data (C, spikes, reconstructions) must have identical shapes."
        )

    n_neurons, n_frames = list(arrays.values())[0].shape
    log_lines.append(f"Activity: {n_neurons} neurons x {n_frames} frames")

    return arrays, n_neurons, n_frames


def _load_activity_timeline_new_format(expname: str, get_path_func, n_frames: int,
                                       log_lines: List[str]) -> Optional[np.ndarray]:
    """
    Load and convert activity timeline from new format.

    Args:
        expname: Experiment name
        get_path_func: Function to get path for file type
        n_frames: Expected number of frames
        log_lines: List to append log messages

    Returns:
        Activity timeline array or None if missing

    Raises:
        ValueError: If timeline length doesn't match n_frames
    """
    mini_ts_path = get_path_func('activity_timeline', f"{expname}_Mini_TS.csv")

    if not os.path.exists(mini_ts_path):
        log_lines.append("WARNING: Activity timeline missing")
        return None

    activity_timeline = pd.read_csv(mini_ts_path, header=None).values.flatten()
    if len(activity_timeline) != n_frames:
        raise ValueError(
            f"Activity timeline length ({len(activity_timeline)}) "
            f"doesn't match activity data ({n_frames})"
        )

    # Convert to seconds if needed
    activity_timeline, log_msg = _convert_timeline_to_seconds(activity_timeline, "Activity timeline")
    log_lines.append(log_msg)

    return activity_timeline


def _load_behavior_data_new_format(expname: str, get_path_func,
                                    log_lines: List[str]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Load behavior features and timeline from new format.

    Args:
        expname: Experiment name
        get_path_func: Function to get path for file type
        log_lines: List to append log messages

    Returns:
        tuple: (behavior_df, behavior_timeline)

    Raises:
        ValueError: If timeline length doesn't match features
    """
    features_path = get_path_func('behavior_features', f"{expname}_Features.csv")
    behavior_df = pd.read_csv(features_path)
    n_behavior_frames = len(behavior_df)
    log_lines.append(f"Behavior: {n_behavior_frames} frames, {len(behavior_df.columns)} features")

    # Load behavior timeline
    vt_ts_path = get_path_func('behavior_timeline', f"{expname}_VT_TS.csv")
    behavior_timeline = None

    if os.path.exists(vt_ts_path):
        behavior_timeline = pd.read_csv(vt_ts_path, header=None).values.flatten()
        if len(behavior_timeline) != n_behavior_frames:
            raise ValueError(
                f"Behavior timeline length ({len(behavior_timeline)}) "
                f"doesn't match Features.csv ({n_behavior_frames})"
            )

        # Convert to seconds if needed
        behavior_timeline, log_msg = _convert_timeline_to_seconds(behavior_timeline, "Behavior timeline")
        log_lines.append(log_msg)
    else:
        log_lines.append("WARNING: Behavior timeline missing")

    return behavior_df, behavior_timeline


def _load_metadata_new_format(expname: str, get_path_func, log_lines: List[str]) -> Dict:
    """
    Load metadata JSON from new format.

    Args:
        expname: Experiment name
        get_path_func: Function to get path for file type
        log_lines: List to append log messages

    Returns:
        Metadata dictionary (empty if file not found)
    """
    import json

    metadata_path = get_path_func('metadata', f"{expname}_metadata.json")
    metadata = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        log_lines.append(f"Metadata loaded: FPS={metadata.get('fps', 'N/A')}")
    else:
        log_lines.append("WARNING: Metadata file not found")

    return metadata


def _validate_fps_new_format(activity_timeline: Optional[np.ndarray],
                              metadata: Dict, log_lines: List[str]) -> None:
    """
    Validate FPS consistency between timeline and metadata.

    Args:
        activity_timeline: Activity timeline array (or None)
        metadata: Metadata dictionary
        log_lines: List to append log messages
    """
    from ..utils.gap_detection import calculate_robust_fps

    metadata_fps = metadata.get('fps')
    if activity_timeline is not None and metadata_fps is not None:
        timeline_fps, confident = calculate_robust_fps(
            activity_timeline,
            allowed_fps=ALLOWED_FPS
        )

        if not np.isclose(timeline_fps, metadata_fps, rtol=0.05):
            warning = (
                f"FPS mismatch: metadata={metadata_fps:.2f}, "
                f"timeline-derived={timeline_fps:.2f}"
            )
            log_lines.append(f"WARNING: {warning}")
            print(f"[WARNING] {warning}")


def _convert_to_legacy_format(arrays: Dict[str, np.ndarray], n_neurons: int,
                                activity_timeline: Optional[np.ndarray],
                                metadata_fps: Optional[float],
                                behavior_df: pd.DataFrame,
                                behavior_timeline: Optional[np.ndarray]) -> Dict:
    """
    Convert new format data structures to legacy format for pipeline compatibility.

    Args:
        arrays: Dictionary of activity arrays (C, spikes, reconstructions)
        n_neurons: Number of neurons
        activity_timeline: Activity timeline (or None)
        metadata_fps: FPS from metadata (or None)
        behavior_df: Behavior features DataFrame
        behavior_timeline: Behavior timeline (or None)

    Returns:
        extracted_info dictionary in legacy format
    """
    extracted_info = {}

    # Activity data - add all available arrays
    if 'C' in arrays:
        extracted_info['Calcium'] = [(
            arrays['C'],
            [f'neuron_{i}' for i in range(n_neurons)],
            activity_timeline,
            metadata_fps
        )]

    if 'spikes' in arrays:
        extracted_info['Spikes'] = [(
            arrays['spikes'],
            [f'neuron_{i}' for i in range(n_neurons)],
            activity_timeline,
            metadata_fps
        )]

    if 'reconstructions' in arrays:
        extracted_info['Reconstructions'] = [(
            arrays['reconstructions'],
            [f'neuron_{i}' for i in range(n_neurons)],
            activity_timeline,
            metadata_fps
        )]

    # Behavior data
    behavior_data = behavior_df.values  # (n_timepoints, n_features)
    behavior_names = behavior_df.columns.tolist()

    extracted_info['Behavior_auto'] = [(
        behavior_data,
        behavior_names,
        behavior_timeline,
        None  # Behavior FPS derived from timeline if needed
    )]

    return extracted_info


def read_new_format(expname: str, root: str = '.', path_config: Optional[Dict[str, str]] = None) -> Optional[Tuple[Dict, str, Dict]]:
    """
    Load data from new standardized format.

    Expected files:
    - {expname}_data.npz          : Activity (C, spikes, reconstructions)
    - {expname}_Features.csv      : Behavioral features
    - {expname}_Mini_TS.csv       : Activity timeline
    - {expname}_VT_TS.csv         : Behavior timeline
    - {expname}_metadata.json     : Metadata

    Args:
        expname: Experiment name
        root: Default root directory (used if path_config not provided)
        path_config: Optional dict specifying custom paths for each file type:
            {
                'activity_data': '/path/to/imaging',
                'behavior_features': '/path/to/behavior',
                'activity_timeline': '/path/to/timestamps',
                'behavior_timeline': '/path/to/timestamps',
                'metadata': '/path/to/metadata'
            }
            If a key is missing, falls back to 'root'

    Returns:
        tuple: (extracted_info, load_log, metadata) or None if format not found
        - extracted_info: Same format as legacy loader
        - load_log: Loading messages
        - metadata: Parsed metadata dict

    Raises:
        ValueError: If activity arrays have mismatched shapes
        Warning: If timeline-derived FPS differs from metadata FPS
    """
    # Build paths for each file type
    if path_config is None:
        path_config = {}

    # Helper to get path for a specific file type
    def get_path(file_type: str, filename: str) -> str:
        base_dir = path_config.get(file_type, root)
        return os.path.join(base_dir, filename)

    # Check if new format exists (check primary file)
    data_path = get_path('activity_data', f"{expname}_data.npz")
    if not os.path.exists(data_path):
        return None  # Not new format, try legacy

    print(f"[New Format] Loading {expname}...")
    log_lines = []

    # 1. Load activity data using helper
    arrays, n_neurons, n_frames = _load_activity_data_new_format(expname, get_path, log_lines)

    # 2. Load activity timeline using helper
    activity_timeline = _load_activity_timeline_new_format(expname, get_path, n_frames, log_lines)

    # 3. Load behavior data using helper
    behavior_df, behavior_timeline = _load_behavior_data_new_format(expname, get_path, log_lines)

    # 4. Check timeline consistency
    activity_timeline_present = activity_timeline is not None
    behavior_timeline_present = behavior_timeline is not None

    if activity_timeline_present != behavior_timeline_present:
        # Exactly one timeline is missing - this is ambiguous and error-prone
        missing = "behavior" if activity_timeline_present else "activity"
        present = "activity" if activity_timeline_present else "behavior"
        raise ValueError(
            f"Timeline mismatch: {present} timeline present but {missing} timeline missing. "
            f"Either provide both timelines for temporal interpolation, or provide neither "
            f"(which assumes data is pre-aligned with identical lengths)."
        )

    if not activity_timeline_present and not behavior_timeline_present:
        # Both timelines missing - assume data is pre-aligned
        log_lines.append(
            "NOTICE: Both timelines missing. Assuming data is pre-aligned. "
            "Will use 'simple' mode (requires exact length match)."
        )
        print(
            "NOTICE: Both timelines missing. Assuming data is pre-aligned. "
            "Will use 'simple' mode (requires exact length match)."
        )

    # 5. Load metadata using helper
    metadata = _load_metadata_new_format(expname, get_path, log_lines)

    # 6. Validate FPS using helper
    _validate_fps_new_format(activity_timeline, metadata, log_lines)

    # 7. Convert to legacy format using helper
    extracted_info = _convert_to_legacy_format(
        arrays, n_neurons, activity_timeline, metadata.get('fps'),
        behavior_df, behavior_timeline
    )

    return extracted_info, '\n'.join(log_lines), metadata



def read_all_data(expname: str, root: str = '/content', data_pieces: Optional[List[str]] = None,
                  prefer_new_format: bool = True, path_config: Optional[Dict[str, str]] = None) -> Tuple[List[str], Dict, List[str], Dict]:
    """
    Load and analyze all data pieces for an experiment.

    Master function that:
    1. Tries new format first if prefer_new_format=True (auto-detects)
    2. Falls back to legacy format (nested CSV directories)
    3. Performs automated analysis on each file
    4. Captures all console output for logging

    Args:
        expname: Experiment name (subdirectory under root)
        root: Root directory path (default: '/content')
        data_pieces: List of data pieces to load (default: DEFAULT_DATA_PIECES)
        prefer_new_format: If True, try new format first (default: True)
        path_config: Optional dict specifying custom paths for each file type (new format only):
            {
                'activity_data': '/path/to/imaging',
                'behavior_features': '/path/to/behavior',
                'activity_timeline': '/path/to/timestamps',
                'behavior_timeline': '/path/to/timestamps',
                'metadata': '/path/to/metadata'
            }

    Returns:
        tuple: (active_data_pieces, extracted_info, load_log, metadata) where:
            - active_data_pieces: list of data pieces found for this experiment
            - extracted_info: dict mapping data piece names to list of
              (data, names, timeline, fps) tuples (one per file)
            - load_log: list of captured console output lines from loading process
            - metadata: dict of metadata (empty dict for legacy format)
    """
    # Try new format first if preferred
    if prefer_new_format:
        result = read_new_format(expname, root, path_config=path_config)
        if result is not None:
            extracted_info, load_log, metadata = result
            active_data_pieces = list(extracted_info.keys())
            print(f"[New Format] Successfully loaded: {', '.join(active_data_pieces)}")
            return active_data_pieces, extracted_info, [load_log], metadata

    # Fall back to legacy format
    print(f"[Legacy Format] Loading {expname}...")

    if data_pieces is None:
        data_pieces = DEFAULT_DATA_PIECES

    extracted_info = {dp: [] for dp in data_pieces}
    active_data_pieces = [dp for dp in data_pieces if dp in os.listdir(join(root, expname))]

    with LogCapture() as load_log:
        for dp in data_pieces:
            if dp in active_data_pieces:
                dfs = load_data_part(dp, expname, root=root)
                for dfname, df in dfs.items():
                    print(f'Processing {dfname}...')
                    data, names, timeline, fps = auto_analysis(df, dp)
                    extracted_info[dp].append((data, names, timeline, fps))
            else:
                print(f'{dp}: no data')

            print('--------------------------------------------------------------------')
            print('--------------------------------------------------------------------')

    return active_data_pieces, extracted_info, load_log.getvalue().split('\n'), {}  # Empty metadata for legacy
