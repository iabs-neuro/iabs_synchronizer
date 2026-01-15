"""
Post-processing operations for aligned neuroscience data.

This module handles operations after alignment, such as renaming features
for consistency and saving results to disk.

Source: Lines 751-850 from IABS_data_synchronizer_1_3.ipynb
"""

from typing import Dict, List, Optional
import numpy as np
from ..utils.logging import LogCapture


def rename_attributes(aligned_data: Dict[str, np.ndarray],
                     rename_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Rename features in aligned data dictionary and normalize to lowercase.

    This function:
    1. Applies user-specified renames (old_name → new_name)
    2. Converts all final keys to lowercase for consistency
    3. Warns if rename source keys are not found

    Args:
        aligned_data: Dictionary of aligned arrays {feature_name: array, ...}
        rename_dict: Mapping of old names to new names {old_name: new_name, ...}

    Returns:
        Dict[str, np.ndarray]: Dictionary with renamed and lowercased keys

    Example:
        >>> aligned = {'Speed': array([...]), 'X': array([...]), 'calcium': array([...])}
        >>> rename_dict = {'X': 'x_position', 'Speed': 'locomotion_speed'}
        >>> result = rename_attributes(aligned, rename_dict)
        >>> # Result: {'x_position': array([...]), 'locomotion_speed': array([...]), 'calcium': array([...])}
        >>> # Note: All keys are lowercased in final output
    """
    # Apply renames
    for old_name, new_name in rename_dict.items():
        if old_name in aligned_data:
            aligned_data[new_name] = aligned_data.pop(old_name)
        else:
            print(f'{old_name} not in aligned_data!')

    # Normalize all keys to lowercase for consistency
    final_aligned_data = {key.lower(): aligned_data[key] for key in aligned_data.keys()}

    return final_aligned_data


def save_aligned_data(aligned_data: Dict[str, np.ndarray],
                      output_path: str,
                      compressed: bool = False,
                      metadata: Optional[Dict] = None,
                      sync_info: Optional[Dict] = None) -> None:
    """
    Save aligned data to .npz file with optional metadata.

    Args:
        aligned_data: Dictionary of aligned arrays {feature_name: array, ...}
        output_path: Path to output file (should end with .npz)
        compressed: If True, use compressed format (smaller file, slower load)
        metadata: Optional source metadata dict (from new format loading)
        sync_info: Optional synchronization metadata (alignment info, warnings, etc.)

    Raises:
        ValueError: If no data to save
        IOError: If unable to write to output path

    Note:
        Metadata is saved with keys '_metadata' and '_sync_info' and requires
        allow_pickle=True when loading: np.load(file, allow_pickle=True)

    Example:
        >>> aligned_data = {'calcium': array(...), 'speed': array(...)}
        >>> metadata = {'fps': 30.0, 'session_name': 'exp_001'}
        >>> sync_info = {'alignment_mode': '2 timelines', 'n_frames': 5000}
        >>> save_aligned_data(aligned_data, 'experiment_001.npz', metadata=metadata, sync_info=sync_info)
    """
    if not aligned_data:
        raise ValueError("No aligned data to save")

    # Prepare save dict with data arrays
    save_dict = aligned_data.copy()

    # Add metadata if provided (use object dtype for pickle support)
    if metadata:
        save_dict['_metadata'] = np.array(metadata, dtype=object)

    if sync_info:
        save_dict['_sync_info'] = np.array(sync_info, dtype=object)

    # Save to file
    if compressed:
        np.savez_compressed(output_path, **save_dict)
    else:
        np.savez(output_path, **save_dict)

    # Print summary
    print(f"\nSaved to: {output_path}")
    print(f"  Data arrays: {len(aligned_data)} features")
    if metadata:
        print(f"  Metadata: {len(metadata)} fields attached")
    if sync_info:
        print(f"  Sync info: {len(sync_info)} fields attached")
    if metadata or sync_info:
        print(f"  Note: Use allow_pickle=True when loading metadata")


def load_aligned_data(input_path: str, load_metadata: bool = False):
    """
    Load aligned data from .npz file, optionally with metadata.

    Args:
        input_path: Path to input .npz file
        load_metadata: If True, return (aligned_data, metadata, sync_info) tuple

    Returns:
        If load_metadata=False:
            Dict[str, np.ndarray]: Dictionary of loaded arrays
        If load_metadata=True:
            Tuple[Dict, Dict, Dict]: (aligned_data, metadata, sync_info)
                - aligned_data: Dictionary of data arrays
                - metadata: Original source metadata (empty dict if none)
                - sync_info: Synchronization metadata (empty dict if none)

    Raises:
        FileNotFoundError: If input file doesn't exist
        IOError: If unable to read file

    Example:
        >>> # Load data only (backward compatible)
        >>> aligned_data = load_aligned_data('experiment_001.npz')
        >>> print(aligned_data.keys())
        dict_keys(['calcium', 'speed', 'x', 'y', ...])

        >>> # Load with metadata
        >>> data, metadata, sync_info = load_aligned_data('experiment_001.npz', load_metadata=True)
        >>> print(metadata['fps'])
        30.0
    """
    with np.load(input_path, allow_pickle=True) as data:
        # Separate data arrays from metadata
        aligned_data = {}
        metadata = {}
        sync_info = {}

        for key in data.keys():
            if key == '_metadata':
                metadata = data[key].item() if data[key].shape == () else data[key]
            elif key == '_sync_info':
                sync_info = data[key].item() if data[key].shape == () else data[key]
            else:
                aligned_data[key] = data[key]

    print(f"Loaded {len(aligned_data)} features from {input_path}")
    if metadata:
        print(f"  Metadata: {len(metadata)} fields")
    if sync_info:
        print(f"  Sync info: {len(sync_info)} fields")

    if load_metadata:
        return aligned_data, metadata, sync_info
    else:
        return aligned_data


def exclude_features(aligned_data: Dict[str, np.ndarray],
                    exclude_list: List[str]) -> Dict[str, np.ndarray]:
    """
    Remove specified features from aligned data.

    Args:
        aligned_data: Dictionary of aligned arrays
        exclude_list: List of feature names to remove

    Returns:
        Dict[str, np.ndarray]: Dictionary with specified features removed

    Example:
        >>> aligned = {'calcium': array(...), 'speed': array(...), 'x_green': array(...)}
        >>> result = exclude_features(aligned, ['x_green'])
        >>> print(result.keys())
        dict_keys(['calcium', 'speed'])
    """
    filtered_data = {k: v for k, v in aligned_data.items() if k not in exclude_list}

    excluded_count = len(aligned_data) - len(filtered_data)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} features: {', '.join(exclude_list)}")

    return filtered_data


def validate_alignment(aligned_data: Dict[str, np.ndarray]) -> Dict[str, any]:
    """
    Validate that all aligned arrays have consistent dimensions.

    Checks that all arrays are synchronized to the same timepoint dimension.
    For 1D arrays, checks length. For 2D arrays, checks second dimension (timepoints).

    Args:
        aligned_data: Dictionary of aligned arrays

    Returns:
        dict: Validation report with keys:
            - 'valid': bool indicating if all arrays are properly aligned
            - 'timepoints': common timepoint count (if valid)
            - 'lengths': dict mapping feature names to their lengths
            - 'errors': list of error messages (if invalid)

    Example:
        >>> aligned = {'calcium': np.zeros((100, 5000)), 'speed': np.zeros(5000)}
        >>> report = validate_alignment(aligned)
        >>> print(report)
        {'valid': True, 'timepoints': 5000, 'lengths': {'calcium': 5000, 'speed': 5000}, 'errors': []}
    """
    lengths = {}
    errors = []

    for name, arr in aligned_data.items():
        if arr.ndim == 1:
            lengths[name] = len(arr)
        elif arr.ndim == 2:
            lengths[name] = arr.shape[1]  # Assume (neurons, timepoints) format
        else:
            errors.append(f"Feature '{name}' has unexpected dimensionality: {arr.ndim}")

    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:
        valid = False
        errors.append(f"Inconsistent timepoint lengths found: {unique_lengths}")
        timepoints = None
    else:
        valid = True
        timepoints = unique_lengths.pop() if unique_lengths else None

    return {
        'valid': valid,
        'timepoints': timepoints,
        'lengths': lengths,
        'errors': errors
    }


def print_alignment_summary(aligned_data: Dict[str, np.ndarray]) -> None:
    """
    Print a summary of aligned data.

    Displays feature names, shapes, and data types for quick inspection.

    Args:
        aligned_data: Dictionary of aligned arrays

    Example:
        >>> print_alignment_summary(aligned_data)
        Alignment Summary:
        ==================
        Features: 15
        Timepoints: 17855

        Feature Details:
        - calcium: (251, 17855) float64
        - speed: (17855,) float64
        - x: (17855,) float64
        ...
    """
    validation = validate_alignment(aligned_data)

    print("Alignment Summary:")
    print("=" * 50)
    print(f"Features: {len(aligned_data)}")
    print(f"Timepoints: {validation['timepoints']}")
    print(f"Valid: {validation['valid']}")

    if validation['errors']:
        print("\nErrors:")
        for error in validation['errors']:
            print(f"  - {error}")

    print("\nFeature Details:")
    for name, arr in aligned_data.items():
        print(f"  - {name}: {arr.shape} {arr.dtype}")
