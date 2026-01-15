"""
Test missing timelines handling in new format.

Tests that the system handles missing timeline files with strict validation:
- Missing activity timeline (Mini_TS.csv) → ERROR
- Missing behavior timeline (VT_TS.csv) → ERROR
- Both timelines missing → 'simple' mode (requires exact length match)

Expected behavior:
- Exactly one timeline missing: Raises ValueError
- Both timelines missing + lengths match: 'simple' mode (pass-through)
- Both timelines missing + lengths differ: Raises ValueError
- force_mode can override to allow cropping

Requires test data in data/new/ directory. Tests will be skipped gracefully
if test data is not available.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from iabs_synchronizer.pipeline.synchronizer import Synchronizer


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with new format files."""
    import gc
    import time

    # Use actual data from data/new/
    source_dir = Path('data/new')
    if not source_dir.exists():
        pytest.skip("Test data not available: data/new/ directory not found")

    # Check for required files
    required_files = [
        'LNOF_J01_1D_data.npz',
        'LNOF_J01_1D_Features.csv',
        'LNOF_J01_1D_Mini_TS.csv',
        'LNOF_J01_1D_VT_TS.csv',
        'LNOF_J01_1D_metadata.json'
    ]

    missing_files = [f for f in required_files if not (source_dir / f).exists()]
    if missing_files:
        pytest.skip(
            f"Test data not available: missing files in data/new/: {', '.join(missing_files)}"
        )

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Copy all files
        files_copied = []
        for file in source_dir.glob('LNOF_J01_1D_*'):
            shutil.copy(file, temp_dir)
            files_copied.append(file.name)

        if not files_copied:
            pytest.skip("Test data not available: no LNOF_J01_1D_* files found in data/new/")

        # Verify essential files were copied
        essential_files = ['LNOF_J01_1D_data.npz', 'LNOF_J01_1D_Features.csv']
        for essential_file in essential_files:
            if not (temp_dir / essential_file).exists():
                pytest.skip(f"Test data not available: {essential_file} not found after copy")

    except Exception as e:
        # Cleanup on error
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        pytest.skip(f"Test data setup failed: {e}")

    yield temp_dir

    # Cleanup (with retry for Windows file handles)
    gc.collect()  # Force garbage collection to close file handles
    time.sleep(0.1)  # Small delay for Windows
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # Windows sometimes holds file handles - try again after delay
        time.sleep(0.5)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass  # Give up if still fails


def test_missing_activity_timeline(temp_data_dir):
    """Test with missing Mini_TS.csv (activity timeline) - should raise error."""
    # Remove activity timeline
    mini_ts = temp_data_dir / 'LNOF_J01_1D_Mini_TS.csv'
    mini_ts.unlink()

    # Should raise ValueError because only one timeline is missing
    sync = Synchronizer(root_path=str(temp_data_dir))

    with pytest.raises(ValueError) as exc_info:
        result = sync.synchronize_experiment('LNOF_J01_1D')

    # Check error message is informative
    error_msg = str(exc_info.value)
    assert 'Timeline mismatch' in error_msg
    assert 'activity timeline missing' in error_msg
    assert 'behavior timeline present' in error_msg
    assert 'provide both' in error_msg or 'provide neither' in error_msg


def test_missing_behavior_timeline(temp_data_dir):
    """Test with missing VT_TS.csv (behavior timeline) - should raise error."""
    # Remove behavior timeline
    vt_ts = temp_data_dir / 'LNOF_J01_1D_VT_TS.csv'
    vt_ts.unlink()

    # Should raise ValueError because only one timeline is missing
    sync = Synchronizer(root_path=str(temp_data_dir))

    with pytest.raises(ValueError) as exc_info:
        result = sync.synchronize_experiment('LNOF_J01_1D')

    # Check error message is informative
    error_msg = str(exc_info.value)
    assert 'Timeline mismatch' in error_msg
    assert 'behavior timeline missing' in error_msg
    assert 'activity timeline present' in error_msg
    assert 'provide both' in error_msg or 'provide neither' in error_msg


def test_both_timelines_missing(temp_data_dir):
    """Test with both timelines missing - should use cast_to_ca fallback."""
    # Remove both timelines
    mini_ts = temp_data_dir / 'LNOF_J01_1D_Mini_TS.csv'
    vt_ts = temp_data_dir / 'LNOF_J01_1D_VT_TS.csv'
    mini_ts.unlink()
    vt_ts.unlink()

    # With both timelines missing and length mismatch (107107 vs 108088),
    # should now automatically use 'cast_to_ca' fallback
    sync = Synchronizer(root_path=str(temp_data_dir))

    # Should succeed with cast_to_ca fallback
    result = sync.synchronize_experiment('LNOF_J01_1D')

    # Verify result structure
    assert result is not None
    assert 'Calcium' in result.aligned_data
    assert 'speed' in result.aligned_data  # Behavior feature aligned

    # All should be aligned to calcium length
    ca_length = result.aligned_data['Calcium'].shape[1]
    assert result.aligned_data['speed'].shape[0] == ca_length


def test_both_timelines_missing_with_force_mode(temp_data_dir):
    """Test that force_mode='crop' works when both timelines missing."""
    # Remove both timelines
    mini_ts = temp_data_dir / 'LNOF_J01_1D_Mini_TS.csv'
    vt_ts = temp_data_dir / 'LNOF_J01_1D_VT_TS.csv'
    mini_ts.unlink()
    vt_ts.unlink()

    # With force_mode='crop', should succeed even with length mismatch
    sync = Synchronizer(root_path=str(temp_data_dir))
    result = sync.synchronize_experiment('LNOF_J01_1D', force_mode='crop')

    # Verify successful synchronization
    assert len(result.aligned_data) > 0
    assert 'Calcium' in result.aligned_data

    # Check that notice was logged
    read_log = '\n'.join(result.read_log)
    assert 'Both timelines missing' in read_log or 'pre-aligned' in read_log

    # Verify all features aligned to same length
    lengths = {name: len(data) if data.ndim == 1 else data.shape[1]
               for name, data in result.aligned_data.items()}
    assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
