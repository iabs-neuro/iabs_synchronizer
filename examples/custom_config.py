"""
Custom configuration example for IABS Data Synchronizer.

This example demonstrates how to customize synchronization parameters
for specific experimental requirements.
"""

from iabs_synchronizer import Synchronizer, SyncConfig

# Example 1: Stricter alignment precision
print("Example 1: Stricter alignment precision")
strict_config = SyncConfig(
    align_precision=0.01,  # 1% tolerance instead of default 2%
    too_many_nans_thr=3    # 3% NaN threshold instead of default 5%
)

sync_strict = Synchronizer(root_path='/data/neuroscience', config=strict_config)
result = sync_strict.synchronize_experiment('RT_A04_S01')
print(f"Strict alignment: {len(result.aligned_data)} features synchronized\n")

# Example 2: Custom FPS values
print("Example 2: Custom FPS values")
custom_fps_config = SyncConfig(
    allowed_fps=[20, 30, 40],  # Only allow specific FPS values
)

sync_custom = Synchronizer(root_path='/data/neuroscience', config=custom_fps_config)
result = sync_custom.synchronize_experiment('RT_A04_S01')
print(f"Custom FPS: {len(result.aligned_data)} features synchronized\n")

# Example 3: Relaxed quality thresholds
print("Example 3: Relaxed quality thresholds (for noisy data)")
relaxed_config = SyncConfig(
    too_many_nans_thr=10,   # Allow up to 10% NaNs
    too_short_ts_thr=5,     # Allow data as short as 5% of calcium length
    align_precision=0.05    # Allow 5% length difference
)

sync_relaxed = Synchronizer(root_path='/data/neuroscience', config=relaxed_config)
result = sync_relaxed.synchronize_experiment('RT_A04_S01')
print(f"Relaxed thresholds: {len(result.aligned_data)} features synchronized\n")

# Example 4: Force specific alignment mode with custom config
print("Example 4: Forced mode with custom config")
result = sync_strict.synchronize_experiment(
    'RT_A04_S01',
    force_mode='cast_to_ca',  # Force this mode for all behavioral data
    rename_dict={
        'X': 'x_position',
        'Y': 'y_position',
        'Speed': 'locomotion_speed'
    },
    exclude_list=['x_green', 'y_green']  # Exclude unreliable features
)

print(f"Forced mode: {len(result.aligned_data)} features synchronized")
print(f"Features: {list(result.aligned_data.keys())}\n")

# Example 5: Accessing and validating results
print("Example 5: Result validation and inspection")
from iabs_synchronizer.core.postprocessing import validate_alignment, print_alignment_summary

# Validate alignment quality
validation = validate_alignment(result.aligned_data)
if validation['valid']:
    print("✓ Alignment validation passed")
    print(f"  Timepoints: {validation['timepoints']}")
else:
    print("✗ Alignment validation failed")
    for error in validation['errors']:
        print(f"  Error: {error}")

# Print detailed summary
print("\nDetailed summary:")
print_alignment_summary(result.aligned_data)
