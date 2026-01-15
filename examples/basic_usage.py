"""
Basic usage example for IABS Data Synchronizer.

This example demonstrates the simplest way to synchronize a single experiment.
"""

from iabs_synchronizer import Synchronizer

# Initialize synchronizer with root data path
sync = Synchronizer(root_path='/data/neuroscience')

# Synchronize single experiment
# This will:
# 1. Load all data pieces (Calcium, Behavior, Coordinates, etc.)
# 2. Filter data based on quality thresholds
# 3. Align all data to calcium timeline
# 4. Return SyncResult object
result = sync.synchronize_experiment('RT_A04_S01')

# Save aligned data to .npz file
result.save('RT_A04_S01_aligned.npz')

print(f"Successfully synchronized {len(result.aligned_data)} features")
print(f"Timepoints: {result.get_timepoints()}")

# Access specific aligned arrays
calcium = result.aligned_data['calcium']  # Shape: (neurons, timepoints)
speed = result.aligned_data['speed']      # Shape: (timepoints,)

print(f"Calcium shape: {calcium.shape}")
print(f"Speed shape: {speed.shape}")

# View processing logs
print("\n=== Processing Logs ===")
print(result.get_full_log())
