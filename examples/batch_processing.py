"""
Batch processing example for IABS Data Synchronizer.

This example demonstrates how to synchronize multiple experiments efficiently.
"""

from iabs_synchronizer import Synchronizer

# Initialize synchronizer
sync = Synchronizer(root_path='/data/neuroscience')

# Define experiments to process
experiments = [
    'RT_A04_S01',
    'RT_A04_S02',
    'RT_A04_S03',
    'RT_A05_S01',
    'RT_A05_S02'
]

# Option 1: Batch processing with automatic saving
print("Processing batch with automatic saving...")
results = sync.synchronize_batch(
    experiments,
    output_dir='/data/aligned',  # Automatically saves each result
    force_mode='cast_to_ca'      # Force specific alignment mode
)

print(f"Successfully processed: {len(results)}/{len(experiments)} experiments")

# Option 2: Manual batch processing with custom handling
print("\nProcessing batch with custom handling...")
for exp_name in experiments:
    try:
        # Synchronize experiment
        result = sync.synchronize_experiment(
            exp_name,
            force_mode='cast_to_ca'
        )

        # Custom validation
        n_features = len(result.aligned_data)
        n_timepoints = result.get_timepoints()

        print(f"{exp_name}: {n_features} features, {n_timepoints} timepoints")

        # Save with custom path
        output_path = f'/data/aligned/{exp_name}_synchronized.npz'
        result.save(output_path)

    except Exception as e:
        print(f"Failed on {exp_name}: {e}")

# Option 3: Selective batch processing
print("\nSelective processing based on validation...")
all_experiments = sync.list_experiments()

valid_experiments = []
for exp in all_experiments:
    report = sync.validate_experiment(exp)
    if report['valid']:
        valid_experiments.append(exp)
    else:
        print(f"Skipping {exp}: {report['errors']}")

print(f"\nProcessing {len(valid_experiments)} valid experiments...")
results = sync.synchronize_batch(
    valid_experiments,
    output_dir='/data/aligned'
)
