"""
Example: Using gap detection to validate timeline quality before alignment.
"""

import numpy as np
from iabs_synchronizer.utils.gap_detection import (
    detect_timeline_gaps,
    print_gap_report,
    check_timeline_quality,
    detect_gaps_in_dataset
)

print("=" * 60)
print("Gap Detection - Practical Example")
print("=" * 60)

# Scenario: Real neuroscience experiment
# - Calcium imaging: 10 minutes @ 30fps
# - Behavior tracking: Paused for 2 minutes at 5-minute mark

print("\n[Scenario] Mouse behavior tracking with experimenter intervention")
print("-" * 60)

# Simulate calcium timeline (continuous)
calcium_fps = 30
calcium_duration = 600  # 10 minutes
calcium_timeline = np.arange(0, calcium_duration, 1/calcium_fps)

# Simulate behavior timeline (with 2-minute gap)
behavior_fps = 40
part1 = np.arange(0, 300, 1/behavior_fps)  # 0-5 minutes
part2 = np.arange(420, 600, 1/behavior_fps)  # 7-10 minutes (120s gap)
behavior_timeline = np.concatenate([part1, part2])

print(f"Calcium: {len(calcium_timeline)} frames @ {calcium_fps}fps (continuous)")
print(f"Behavior: {len(behavior_timeline)} frames @ {behavior_fps}fps (with gap)")

# Check calcium timeline
print("\n1. Checking calcium timeline:")
has_gaps, ca_info = detect_timeline_gaps(calcium_timeline)
print(f"   Gaps detected: {has_gaps}")
print(f"   Estimated FPS: {ca_info['estimated_fps']:.2f}")

# Check behavior timeline
print("\n2. Checking behavior timeline:")
has_gaps, beh_info = detect_timeline_gaps(behavior_timeline)
if has_gaps:
    print(f"   [!] WARNING: {beh_info['n_gaps']} gap(s) detected!")
    print(f"   Gap location: {beh_info['gap_locations'][0][0]:.1f}s -> {beh_info['gap_locations'][0][1]:.1f}s")
    print(f"   Gap duration: {beh_info['gap_durations'][0]:.1f}s")
    print(f"   Data coverage: {beh_info['data_coverage']:.1f}%")

# Print detailed report
print_gap_report(behavior_timeline, beh_info, "Behavior Tracking")

# Example: Batch gap detection on dataset
print("\n\n[Example] Batch gap detection on full dataset")
print("-" * 60)

# Simulate filtered dataset
filtered_data = {
    'Calcium': {
        'Calcium': np.random.rand(10, len(calcium_timeline)),
        'timeline': calcium_timeline,
        'fps': calcium_fps
    },
    'Behavior_auto': [
        {
            'Speed': np.random.rand(len(behavior_timeline)),
            'timeline': behavior_timeline,
            'fps': behavior_fps
        },
        {
            'X': np.random.rand(len(behavior_timeline)),
            'timeline': behavior_timeline,
            'fps': behavior_fps
        }
    ]
}

# Detect gaps in all features
gap_summary = detect_gaps_in_dataset(filtered_data, warn_on_gaps=True)

# Print summary table
print("\n=== Gap Detection Summary ===")
print(f"{'Feature':<30} {'Status':<20} {'Coverage':<15}")
print("-" * 65)
for feature, info in gap_summary.items():
    status = "OK" if info['n_gaps'] == 0 else f"{info['n_gaps']} gap(s)"
    print(f"{feature:<30} {status:<20} {info['data_coverage']:>6.1f}%")

# Recommendations based on gap analysis
print("\n[Recommendations]")
print("-" * 60)

for feature, info in gap_summary.items():
    if info['n_gaps'] > 0:
        if info['data_coverage'] < 50:
            print(f"[X] {feature}: Low coverage ({info['data_coverage']:.1f}%) - consider excluding")
        elif info['data_coverage'] < 80:
            print(f"[!]  {feature}: Moderate gaps - review alignment results carefully")
        else:
            print(f"[OK]  {feature}: Acceptable gaps - can proceed with caution")
    else:
        print(f"[OK]  {feature}: No gaps detected - good quality")

# Example: Using with quality check
print("\n\n[Example] Quality check with error handling")
print("-" * 60)

try:
    # This will warn but not error
    print("Testing with warn=True:")
    gap_info = check_timeline_quality(behavior_timeline, "Behavior", warn=True)

    print("\nTesting with raise_on_gaps=True:")
    # This will raise error
    check_timeline_quality(behavior_timeline, "Behavior",
                          warn=False, raise_on_gaps=True)

except ValueError as e:
    print(f"  Error caught (expected): {str(e)[:80]}...")
    print("\n  Options to proceed:")
    print("    1. Trim gap regions before alignment")
    print("    2. Process recording segments separately")
    print("    3. Use force_mode='cast_to_ca' with understanding of gap limitations")

print("\n" + "=" * 60)
print("Gap detection helps ensure data quality before alignment!")
print("=" * 60)
