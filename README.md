# IABS Data Synchronizer

A Python package for synchronizing multi-modal neuroscience recordings (calcium imaging, behavior tracking, coordinates) to a common temporal reference. Designed for maximum synchronization quality with support for various data acquisition scenarios.

## Features

### Core Capabilities
- **Intelligent Automatic Mode Selection** - 6-priority decision tree chooses best alignment
- **5 Alignment Modes** for different synchronization scenarios
- **Dual Format Support** - Modern (npz+csv+json) and Legacy (nested CSV) formats
- **Automatic Gap Detection** - Identifies recording interruptions with robust statistics
- **Robust FPS Calculation** - Gap-aware, median-based estimation
- **Metadata Preservation** - Full experiment metadata attached to outputs
- **Quality Filtering** with configurable thresholds
- **Multi-Array Support** - Calcium, spikes, reconstructions synchronized together

### Workflow Features
- **Batch Processing** support for multiple experiments
- **CLI and Library Interfaces** for different workflows
- **Optional Google Drive Integration**
- **Comprehensive Logging** for reproducibility
- **Temporal Validation** - Ensures proper alignment across all data streams

## What's New (2026)

### Version 1.0 Updates ✨

- **🆕 Modern Data Format**: Primary support for standardized npz+csv+json format
- **📊 Metadata Preservation**: Full experiment metadata attached to outputs (FPS, CNMF params, quality metrics)
- **🔍 Automatic Gap Detection**: Identifies recording interruptions with robust statistics (enabled by default)
- **📈 Robust FPS Calculation**: Gap-aware, median-based estimation eliminates duplication
- **🔄 Dual Format Support**: Auto-detects modern or legacy format, seamless fallback
- **🧬 Multi-Array Support**: Calcium, spikes, and reconstructions synchronized together
- **⚙️ Enhanced Configuration**: New gap detection and quality control parameters
- **🪟 Windows Ticks Support**: Automatic conversion of Windows timestamp format
- **📁 Scattered Files Support**: `path_config` parameter for files in different directories
- **🔎 Auto-Discovery**: Automatically find and synchronize all experiments in a directory

All changes are **backward compatible** - existing workflows continue to work!

## Installation

### Basic Installation

```bash
pip install iabs-synchronizer
```

### With Google Drive Support

```bash
pip install iabs-synchronizer[gdrive]
```

### Development Installation

```bash
git clone https://github.com/iabs-neuro/iabs_synchronizer.git
cd iabs_synchronizer
pip install -e ".[dev]"
```

## Data Formats

The synchronizer supports two data formats with automatic detection:

### Modern Format (Primary) ⭐

Standardized format with metadata preservation:

```
data/
├── LNOF_J01_1D_data.npz           # Activity arrays (C, spikes, reconstructions)
├── LNOF_J01_1D_Features.csv       # Behavioral features (x, y, speed, etc.)
├── LNOF_J01_1D_Mini_TS.csv        # Activity timeline
├── LNOF_J01_1D_VT_TS.csv          # Behavior timeline
└── LNOF_J01_1D_metadata.json      # Full metadata (FPS, CNMF params, etc.)
```

**Benefits**:
- ✅ Metadata preserved (FPS, processing parameters, quality metrics)
- ✅ Multiple activity arrays synchronized (C, spikes, reconstructions)
- ✅ Efficient storage (npz format)
- ✅ Automatic Windows ticks conversion

### Legacy Format (Backward Compatible)

Traditional nested directory structure:

```
data/
└── experiment_name/
    ├── Calcium/
    │   └── traces.csv
    ├── Behavior_auto/
    │   └── features.csv
    ├── Spikes/           # Optional
    │   └── spikes.csv
    └── Coordinates/      # Optional
        └── tracking.csv
```

**Auto-detection**: The loader tries modern format first, then falls back to legacy automatically.

## Quick Start

### Python Library

```python
from iabs_synchronizer import Synchronizer

# Initialize synchronizer (auto-detects format)
sync = Synchronizer(root_path='/data/neuroscience')

# By default, loads: 'Calcium', 'Spikes', 'Reconstructions', 'Behavior_auto'
# To customize: Synchronizer(root_path='/data', data_pieces=['Calcium', 'Behavior_auto'])

# Synchronize single experiment
result = sync.synchronize_experiment('LNOF_J01_1D')

# Save aligned data with metadata
result.save('LNOF_J01_1D_aligned.npz')

# Access aligned arrays
calcium = result.aligned_data['Calcium']           # (neurons, timepoints)
reconstructions = result.aligned_data['Reconstructions']  # (neurons, timepoints)
speed = result.aligned_data['speed']                # (timepoints,)

# Access metadata (modern format only)
print(f"FPS: {result.source_metadata.get('fps')}")
print(f"Session: {result.source_metadata.get('session_name')}")

# Check synchronization info (includes version tracking)
print(f"Synchronizer version: {result.sync_info['synchronizer_version']}")  # e.g., '1.0.0'
print(f"Source format: {result.sync_info['source_format']}")  # 'new' or 'legacy'
print(f"Timepoints: {result.sync_info['n_timepoints']}")
```

### Loading Data with Metadata

```python
from iabs_synchronizer.core.postprocessing import load_aligned_data

# Load synchronized data with metadata
aligned_data, metadata, sync_info = load_aligned_data(
    'LNOF_J01_1D_aligned.npz',
    load_metadata=True
)

# Access metadata
print(f"Original FPS: {metadata['fps']}")
print(f"CNMF parameters: {metadata['cnmf_params']}")

# Access sync info (includes version for reproducibility)
print(f"Synchronized with version: {sync_info['synchronizer_version']}")
print(f"Alignment mode used: {sync_info['alignment_mode']}")
print(f"Sync timestamp: {sync_info['sync_timestamp']}")
```

### Command Line Interface

```bash
# Synchronize single experiment
iabs-sync sync RT_A04_S01 --root /data/neuroscience --output result.npz

# Batch processing
iabs-sync batch RT_A04_S01 RT_A04_S02 RT_A04_S03 \
    --root /data --output-dir /data/aligned

# List available experiments
iabs-sync list --root /data/neuroscience

# Validate experiment data
iabs-sync validate RT_A04_S01 --root /data
```

## Pipeline Architecture

The synchronizer processes data through **4 sequential stages**:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  READ    │ →  │  FILTER  │ →  │  ALIGN   │ →  │ POSTPROCESS  │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
```

### Stage 1: READ
- Loads data from disk (auto-detects modern or legacy format)
- Extracts calcium, behavioral features, timelines
- Converts Windows ticks to seconds if needed

### Stage 2: FILTER
- Removes NaN values
- Validates data quality (length, coverage, sampling rate)
- Detects recording gaps (optional)
- Excludes low-quality features

### Stage 3: ALIGN
- **Automatic mode selection** (see below)
- Synchronizes all data to calcium's timeline
- Handles different sampling rates and temporal offsets

### Stage 4: POSTPROCESS
- Packages aligned data into SyncResult
- Attaches metadata and logs
- Validates output structure

---

## Automatic Alignment Mode Selection

**The ALIGN stage automatically chooses the best synchronization method** based on your data characteristics.

### The 6-Priority Decision Tree

| Priority | Condition | Mode Selected | Why |
|----------|-----------|---------------|-----|
| **1** | Force mode specified | User's choice | User override |
| **2** | Both timelines present | `'2 timelines'` | **Best accuracy** - uses timestamps |
| **3** | Exact length match | `'simple'` | Already aligned |
| **4** | Close length (±10 samples) | `'crop'` | Trim excess |
| **5** | Both timelines missing | `'cast_to_ca'` | Interpolate with warning ⚠️ |
| **6** | One timeline missing | **ERROR** | Suspicious - fix data |

### Valid Timeline Configurations

| Source Timeline | Target Timeline | Different Lengths | Behavior |
|-----------------|-----------------|-------------------|----------|
| ✅ Present | ✅ Present | Any | ✅ Uses `'2 timelines'` (most accurate) |
| ❌ Missing | ❌ Missing | No (exact match) | ✅ Uses `'simple'` (pass-through) |
| ❌ Missing | ❌ Missing | Yes (close) | ✅ Uses `'crop'` (trim excess) |
| ❌ Missing | ❌ Missing | Yes (different) | ⚠️ Uses `'cast_to_ca'` (assumes same time span) |
| ❌ Missing | ✅ Present | Any | ❌ **ERROR** - provide both or neither |
| ✅ Present | ❌ Missing | Any | ❌ **ERROR** - provide both or neither |

### When You See Errors

**"Only [source/target] timeline present"** means you have incomplete data:

**Why single missing timeline is an error:** It's suspicious and likely indicates:
- Lost or corrupted file
- Incomplete data transfer
- Configuration mistake

---

## The 5 Alignment Modes

Each mode is a specialized algorithm for different synchronization scenarios. **The system automatically selects the best mode** (see above), but you can force a specific mode if needed.

### 1. `'2 timelines'` - Temporal Overlap Alignment
**Automatically selected when:** Both calcium and behavior have timestamps

- **Algorithm:**
  1. Find temporal overlap (time range present in both)
  2. Trim both datasets to overlap region
  3. Interpolate behavior onto calcium's timeline
- **Handles:** Different sampling rates, temporal offsets, gaps
- **Most accurate** - uses actual timestamps

### 2. `'simple'` - Pass-Through
**Automatically selected when:** Lengths match exactly

- **Algorithm:** Returns data unchanged
- **Validates:** Lengths must match or raises error
- **Use case:** Pre-aligned data

### 3. `'cast_to_ca'` - Synthetic Timeline Interpolation ⭐
**Automatically selected when:** Both timelines missing AND lengths differ

- **Algorithm:**
  1. Create synthetic timeline for behavior matching calcium's time range
  2. Interpolate behavior from N samples → M samples
- **Assumption:** ⚠️ Both datasets span **exactly the same time period**
- **Warning printed:** "Verify recordings started/ended simultaneously!"
- **Most common fallback** for data without timestamps

### 4. `'crop'` - Trim Excess
**Automatically selected when:** Lengths approximately equal (within 10 samples)

- **Algorithm:** Trims excess samples from end
- **Use case:** Recordings stopped at slightly different times
- **No interpolation** - preserves data quality

### 5. `'factor'` - FPS-Based Scaling
**Not implemented yet** - raises `NotImplementedError`

- Reserved for future FPS ratio-based alignment
- Use `force_mode='cast_to_ca'` instead

### Forcing a Specific Mode

```python
# Override automatic selection
result = sync.synchronize_experiment(
    'RT_A04_S01',
    force_mode='cast_to_ca'  # Use this mode regardless of data
)
```

## Gap Detection

Automatic detection of recording interruptions with robust statistics:

### Features

- **Automatic Detection**: Runs during filtering stage by default
- **Robust Statistics**: Uses median/percentile-based estimation to ignore noise
- **Gap Metrics**: Reports number of gaps, locations, duration, and data coverage
- **Configurable Thresholds**: Adjust sensitivity via `gap_threshold_multiplier`
- **Optional Exclusion**: Can automatically exclude low-coverage features

### Basic Usage

```python
from iabs_synchronizer import Synchronizer

# Gap detection runs automatically
sync = Synchronizer(root_path='/data')
result = sync.synchronize_experiment('experiment_001')

# Output:
# [Gap Detection] Checking timeline quality...
#   WARNING: Calcium has 2 gap(s), coverage: 87.3%
#   WARNING: x has 1 gap(s), coverage: 95.2%
# [Gap Detection] Found gaps in 2 feature(s)
```

### Configuration Options

```python
from iabs_synchronizer import Synchronizer, SyncConfig

# Disable gap detection
config = SyncConfig(detect_gaps=False)
sync = Synchronizer(root_path='/data', config=config)

# Enable with custom threshold (more sensitive)
config = SyncConfig(
    detect_gaps=True,
    warn_on_gaps=True,
    gap_threshold_multiplier=2.0  # Gap = 2x typical interval (default: 3.0)
)

# Auto-exclude features with low coverage
config = SyncConfig(
    detect_gaps=True,
    exclude_low_coverage=True,
    min_coverage_threshold=90.0  # Require 90% coverage (default: 50%)
)
```

### Manual Gap Detection

```python
from iabs_synchronizer.utils.gap_detection import detect_timeline_gaps, print_gap_report

# Check a specific timeline
timeline = np.array([0, 0.033, 0.066, ..., 10.0])  # Your timeline
has_gaps, gap_info = detect_timeline_gaps(timeline)

if has_gaps:
    print_gap_report(gap_info, feature_name='my_data')
    # Output:
    # === Gap Detection Report: my_data ===
    # Gaps detected: 2
    # Total gap duration: 5.2 seconds
    # Data coverage: 87.3%
    # Estimated FPS: 30.0
    # ... (detailed gap locations)
```

## Advanced Usage

### Custom Configuration

```python
from iabs_synchronizer import Synchronizer, SyncConfig

# Create custom config with stricter alignment
config = SyncConfig(
    allowed_fps=[20, 30, 40],
    align_precision=0.01,  # 1% tolerance instead of 2%
    too_many_nans_thr=3    # 3% NaN threshold instead of 5%
)

sync = Synchronizer(root_path='/data', config=config)
```

### Feature Renaming and Exclusion

```python
# Option 1: Rename features directly in code
rename_dict = {
    'X': 'x_position',
    'Y': 'y_position',
    'Speed': 'locomotion_speed'
}

result = sync.synchronize_experiment(
    'RT_A04_S01',
    rename_dict=rename_dict,
    exclude_list=['x_green', 'y_green']
)
```

**Option 2: Load rename mapping from JSON file**

Create `rename_mapping.json`:
```json
{
    "X": "x_position",
    "Y": "y_position",
    "Speed": "locomotion_speed"
}
```

Then load and use it:
```python
import json

with open('rename_mapping.json', 'r') as f:
    rename_dict = json.load(f)

result = sync.synchronize_experiment('RT_A04_S01', rename_dict=rename_dict)
```

**CLI usage with rename file:**
```bash
iabs-sync sync RT_A04_S01 --root /data --rename rename_mapping.json
```

### Working with Scattered Files (path_config)

If your data files are in different directories, use `path_config`:

```python
from iabs_synchronizer import Synchronizer

sync = Synchronizer(root_path='/data/neuroscience')

# Define path_config ONCE for all experiments
# (All experiment files follow the same scattered structure)
path_config = {
    'activity_data': '/mnt/imaging',        # All *_data.npz files here
    'behavior_features': '/mnt/behavior',   # All *_Features.csv files here
    'activity_timeline': '/mnt/timestamps', # All *_Mini_TS.csv files here
    'behavior_timeline': '/mnt/timestamps', # All *_VT_TS.csv files here
    'metadata': '/mnt/metadata'             # All *_metadata.json files here
}

# Single experiment
result = sync.synchronize_experiment(
    'RT_A04_S01',
    path_config=path_config
)

# Batch processing - path_config applies to ALL experiments
experiments = ['RT_A04_S01', 'RT_A04_S02', 'RT_A04_S03', 'LNOF_J01_1D']
results = sync.synchronize_batch(
    experiments,
    output_dir='/data/synchronized',
    path_config=path_config  # Applied to all experiments
)

# Auto-discover and sync ALL experiments with scattered files
results = sync.synchronize_all(
    output_dir='/data/synchronized',
    path_config=path_config,  # Applied to all experiments
    only_complete=True
)

print(f"Successfully processed {len(results)} experiments")
```

**Key points:**
- Define `path_config` once and reuse it for all experiments
- Works with `synchronize_experiment()`, `synchronize_batch()`, and `synchronize_all()`
- Only works with modern format (npz+csv+json). Legacy format requires all files under a single experiment directory.

### Batch Processing with Progress Bar

```python
experiments = ['RT_A04_S01', 'RT_A04_S02', 'RT_A04_S03']

results = sync.synchronize_batch(
    experiments,
    output_dir='/data/aligned',
    force_mode='cast_to_ca'
)

print(f"Successfully processed {len(results)} experiments")
```

### Auto-Discovery and Batch Synchronization

The synchronizer can automatically find all experiments in a directory and process them:

```python
from iabs_synchronizer import Synchronizer

sync = Synchronizer(root_path='/data/neuroscience')

# Discover all experiments (both modern and legacy formats)
discovered = sync.discover_experiments(format='auto')

# Shows what was found:
# discovered = {
#     'LNOF_J01_1D': {'format': 'new', 'complete': True, 'has_timelines': True, ...},
#     'RT_A04_S01': {'format': 'new', 'complete': True, 'has_timelines': True, ...},
#     'OLD_EXP_01': {'format': 'legacy', 'complete': True, ...}
# }

for name, info in discovered.items():
    print(f"{name}: {info['format']}, complete={info['complete']}")

# Synchronize ALL discovered experiments automatically
results = sync.synchronize_all(
    output_dir='/data/synchronized',
    only_complete=True,  # Skip incomplete experiments
    force_mode='cast_to_ca'
)

print(f"Successfully synchronized {len(results)} experiments")
```

**Discovery options:**
- `format='auto'` - Find both modern and legacy (default)
- `format='new'` - Only find modern format experiments
- `format='legacy'` - Only find legacy format experiments

**Synchronize all options:**
- `only_complete=True` - Skip experiments missing required files (default)
- `only_complete=False` - Try to process all experiments (may fail)

### Accessing Logs

```python
result = sync.synchronize_experiment('RT_A04_S01')

# Print full processing log
print(result.get_full_log())

# Access logs by phase
print("Read log:", result.read_log)
print("Filter log:", result.filter_log)
print("Align log:", result.align_log)
```

## Data Structure

### Input Formats

**Modern Format** (Recommended):
```
data/
├── experiment_data.npz           # Activity: C, spikes, reconstructions
├── experiment_Features.csv       # Behavior: x, y, speed, states, etc.
├── experiment_Mini_TS.csv        # Activity timeline
├── experiment_VT_TS.csv          # Behavior timeline
└── experiment_metadata.json      # Full metadata
```

**Legacy Format** (Backward Compatible):
```
root_path/
├── experiment_name/
│   ├── Calcium/
│   │   └── traces.csv
│   ├── Behavior_auto/
│   │   └── features.csv
│   ├── Spikes/              # Optional
│   │   └── spikes.csv
│   └── Coordinates/         # Optional
│       └── tracking.csv
```

### Output .npz File

```python
import numpy as np

# Load synchronized data
data = np.load('experiment_aligned.npz', allow_pickle=True)

# Activity arrays (all have same timepoint dimension)
print(data['Calcium'].shape)           # (251, 17855) - 251 neurons × 17855 timepoints
print(data['Reconstructions'].shape)   # (251, 17855) - if available
print(data['Spikes'].shape)            # (251, 17855) - if available

# Behavioral features
print(data['speed'].shape)      # (17855,)
print(data['x'].shape)          # (17855,)
print(data['y'].shape)          # (17855,)

# Metadata (modern format only)
metadata = data['_metadata'].item()
sync_info = data['_sync_info'].item()

print(f"Original FPS: {metadata.get('fps')}")
print(f"Session: {metadata.get('session_name')}")

# Sync info includes version for reproducibility
print(f"Synchronized with version: {sync_info.get('synchronizer_version')}")
print(f"Alignment mode: {sync_info.get('alignment_mode')}")
print(f"Timestamp: {sync_info.get('sync_timestamp')}")
```

## Configuration Parameters

### Quality Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `allowed_fps` | [10,20,30,40,50] | Accepted FPS values for detection |
| `too_many_nans_thr` | 5% | Reject data with >5% NaN values |
| `too_short_ts_thr` | 10% | Reject data <10% of calcium length |
| `max_unique_vals` | 0.02 | Flag if >2% unique timestamp diffs |
| `align_precision` | 0.02 | Max percentage difference for 'crop' mode (2% of length) |

### Gap Detection (New)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_gaps` | True | Enable automatic gap detection |
| `warn_on_gaps` | True | Print warnings when gaps detected |
| `gap_threshold_multiplier` | 3.0 | Gap threshold as multiple of typical interval |
| `exclude_low_coverage` | False | Auto-exclude features with low coverage |
| `min_coverage_threshold` | 50.0 | Minimum data coverage % to keep feature |

## Backward Compatibility

Old notebook code works with minimal changes:

```python
# Old notebook style (still works)
from iabs_synchronizer import read_all_data, filter_data, align_all_data

extracted_info = read_all_data(expname, root=ROOT)
filtered_info = filter_data(active_data_pieces, extracted_info)
aligned_data = align_all_data(filtered_info, force_pathway='cast_to_ca')

# New recommended style
from iabs_synchronizer import Synchronizer

sync = Synchronizer(root_path=ROOT)
result = sync.synchronize_experiment(expname)
aligned_data = result.aligned_data
```

## CLI Reference

### Sync Command

```bash
iabs-sync sync EXPERIMENT --root PATH [OPTIONS]

Options:
  --output, -o PATH         Output .npz file path
  --mode MODE               Force alignment mode
  --rename FILE             JSON file with rename mapping
  --exclude FEATURES        Features to exclude
  --verbose, -v             Print detailed logs
```

### Batch Command

```bash
iabs-sync batch EXPERIMENT [EXPERIMENT ...] --root PATH [OPTIONS]

Options:
  --output-dir, -o DIR      Output directory
  --mode MODE               Force alignment mode (applied to all)
  --rename FILE             JSON rename mapping (applied to all)
  --exclude FEATURES        Features to exclude (applied to all)
  --verbose, -v             Print detailed logs
```

### Validate Command

```bash
iabs-sync validate EXPERIMENT --root PATH [--verbose]
```

### List Command

```bash
iabs-sync list --root PATH
```

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0
- openpyxl ≥ 3.0.0 (for Excel support)

### Optional Dependencies

- driada ≥ 1.0.0 (for Google Drive integration)
- tqdm ≥ 4.60.0 (for progress bars)

## Project Structure

```
iabs-synchronizer/
├── iabs_synchronizer/
│   ├── core/               # Core processing modules (4-stage pipeline)
│   │   ├── io.py          # READ: Data loading (modern + legacy formats)
│   │   ├── filtering.py   # FILTER: Quality filtering + gap detection
│   │   ├── alignment.py   # ALIGN: Synchronization (5 modes, auto-selection)
│   │   └── postprocessing.py  # POSTPROCESS: Metadata, validation, saving
│   ├── models/            # Data structures (SyncResult, DataPiece)
│   ├── pipeline/          # Main orchestration
│   │   ├── synchronizer.py    # Synchronizer class
│   │   ├── discovery.py       # Experiment discovery (new & legacy)
│   │   └── validation.py      # Experiment validation
│   ├── utils/             # Utilities
│   │   ├── logging.py     # Log capture
│   │   └── gap_detection.py  # Gap detection (robust statistics)
│   ├── gdrive/            # Optional GDrive support
│   └── cli/               # Command-line interface
├── tests/                 # Comprehensive test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── edge_cases/        # Edge case tests
├── examples/              # Usage examples
└── pyproject.toml         # Package configuration
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:

```
@software{iabs_synchronizer,
  title={IABS Data Synchronizer},
  author={IABS team},
  year={2026},
  url={https://github.com/iabs-neuro/iabs_synchronizer}
}
```

## Support

- **Issues:** https://github.com/iabs-neuro/iabs_synchronizer/issues
- **Documentation:** https://github.com/iabs-neuro/iabs_synchronizer#readme

## Acknowledgments

Developed by Institute for Advanced Brain Studies team.
Special thanks to all contributors and users who provided feedback.
