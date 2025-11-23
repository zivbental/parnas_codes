# Multiplex V1 Trial Analysis Package

A well-organized, production-ready Python package for behavioral analysis of fly behavior trial data. This package provides the same functionality as the original MATLAB code but with modern Python best practices, proper package structure, and comprehensive error handling.

## Installation

```bash
# From project root
pip install -r requirements.txt

# Or install as package (for development)
pip install -e .
```

## Quick Start

### Command Line Interface

**Interactive mode** (opens file dialog):
```bash
python -m multiplex_v1_trial_analysis
```

**Process specific file**:
```bash
python -m multiplex_v1_trial_analysis input.txt
```

**Process without displaying plots**:
```bash
python -m multiplex_v1_trial_analysis input.txt --no-plots
```

**Process without saving .mat files**:
```bash
python -m multiplex_v1_trial_analysis input.txt --no-save
```

**Set logging level and save to file**:
```bash
python -m multiplex_v1_trial_analysis input.txt --log-level DEBUG --log-file analysis.log
```

### Programmatic Usage

```python
from multiplex_v1_trial_analysis import Config, FileReader, Pipeline

# Setup configuration
config = Config()

# Read file
reader = FileReader(config)
trial_data = reader.read('input.txt')

# Process
pipeline = Pipeline(config)
metrics_data = pipeline.process(trial_data, 'input.txt', show_plots=True, save_results=True)
```

## Package Structure

```
multiplex_v1_trial_analysis/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point
├── config.py                # Configuration management
├── logger.py                # Logging setup
├── exceptions.py            # Custom exceptions
│
├── io/                       # Input/Output operations
│   ├── file_reader.py       # File reading and parsing
│   ├── file_writer.py       # .mat file writing
│   └── file_dialog.py       # File selection UI
│
├── models/                   # Data structures
│   ├── trial_data.py        # TrialData class
│   └── metrics.py           # MetricsData class
│
├── processing/              # Core processing modules
│   ├── pipeline.py          # Main orchestrator
│   ├── smoothing.py         # Path smoothing
│   ├── epochs.py            # Epoch detection
│   ├── decisions.py         # Decision counting
│   ├── timing.py            # Time ratio calculations
│   └── trained_odor.py      # Trained odor identification
│
├── analysis/                 # Analysis and metrics
│   ├── metrics_calculator.py # Main metrics calculation
│   └── statistics.py        # Statistical summaries
│
├── visualization/            # Plotting modules
│   ├── traces.py            # Position trace plots
│   ├── summary.py           # Summary plots
│   └── plot_manager.py      # Plot orchestration
│
└── utils/                    # Utility functions
    ├── filters.py           # Savitzky-Golay filters
    ├── location.py          # Location classification
    ├── data_cleaning.py     # Data preprocessing
    └── matlab_compat.py    # MATLAB compatibility helpers
```

## Features

- **Well-organized package structure**: Clear separation of concerns
- **Type hints**: Full type annotations for better code clarity
- **Comprehensive error handling**: Custom exceptions and graceful error recovery
- **Logging**: Structured logging with configurable levels
- **Configuration management**: Centralized configuration with environment variable support
- **CLI interface**: Easy-to-use command-line interface
- **MATLAB compatibility**: Output .mat files are fully compatible with MATLAB

## Output

The pipeline generates:

1. **`.mat` files**: MATLAB-compatible output files
   - `filename.mat`: Contains `fb` and `fbm` structures
   - `filename-flies.mat`: Contains `decs_times` matrix

2. **Plots**: Two matplotlib figures
   - Position traces with annotations and digital outputs
   - Summary plots (4 subplots: changes scatter, time before/after, decisions before/after, mean speed)

## Data Format

The input file should be a tab-delimited text file with:
- First line: Header row with column names
- Subsequent lines: Data rows with 55 columns of float32 values

Expected column structure:
- Columns 1-13: Named data fields (Time, LEFTODOR1, etc.)
- Columns 14-33: EE (shock) data (20 columns, one per fly)
- Columns 34-35: Additional named fields
- Columns 36-55: cX position data (20 columns, one per fly)

## Configuration

Configuration can be set via:

1. **Default values** in `Config` class
2. **Environment variables**:
   - `MULTIPLEX_NFLIES`: Number of flies (default: 20)
   - `MULTIPLEX_LOG_LEVEL`: Logging level (default: INFO)
   - `MULTIPLEX_LOG_FILE`: Path to log file (enables file logging)

3. **Command-line arguments**:
   - `--no-plots`: Do not display plots
   - `--no-save`: Do not save .mat output files
   - `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
   - `--log-file`: Enable file logging

## Functional Equivalence

This package maintains 100% functional equivalence with the original MATLAB code:
- All algorithms are identical
- All calculations match MATLAB exactly
- Output .mat files are identical
- Plots are visually identical
- Edge cases (NaN, empty data, etc.) handled identically

## Requirements

- Python 3.7+
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

## Notes

- Index conversions from MATLAB's 1-based to Python's 0-based indexing are handled internally
- All MATLAB-specific behaviors (e.g., `fix()`, `textscan` whitespace handling) are replicated
- The package is designed for both interactive use and programmatic integration
