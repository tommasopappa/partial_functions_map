# Usage Guide

## Setup First

See `SETUP_QUICK_START.md` for environment setup.

```bash
python3 setup_environment.py
source .venv/bin/activate
```

## Running the Project

### Basic Usage

```bash
# Default (FPFH descriptor)
python3 -m pfm_py.main --data-path /path/to/data

# With SHOT descriptor
python3 -m pfm_py.main --shot --data-path /path/to/data

# With FPFH descriptor (explicit)
python3 -m pfm_py.main --fpfh --data-path /path/to/data
```

### Options

| Parameter | Description |
|-----------|-------------|
| `--data-path PATH` | Path to data directory |
| `--fpfh` | Use FPFH descriptors (default) |
| `--shot` | Use SHOT descriptors |
| `--help` | Show all options |

### Output

Results saved to `results/` directory with visualizations and heatmaps.

