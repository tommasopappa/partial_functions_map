# Usage Guide

## Basic Usage

The project supports configuring **descriptor type** and **data path** through command-line parameters.

### 1. Using FPFH Descriptors (Default)

```bash
cd /usr/prakt/w0010/partial_functions_map
. venv/bin/activate
export PYTHONPATH=.
python pfm_py/main.py
```

Or explicitly specify:

```bash
python pfm_py/main.py --fpfh
```

### 2. Using SHOT Descriptors

```bash
python pfm_py/main.py --shot
```

### 3. Specifying a Custom Data Path

If your data is not in the default location `/usr/prakt/w0010/SAVHA/shape_data`, you can specify a custom path:

```bash
python pfm_py/main.py --data-path /your/path/to/data
```

### 4. Combining Parameters

```bash
# Use SHOT with custom data path
python pfm_py/main.py --shot --data-path /path/to/data

# Use FPFH with custom data path
python pfm_py/main.py --fpfh --data-path ~/data/shapes
```

## Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fpfh` | Use FPFH descriptors | ✓ Default |
| `--shot` | Use SHOT descriptors | - |
| `--data-path PATH` | Specify data root directory path | `/usr/prakt/w0010/SAVHA/shape_data` |

## Data Directory Structure

Your data directory should contain the following structure:

```
data_path/
├── SHREC16/
│   ├── null/off/          # Complete meshes
│   ├── cuts/off/          # Cut meshes
│   ├── holes/off/         # Meshes with holes
│   ├── cuts/corres/       # Correspondence files
│   └── holes/corres/      # Correspondence files
```

## Output

Results are saved in the `results/` directory. Each test case generates:

- `pfm_visualization.png` - Function transfer quality and error heatmaps
- `indexed_color_transfer.png` - Vertex color mapping visualization

## Help

To view all available options:

```bash
python pfm_py/main.py --help
```

## Launch Script

You can also use the project's launch script (if created):

```bash
cd /usr/prakt/w0010/partial_functions_map
./run.sh --shot --data-path /your/data/path
```

## Examples

### Example 1: Run with default settings (FPFH, default data path)
```bash
python pfm_py/main.py
```

### Example 2: Run with SHOT descriptors
```bash
python pfm_py/main.py --shot
```

### Example 3: Run with custom data location
```bash
python pfm_py/main.py --data-path /mnt/shared/shape_data
```

### Example 4: Run SHOT with custom data path
```bash
python pfm_py/main.py --shot --data-path ~/projects/data/shapes
```

## Notes

- FPFH (Fast Point Feature Histograms) is faster but may be less distinctive
- SHOT (Signature of Histograms of Orientations) is more computationally intensive but more robust
- Ensure your data path is absolute or properly expanded (use `~` for home directory)
- Results directory will be created automatically if it doesn't exist

