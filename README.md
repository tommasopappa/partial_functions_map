# Partial Functional Maps (PFM) - Python Implementation

Comprehensive Python implementation of partial functional maps and related methods for non-rigid shape matching, including state-of-the-art feature extraction and refinement techniques.

## Overview

This repository implements the partial functional maps framework for establishing correspondences between 3D shapes with partial geometry. It combines spectral geometric methods with modern deep learning features and classical descriptors to achieve robust shape matching.

## Implemented Methods

### Core Methods

1. **Partial Functional Maps (PFM)** - Rodolà et al.
   - Spectral framework for partial shape matching
   - Handles missing geometry and topological changes
   - Laplace-Beltrami eigenbasis computation

2. **Functional Maps (FM)** - Ovsjanikov et al.
   - Complete shape correspondence framework
   - Spectral representation of correspondences
   - Foundation for partial methods

3. **Iterative Closest Point (ICP)**
   - Classical point-to-point registration
   - Baseline comparison method

### Feature Extractors

1. **DINOv2** - Meta AI
   - Self-supervised vision transformer features
   - Projection of 2D features onto 3D meshes
   - Multiple views rendering and aggregation

2. **SHOT** (Signature of Histograms of OrienTations)
   - Local geometric descriptor
   - Histogram-based feature encoding
   - Robust to noise and partial data

3. **FPFH** (Fast Point Feature Histograms)
   - Fast local geometric descriptor
   - Multi-scale neighborhood encoding
   - Efficient computation

### Method Combinations

All feature extractors can be combined with both FM and PFM refinement:
- DINO only
- DINO + FM
- DINO + PFM
- SHOT only
- SHOT + FM
- SHOT + PFM
- FPFH only
- FPFH + FM
- FPFH + PFM
- ICP (baseline)

## Benchmark Results

### Visual Comparison

Example showing correspondence quality across different methods on a challenging partial matching case:

![Method Comparison](method_comparison.png)

The visualization shows source shape (left), transferred/matched shape (center), and error visualization (right) for each method. Note how:
- **Ground Truth (GT)**: Perfect correspondence (0.000 error)
- **DINO+PFM**: Achieves excellent results (0.014 error) with proper part matching
- **DINO+FM**: Moderate performance (0.478 error) - struggles with partial geometry
- **DINO alone**: Poor performance (0.443 error) without refinement
- **ICP**: Worst performer (0.310 error) - fails on partial/cut shapes

### Performance by Shape Type

![Performance Heatmap](heatmap_by_shape.png)

Mean error across all test cases (ICP seems to perform well but usually doesn't have as full coverage over the matched shapes as the methods in question):

| Method | Mean Error |
|--------|------------|
| ICP | 0.133 |
| SHOT | 0.200 |
| SHOT+FM | 0.258 |
| SHOT+PFM | 0.120 |
| DINO | 0.238 |
| DINO+FM | 0.253 |
| DINO+PFM | 0.187 |
| FPFH | 0.520 |
| FPFH+FM | 0.525 |
| FPFH+PFM | 0.268 |

### Winner Distribution

No single method dominates across all cases. Each combination wins on different mesh pairs:

| Method | Wins | % | Avg Margin | Min Margin | Max Margin |
|--------|------|---|------------|------------|------------|
| SHOT+PFM | 82 | 41.4% | 0.0629 | 0.0000 | 0.3487 |
| DINO+PFM | 60 | 30.3% | 0.0300 | 0.0000 | 0.2366 |
| FPFH+PFM | 31 | 15.7% | 0.0455 | 0.0000 | 0.1732 |
| SHOT+argmax | 13 | 6.6% | 0.0773 | 0.0040 | 0.2617 |
| DINO+FM | 5 | 2.5% | 0.0261 | 0.0074 | 0.0538 |
| SHOT+FM | 4 | 2.0% | 0.0437 | 0.0295 | 0.0641 |
| DINO+argmax | 3 | 1.5% | 0.0196 | 0.0006 | 0.0386 |

**Key insight**: For optimal results, test all combinations and select the best performer per case. The average doesn't tell the whole story - different methods excel on different shape types and deformation patterns.

### Performance Characteristics by Shape

- **Cat**: PFM methods consistently strong (DINO+PFM, SHOT+PFM best)
- **Centaur**: SHOT-based methods excel, especially with PFM refinement
- **David**: High variability, no clear winner - test all methods
- **Dog**: SHOT+PFM and DINO+PFM competitive across all variants
- **Horse**: Classical SHOT performs well, PFM refinement helps
- **Michael**: Mixed results, SHOT+PFM edges ahead
- **Victoria**: DINO+PFM shows strong performance
- **Wolf**: ICP surprisingly effective, geometric descriptors strong

## Installation

### Recommended: Conda (Fast & Reliable)

Best for most users - installs PyTorch3D and all dependencies in ~10 minutes:

```bash
# Create environment
conda create -n pfm python=3.10 -y
conda activate pfm

# Core dependencies
conda install -c conda-forge matplotlib=3.7.1 numpy=1.25.0 scikit-learn=1.2.2 scipy=1.10.1 -y

# PyTorch (adjust cuda version as needed)
pip install torch==2.1.0 torchvision

# 3D/Geometry packages
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html

# Other dependencies
pip install open3d robust_laplacian==0.2.7 trimesh==4.0.0 potpourri3d==1.0.0
pip install transformers==4.34.1 huggingface-hub==0.17.3
pip install einops==0.7.0 meshio==5.3.4 opencv-python==4.8.1.78 plyfile==1.0.1
```

**See [INSTALLATION.md](INSTALLATION.md) for complete conda installation instructions and verification steps.**

### Alternative: Python venv

If you cannot use conda (e.g., restricted environments), use the automated setup script. **Note**: PyTorch3D compilation may take 30+ minutes:

```bash
python3 setup_environment.py
source .venv/bin/activate
```

**See [SETUP_ALTERNATIVE.md](SETUP_ALTERNATIVE.md) for venv installation options.**

## CLI Usage

PFM provides a simple CLI for single runs and dataset processing. The entry point is [pfm_py/main.py](pfm_py/main.py).

- Descriptors: `--fpfh` (default), `--shot`, `--dino`, `--dinov3`
- Inputs: `--full-mesh` (.off file or directory), `--partial-mesh` (.off file), `--gt-path` (optional .vts)
- Output: `--target-path` (results directory; default: `results`)
- Benchmark: `--benchmark` runs DINO, DINOv3, SHOT, FPFH and writes a comparative summary
- Iteration overrides: `--v-max-iter`, `--C-max-iter`, `--max-outer-iter`

Examples:

```bash
# Single run (SHOT)
python3 -m pfm_py.main \
   --full-mesh /path/to/full.off \
   --partial-mesh /path/to/partial.off \
   --shot \
   --target-path results/shot_single

# Single run (DINOv3) + GT + benchmark
python3 -m pfm_py.main \
   --full-mesh /path/to/full.off \
   --partial-mesh /path/to/partial.off \
   --gt-path /path/to/corres.vts \
   --dinov3 --benchmark \
   --target-path results/dinov3_benchmark

# Provide a full directory; auto-resolve the full mesh from the partial name
python3 -m pfm_py.main \
   --full-mesh /usr/prakt/w0010/SAVHA/shape_data/SHREC16/null/off \
   --partial-mesh /usr/prakt/w0010/SAVHA/shape_data/SHREC16/cuts/off/cuts_horse_shape_14.off \
   --dinov3 \
   --target-path results/dinov3

# Override iteration parameters (keep other defaults)
python3 -m pfm_py.main \
   --full-mesh /path/to/full.off \
   --partial-mesh /path/to/partial.off \
   --dino \
   --v-max-iter 3000 --C-max-iter 2500 --max-outer-iter 10 \
   --target-path results/dino_custom_iters
```

Notes:
- When `--full-mesh` is a directory, the CLI tries candidates derived from the partial name, e.g., `horse_shape_14.off → horse.off → partial_basename.off`.
- If single-run paths are omitted, the CLI uses internal SHREC16 defaults and iterates `cuts/holes`; with `--benchmark` it produces a summary page.
- To override the DINO model id, set env `PFM_DINO_MODEL`; for gated repos, set `HF_TOKEN`.

## Web Viewer (Interactive HTML)

- Purpose: Generate an interactive 3D page. Left panel shows the full mesh (continuous colors), right panel shows the partial mesh (method/GT colors), with rotate, zoom, pan and optional Method/GT toggle.
- Enable: Add `--web-view` to the CLI; an `interactive_view.html` is created per sample inside the result folder.

### Quick Example

```bash
python3 -m pfm_py.main \
   --full-mesh /usr/prakt/w0010/SAVHA/shape_data/SHREC16/null/off \
   --partial-mesh /usr/prakt/w0010/SAVHA/shape_data/SHREC16/cuts/off/cuts_horse_shape_14.off \
   --shot --web-view \
   --target-path results/webview \
   --v-max-iter 2000 --C-max-iter 2000 --max-outer-iter 7
```

Outputs in the sample folder:
- `interactive_view.html`
- `three.min.js` (bundled automatically by the generator)
- Optional: `full.json`, `partial_method.json`, `partial_gt.json` (non-embedded mode)

### Open in VS Code (recommended)

- Simple Browser:
   - Ctrl+Shift+P (Cmd+Shift+P on macOS) → “Simple Browser: Show”
   - Paste: `http://127.0.0.1:8001/interactive_view.html`
- Local server:
   ```bash
   cd <sample_dir>
   python3 -m http.server 8001 --bind 127.0.0.1
   # Then open http://127.0.0.1:8001/interactive_view.html in a browser or Simple Browser
   ```
   - If the port is busy (EADDRINUSE), pick another port (e.g., 8010) or kill the old process:
   ```bash
   pkill -f "http.server 8001" || true
   python3 -m http.server 8010 --bind 127.0.0.1
   ```

### Offline and Compatibility

- The generator bundles a local `three.min.js` into the output folder and the HTML references local scripts, so the page works without internet.
- A lightweight inline OrbitControls fallback is embedded; rotation (left mouse), zoom (wheel), and pan (right/middle mouse) work even without `OrbitControls.js`.
- Embedded mode (default) writes JSON data directly into the HTML so `file://` or local server both work; non-embedded mode fetches `.json` files next to the page.

### Troubleshooting

- Blank page: Hard refresh (Ctrl/Cmd+Shift+R) or check DevTools that the `<canvas>` has a non-zero size; prefer VS Code Simple Browser or a local server over `file://` if scripts aren’t running.
- Port in use: Change port or terminate the old server (see above).
- No interaction: Ensure `three.min.js` exists in the output folder; the inline fallback still enables basic controls if external `OrbitControls.js` is missing.

See the generator for details: [pfm_py/web_viewer.py](pfm_py/web_viewer.py).

## Key Features

- **Spectral Methods**: Laplace-Beltrami eigenbasis computation with cotangent weights
- **Modern Features**: DINOv2 self-supervised features via multi-view rendering
- **Classical Descriptors**: SHOT and FPFH for geometric feature extraction
- **Refinement**: Both full and partial functional map refinement
- **Comprehensive Benchmarks**: Automated testing across multiple shape categories
- **Visualization**: Built-in tools for correspondence visualization

## Dataset

Benchmarks use the SHREC dataset with two challenge types:
- **cuts**: Shapes with removed parts
- **holes**: Shapes with topological holes

Test shapes include: cat, centaur, david, dog, horse, michael, victoria, wolf

## References

- Rodolà et al. - Partial Functional Correspondence
- Ovsjanikov et al. - Functional Maps Framework
- Meta AI - DINOv2: Learning Robust Visual Features
- Tombari et al. - SHOT Descriptor
- Rusu et al. - FPFH Descriptor

## License

MIT License
