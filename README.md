# Partial Functional Maps (PFM) - Python Implementation

Comprehensive Python implementation of partial functional maps and related methods for non-rigid shape matching, including state-of-the-art feature extraction and refinement techniques.

## Overview

This repository is a **Python reimplementation** of the [original MATLAB Partial Functional Maps (PFM) repository](https://github.com/pitbullil/PFM/tree/master/pfm), based on the foundational paper:

**[Partial Functional Correspondence](https://arxiv.org/pdf/1506.05274)** by Rodolà et al.

The implementation reproduces the core PFM framework for establishing correspondences between 3D shapes with partial geometry. Beyond the original MATLAB implementation (which uses SHOT descriptors), this Python version extends the framework with:
- **Deep learning-based descriptors**: DINOv2 and DINOv3 (self-supervised vision transformers)
- **Additional classical descriptors**: FPFH (Fast Point Feature Histograms)
- **Flexible descriptor combinations**: Different types of descriptors can be freely combined.

This enables robust and accurate shape matching across various deformation patterns and topological changes.

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

# SHREC16 dataset
python3 -m pfm_py.main \
   --shrec16 <path to SHREC16> \
   --shot --fpfh --desc shot+dino \
   --target-path results \
   --web-view \
   --max-outer-iter 5 \
   --C-max-iter 1500 \
   --v-max-iter 1000

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
- To override the DINO model id, set env `PFM_DINO_MODEL`; for gated repos, set `HF_TOKEN`.

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
