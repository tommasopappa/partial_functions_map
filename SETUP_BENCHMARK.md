# Environment Setup for PFM Benchmark

## Quick Setup (Conda)

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

# For Diffusion-3D-Features (required for FM baseline)
pip install diffusers==0.21.4 accelerate==0.20.3
pip install xformers==0.0.22.post7
```

## Clone Diffusion-3D-Features (for FM baseline)

```bash
git clone https://github.com/niladridutt/Diffusion-3D-Features.git
```

## Verify Installation

```python
import torch
import pytorch3d
import open3d as o3d
import robust_laplacian
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch3D: {pytorch3d.__version__}")
```

## Running the Benchmark

```bash
# Representative sample (4 pairs: 2 cuts + 2 holes)
python -m pfm_py.benchmark --data-path /usr/prakt/w0010/SAVHA/shape_data --output benchmark_sample

# Full benchmark (all mesh pairs)
python -m pfm_py.benchmark --data-path /usr/prakt/w0010/SAVHA/shape_data --output benchmark_all --all

# Run in background with logging
nohup python -m pfm_py.benchmark --data-path /usr/prakt/w0010/SAVHA/shape_data --output benchmark_all --all > benchmark.log 2>&1 &

# Monitor progress
tail -f benchmark.log
```

## Benchmark Output

For each descriptor (DINO, SHOT, FPFH), the benchmark computes:

| Method | Description |
|--------|-------------|
| Argmax | Simple nearest neighbor in feature space |
| +FM | Standard Functional Maps (Diff3F) |
| +PFM | Partial Functional Maps (our pipeline) |
| ICP | Iterative Closest Point baseline |

Output files:
- `benchmark_results.json` - Full results
- `benchmark_results.md` - Markdown summary table
- Per-mesh comparison figures (5 rows: GT, Argmax, FM, PFM, ICP)

## Notes

- PyTorch3D wheel is for CUDA 12.1 + PyTorch 2.1.0. Adjust URL for other versions:
  - Check available wheels: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- For CPU-only, skip pytorch3d wheel and build from source
- DINO model downloads automatically from torch.hub on first run
- Random seed is set to 42 for reproducibility
