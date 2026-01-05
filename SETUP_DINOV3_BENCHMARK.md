# DINOv3 Benchmark Setup

## Prerequisites

- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Conda/Mamba installed
- HuggingFace account with access to DINOv3 models

## Environment Setup

### Environment Setup

```bash
# Create environment
conda create -n dinov3_pfm python=3.12 -y
conda activate dinov3_pfm

# Install all dependencies from conda-forge (exact versions from tested Colab)
conda install -c conda-forge \
    pytorch=2.9.1 torchvision=0.24.1 pytorch3d=0.7.9 \
    numpy=2.3.5 scipy=1.16.3 transformers=4.57.3 \
    huggingface_hub=0.36.0 matplotlib pandas \
    fvcore iopath -y

# Install pip packages
pip install open3d robust-laplacian trimesh potpourri3d
```

## HuggingFace Token

DINOv3 models require authentication. Set your HF token:

```bash
# Option 1: Environment variable
export HF_TOKEN="hf_your_token_here"

# Option 2: Login via CLI
huggingface-cli login
```

Get your token at: https://huggingface.co/settings/tokens

## Diff3F Setup (Optional, for FM baseline)

If you want DINOv3+FM comparisons, clone and setup Diff3F:

```bash
cd /path/to/pfm
git clone https://github.com/niladridutt/Diffusion-3D-Features.git
```

The benchmark script will auto-detect it if placed in the pfm directory.

## Data Setup

Ensure SHREC16 data is available:

```
shape_data/
└── SHREC16/
    ├── null/off/          # Full meshes (cat.off, dog.off, etc.)
    ├── cuts/
    │   ├── off/           # Partial meshes
    │   └── corres/        # Ground truth correspondences
    └── holes/
        ├── off/
        └── corres/
```

## Running the Benchmark

```bash
conda activate dinov3_pfm

# Representative sample (4 meshes, quick test)
python benchmark_Dinov3_only.py --data-path /path/to/shape_data

# Full benchmark (all meshes)
python benchmark_Dinov3_only.py --data-path /path/to/shape_data --all

# With Diff3F for FM comparisons
python benchmark_Dinov3_only.py --data-path /path/to/shape_data --diff3f-path ./Diffusion-3D-Features
```

## Output

Results are saved to `benchmark_dinov3_results/`:
- `benchmark_dinov3_results.json` — raw metrics
- `benchmark_dinov3_results.md` — summary table
- Per-mesh comparison figures

## Troubleshooting

### PyTorch3D Installation Issues

```bash
# If conda install fails, try pip with pre-built wheels
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py312_cu124_pyt25/download.html
```

### CUDA OOM Errors

Reduce number of views or image resolution:
```python
# In dinov3.py get_features_per_vertex()
num_views=16  # instead of 32
H=128, W=128  # instead of 256
```

### HuggingFace Access Issues

Ensure you've accepted the model terms at:
https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m
