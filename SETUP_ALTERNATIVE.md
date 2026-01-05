# Alternative Installation (Python venv)

## ⚠️ Alternative Installation Method

**This is an alternative installation method for users who cannot use conda** (e.g., restricted environments, shared systems).

**Note:** PyTorch3D will be compiled from source, which may take 30+ minutes. For faster installation (~10 minutes), use the **recommended conda setup** in [INSTALLATION.md](INSTALLATION.md).

---

## Quick Start

```bash
# Setup (automated, ~15-20 min for dependencies + 30+ min for PyTorch3D compilation)
python3 setup_environment.py

# Activate
source .venv/bin/activate

# Run
python pfm_py/main.py --data-path /path/to/data --fpfh
```

## Setup Options

```bash
python3 setup_environment.py --cpu              # CPU-only
python3 setup_environment.py --python 3.10      # Python 3.10
python3 setup_environment.py --recreate         # Recreate venv
```

Or use Bash script:
```bash
bash setup_environment.sh
```

## When to Use This Method

Use this venv-based installation if:
- You don't have conda installed and cannot install it
- You're on a system where conda is not allowed
- You prefer virtualenv over conda environments
- You're comfortable with longer compilation times

Otherwise, **use conda** (see [INSTALLATION.md](INSTALLATION.md)) for:
- Faster installation (10 minutes vs 40+ minutes)
- Pre-built PyTorch3D binaries
- Better dependency management for scientific computing
- Standard approach for ML/3D geometry projects
