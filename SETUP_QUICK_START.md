# Environment Setup

## Quick Start

```bash
# Setup (automated, ~15-20 min)
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

## Manual Setup (if needed)

See `ENVIRONMENT_SETUP.md` for step-by-step instructions.
