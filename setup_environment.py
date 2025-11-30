#!/usr/bin/env python3
"""
Environment Setup Script for Partial Functions Map Project

This script automates the creation of a virtual environment and installation
of all required dependencies for the project, including PyTorch, PyTorch3D,
and other scientific computing packages.

Usage:
    python3 setup_environment.py [options]

Options:
    --python VERSION    Python version to use (default: 3.12)
    --gpu              Install GPU-enabled PyTorch (default)
    --cpu              Install CPU-only PyTorch
    --recreate         Recreate venv even if it exists
    --help             Show this help message

Examples:
    python3 setup_environment.py
    python3 setup_environment.py --python 3.10 --cpu
    python3 setup_environment.py --recreate --gpu
"""

import argparse
import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def log_info(msg: str):
    """Log info message"""
    print(f"{Colors.GREEN}[INFO]{Colors.RESET} {msg}")


def log_warn(msg: str):
    """Log warning message"""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")


def log_error(msg: str):
    """Log error message"""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")


def print_header(msg: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{msg:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}\n")


def run_command(cmd: List[str], description: str = "") -> bool:
    """Run shell command and return success status"""
    if description:
        log_info(description)
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        log_error(f"Failed to run command: {e}")
        return False


def find_python(version: str) -> str:
    """Find Python executable with specified version"""
    candidates = [f"python{version}", "python3", "python"]
    
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            log_info(f"Found Python: {result.stdout.strip()}")
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return None


def setup_venv(venv_path: Path, python_cmd: str, recreate: bool = False) -> bool:
    """Create or verify virtual environment"""
    print_header("Virtual Environment Setup")
    
    if venv_path.exists():
        if recreate:
            log_warn(f"Removing existing venv at {venv_path}")
            import shutil
            shutil.rmtree(venv_path)
        else:
            log_info(f"Virtual environment already exists at {venv_path}")
            return True
    
    log_info(f"Creating virtual environment with {python_cmd}...")
    if not run_command([python_cmd, "-m", "venv", str(venv_path)], 
                       f"Creating venv at {venv_path}"):
        log_error("Failed to create virtual environment")
        return False
    
    log_info("Virtual environment created successfully")
    return True


def get_venv_commands(venv_path: Path) -> Tuple[str, str]:
    """Get venv Python and pip commands"""
    if sys.platform == "win32":
        python_cmd = str(venv_path / "Scripts" / "python.exe")
        pip_cmd = str(venv_path / "Scripts" / "pip.exe")
    else:
        python_cmd = str(venv_path / "bin" / "python")
        pip_cmd = str(venv_path / "bin" / "pip")
    
    return python_cmd, pip_cmd


def verify_venv(python_cmd: str, pip_cmd: str) -> bool:
    """Verify virtual environment is working"""
    try:
        subprocess.run([python_cmd, "--version"], check=True, capture_output=True)
        subprocess.run([pip_cmd, "--version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def install_packages(pip_cmd: str, packages: List[str], description: str) -> bool:
    """Install a list of packages"""
    print_header(description)
    cmd = [pip_cmd, "install", "--upgrade"] + packages
    return run_command(cmd, f"Installing: {', '.join(packages)}")


def install_pytorch(pip_cmd: str, gpu_mode: bool) -> bool:
    """Install PyTorch with appropriate configuration"""
    print_header("Installing PyTorch")
    
    if gpu_mode:
        log_info("Installing PyTorch for GPU (CUDA 12.1)")
        cmd = [
            pip_cmd, "install",
            "torch==2.1.0", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    else:
        log_info("Installing PyTorch for CPU")
        cmd = [
            pip_cmd, "install",
            "torch==2.1.0", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    
    if not run_command(cmd):
        return False
    
    # Verify GPU support if requested
    if gpu_mode:
        log_info("Verifying GPU support...")
        verify_cmd = [
            "python", "-c",
            "import torch; "
            "print('CUDA available:', torch.cuda.is_available()); "
            "print('CUDA version:', torch.version.cuda); "
            "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        ]
        # This is informational only, don't fail if it doesn't work
        subprocess.run(verify_cmd, capture_output=True)
    
    return True


def install_pytorch3d(pip_cmd: str, venv_path: Path) -> bool:
    """Install PyTorch3D from GitHub"""
    print_header("Installing PyTorch3D")
    
    # First install dependencies
    log_info("Installing PyTorch3D dependencies (fvcore, iopath)...")
    if not run_command([pip_cmd, "install", "fvcore", "iopath"]):
        log_warn("Failed to install PyTorch3D dependencies, continuing...")
    
    log_info("Installing pytorch3d from GitHub (this may take several minutes)...")
    log_warn("Compilation in progress... Please wait.")
    
    # Try with --no-build-isolation first (faster for pre-installed torch)
    cmd_no_isolation = [
        pip_cmd, "install",
        "--no-build-isolation",
        "git+https://github.com/facebookresearch/pytorch3d.git"
    ]
    
    if run_command(cmd_no_isolation, "Attempting no-build-isolation installation..."):
        log_info("PyTorch3D installed successfully (no-build-isolation method)")
        return True
    
    # Fallback to standard installation
    log_warn("No-build-isolation method failed, trying standard installation...")
    cmd_standard = [
        pip_cmd, "install",
        "git+https://github.com/facebookresearch/pytorch3d.git"
    ]
    
    if run_command(cmd_standard):
        log_info("PyTorch3D installed successfully (standard method)")
        return True
    
    log_error("Failed to install PyTorch3D")
    return False


def verify_installation(python_cmd: str) -> bool:
    """Verify all packages are installed correctly"""
    print_header("Verifying Installation")
    
    packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'pytorch3d': 'pytorch3d',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'transformers': 'transformers',
        'diffusers': 'diffusers',
        'einops': 'einops',
        'opencv-python': 'cv2',
        'trimesh': 'trimesh',
        'meshio': 'meshio',
        'plyfile': 'plyfile',
        'potpourri3d': 'potpourri3d',
        'robust_laplacian': 'robust_laplacian',
    }
    
    verify_script = """
import sys
import importlib

packages = {
    %s
}

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}\\n")

failed = []
for display_name, import_name in packages.items():
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            mod = importlib.import_module(import_name)
            version = getattr(mod, '__version__', 'unknown')
            status = "✓"
        else:
            failed.append(display_name)
            status = "✗"
            version = "NOT FOUND"
    except Exception as e:
        failed.append(display_name)
        status = "✗"
        version = f"ERROR: {e}"
    
    print(f"{status} {display_name:<25} {version}")

if failed:
    print(f"\\n⚠ {len(failed)} package(s) failed: {', '.join(failed)}")
    sys.exit(1)
else:
    print(f"\\n✓ All {len(packages)} packages verified!")
    sys.exit(0)
""" % ", ".join(f"'{name}': '{import_name}'" 
                for name, import_name in packages.items())
    
    result = subprocess.run(
        [python_cmd, "-c", verify_script],
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version to use (default: 3.12)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Install GPU-enabled PyTorch (default)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Install CPU-only PyTorch"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate venv even if it exists"
    )
    
    args = parser.parse_args()
    
    if args.cpu:
        args.gpu = False
    
    print_header("Partial Functions Map - Environment Setup")
    
    # Find Python
    log_info(f"Looking for Python {args.python}...")
    python_cmd = find_python(args.python)
    
    if not python_cmd:
        log_error(f"Python {args.python} not found. Please install it first.")
        return 1
    
    # Setup paths
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv"
    
    # Create venv
    if not setup_venv(venv_path, python_cmd, args.recreate):
        return 1
    
    venv_python, venv_pip = get_venv_commands(venv_path)
    
    if not verify_venv(venv_python, venv_pip):
        log_error("Virtual environment verification failed")
        return 1
    
    log_info(f"Using Python: {venv_python}")
    log_info(f"Using Pip: {venv_pip}")
    
    # Upgrade pip
    print_header("Upgrading pip and build tools")
    subprocess.run([venv_pip, "install", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.run([venv_pip, "install", "--upgrade", "build", "cmake", "ninja"])
    
    # Install base packages
    install_packages(venv_pip, [
        "numpy==1.25.0",
        "scipy==1.10.1",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1",
    ], "Installing Base Dependencies")
    
    # Install PyTorch
    if not install_pytorch(venv_pip, args.gpu):
        log_warn("PyTorch installation had issues, continuing...")
    
    # Install ML packages
    install_packages(venv_pip, [
        "transformers==4.34.1",
        "diffusers==0.21.4",
        "huggingface-hub==0.17.3",
        "einops==0.7.0",
        "opencv-python==4.8.1.78",
    ], "Installing ML and Vision Packages")
    
    # Install mesh packages
    install_packages(venv_pip, [
        "meshio==5.3.4",
        "plyfile==1.0.1",
        "trimesh==4.0.0",
        "potpourri3d==1.0.0",
        "robust_laplacian==0.2.7",
    ], "Installing Mesh and Geometry Packages")
    
    # Install optional packages
    print_header("Installing Optional Packages")
    
    optional_packages = [
        ("meshplot", [venv_pip, "install", "meshplot"]),
        ("xformers", [venv_pip, "install", "xformers==0.0.21"]),
    ]
    
    for name, cmd in optional_packages:
        if not run_command(cmd, f"Installing {name}..."):
            log_warn(f"{name} installation failed (optional)")
    
    # Install PyTorch3D
    if not install_pytorch3d(venv_pip, venv_path):
        log_error("PyTorch3D installation failed")
        return 1
    
    # Verify installation
    if not verify_installation(venv_python):
        print_header("Setup Completed with Warnings ⚠")
        log_warn("Some packages failed verification")
        return 1
    
    # Success
    print_header("Setup Complete ✓")
    log_info("Virtual environment is ready!")
    print(f"\n{Colors.GREEN}Next steps:{Colors.RESET}")
    print(f"1. Activate the environment:")
    print(f"   source {venv_path}/bin/activate")
    print(f"\n2. Run the project:")
    print(f"   python pfm_py/main.py --data-path /path/to/data --fpfh")
    print(f"   # or for SHOT descriptor:")
    print(f"   python pfm_py/main.py --data-path /path/to/data --shot")
    print(f"\n3. Deactivate when done:")
    print(f"   deactivate")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
