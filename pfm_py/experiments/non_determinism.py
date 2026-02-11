from asyncio import run
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options
import pfm_py.main as main

import torch
import open3d as o3d
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout

from typing import Iterator

class SampleIterator:
    """Iterator that yields TestMeshData objects for all partial meshes"""
    
    def __init__(self, data_path: str, target_path: str, partial_folders: list):
        self.data_path = data_path
        self.target_path = target_path
        self.partial_folders = partial_folders
    
    def __iter__(self) -> Iterator['main.TestMeshData']:
        """Iterate over all samples across all partial folders"""
        for folder in self.partial_folders:
            partial_files = os.listdir(os.path.join(self.data_path, "SHREC16", folder, "off"))
            
            for partial_file in partial_files:
                # Remove extension safely
                partial_mesh_name = os.path.splitext(partial_file)[0]
                
                # Safe extraction of the full mesh name from the partial's filename
                parts = partial_mesh_name.split('_')
                if len(parts) >= 2:
                    full_mesh_name = parts[1]
                else:
                    full_mesh_name = partial_mesh_name
                
                # Create and yield mesh data
                yield main.TestMeshData(
                    name=partial_mesh_name,
                    full_mesh=os.path.join(self.data_path, "SHREC16", "null", "off", f"{full_mesh_name}.off"),
                    partial_mesh=os.path.join(self.data_path, "SHREC16", folder, "off", partial_file),
                    ground_truth=os.path.join(self.data_path, "SHREC16", folder, "corres", f"{partial_mesh_name}.vts")
                )

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description='Partial Functions Map - Randomness Statistics',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=""
)

parser.add_argument(
    '--data-path',
    type=str,
    default='/usr/prakt/w0010/SAVHA/shape_data',
    help='Path to the shape data directory'
)
parser.add_argument(
    '--target-path',
    type=str,
    default='results',
    help='Path to the output results directory'
)

args = parser.parse_args()

# Data path
data_path = args.data_path
target_path = args.target_path

print(f"Data path: {data_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
opts = Options(device)
opts.C_max_iter = 2000
opts.v_max_iter = 2000
opts.early_stopping = False
opts.descriptor_type = "shot"

# Collect 10 samples (5 cuts, 5 holes)
sample_iterator = SampleIterator(data_path, target_path, ["cuts", "holes"]).__iter__()

samples = []
sample_categories = []
cuts_count = 0
holes_count = 0

for sample in sample_iterator:
    if "cuts" in sample.partial_mesh:
        if cuts_count < 5:
            samples.append(sample)
            sample_categories.append("cuts")
            cuts_count += 1
    else:
        if holes_count < 5:
            samples.append(sample)
            sample_categories.append("holes")
            holes_count += 1
    
    if cuts_count >= 5 and holes_count >= 5:
        break

print(f"\nCollected {len(samples)} samples:")
for i, (s, cat) in enumerate(zip(samples, sample_categories)):
    print(f"  {i+1}. {s.name} ({cat})")

# Open results file
results_file = os.path.join(target_path, "randomness_statistics.txt")
os.makedirs(target_path, exist_ok=True)

with open(results_file, "w") as f:
    f.write("="*100 + "\n")
    f.write("RANDOMNESS STATISTICS: Effect of Random Eigenvector Sign Flips\n")
    f.write("="*100 + "\n")
    f.write("10 samples (5 cuts, 5 holes) × 10 runs per sample with random sign flips\n")
    f.write("Each run uses the default eigendecomposition (random initial vector)\n")
    f.write("="*100 + "\n\n")

print("\n" + "="*80)
print("TESTING RANDOMNESS IN EIGENDECOMPOSITION")
print("="*80)
print(f"Testing {len(samples)} samples with 10 runs each\n")

# Process each sample
for sample_idx, (sample, category) in enumerate(zip(samples, sample_categories)):
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx+1}/{len(samples)}: {sample.name} ({category})")
    print(f"{'='*80}")
    
    mge_values = []
    
    # Run pipeline 10 times
    for run_idx in range(10):
        print(f"  Run {run_idx+1}/10...", end=" ", flush=True)
        
        # Suppress output from main.run
        with redirect_stdout(io.StringIO()):
            mge = main.run(sample, output_folder=None, opts=opts)
        
        mge_values.append(mge)
        print(f"MGE={mge:.6f}")
    
    mge_values = np.array(mge_values)
    
    # Calculate statistics
    min_mge = np.min(mge_values)
    max_mge = np.max(mge_values)
    mean_mge = np.mean(mge_values)
    std_mge = np.std(mge_values, ddof=1)  # ddof=1 for sample std
    width_mge = max_mge - min_mge
    
    # Print to console
    print(f"\nStatistics for {sample.name}:")
    print(f"  Min MGE:      {min_mge:.6f}")
    print(f"  Max MGE:      {max_mge:.6f}")
    print(f"  Mean MGE:     {mean_mge:.6f}")
    print(f"  Std Dev:      {std_mge:.6f}")
    print(f"  Width (max-min): {width_mge:.6f}")
    
    # Write to file
    with open(results_file, "a") as f:
        f.write(f"\nSample {sample_idx+1}: {sample.name} ({category})\n")
        f.write(f"{'-'*100}\n")
        f.write(f"MGE values (10 runs): {np.array2string(mge_values, separator=', ', precision=6)}\n")
        f.write(f"  Min:         {min_mge:.6f}\n")
        f.write(f"  Max:         {max_mge:.6f}\n")
        f.write(f"  Mean:        {mean_mge:.6f}\n")
        f.write(f"  StdDev:      {std_mge:.6f}\n")
        f.write(f"  Width:       {width_mge:.6f}\n")
        f.flush()
    
    print(f"✓ Results written to {results_file}")

print("\n" + "="*80)
print("ALL SAMPLES PROCESSED")
print("="*80)
print(f"Complete results saved to: {results_file}")

# Print final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

with open(results_file, "r") as f:
    content = f.read()

print(content)
