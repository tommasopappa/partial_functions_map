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

# Descriptors to test
descriptors = ["shot", "dino", "shot+dino"]
print(f"Will test descriptors: {descriptors}")

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

print(f"\nCollected {len(samples)} samples:")
for i, (s, cat) in enumerate(zip(samples, sample_categories)):
    print(f"  {i+1}. {s.name} ({cat})")

# Open results file (include descriptor name for clarity)
results_file = os.path.join(target_path, "descriptor_comparison.txt")
os.makedirs(target_path, exist_ok=True)

with open(results_file, "w") as f:
    f.write("="*100 + "\n")
    f.write("DESCRIPTOR COMPARISON: Single Pass Results\n")
    f.write("="*100 + "\n")
    f.write(f"Testing {len(samples)} samples with descriptors: {', '.join(descriptors)}\n")
    f.write("Each sample run once per descriptor.\n")
    f.write("="*100 + "\n\n")

print("\n" + "="*80)
print("TESTING DESCRIPTORS")
print("="*80)
print(f"Testing {len(samples)} samples with {len(descriptors)} descriptors each\n")

# Collect results for all descriptors and samples
results_dict = {desc: {} for desc in descriptors}

# Loop over descriptors
for desc in descriptors:
    print(f"\n{'='*80}")
    print(f"DESCRIPTOR: {desc.upper()}")
    print(f"{'='*80}\n")
    
    opts = Options(device)
    opts.C_max_iter = 2000
    opts.v_max_iter = 2000
    opts.early_stopping = False
    opts.descriptor_type = desc
    
    # Process each sample once
    for sample_idx, (sample, category) in enumerate(zip(samples, sample_categories)):
        print(f"  Sample {sample_idx+1}/{len(samples)}: {sample.name} ({category})...", end=" ", flush=True)
        
        # Suppress output from main.run
        with redirect_stdout(io.StringIO()):
            mge = main.run(sample, output_folder=None, opts=opts, target_path=target_path)
        
        results_dict[desc][sample.name] = {
            'mge': mge,
            'category': category
        }
        print(f"MGE={mge:.6f}")

print("\n" + "="*80)
print("ALL SAMPLES PROCESSED")
print("="*80)

# Write detailed results to file
with open(results_file, "a") as f:
    for sample_idx, (sample, category) in enumerate(zip(samples, sample_categories)):
        f.write(f"\nSample {sample_idx+1}: {sample.name} ({category})\n")
        f.write(f"{'-'*100}\n")
        for desc in descriptors:
            mge = results_dict[desc][sample.name]['mge']
            f.write(f"  {desc:15s}: MGE = {mge:.6f}\n")
        f.write("\n")

# Write summary table
with open(results_file, "a") as f:
    f.write("\n" + "="*100 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("="*100 + "\n\n")
    
    # Table header
    f.write(f"{'Sample':<40} {'Category':<10}")
    for desc in descriptors:
        f.write(f" {desc:>15}")
    f.write("\n")
    f.write("-"*100 + "\n")
    
    # Table rows
    for sample_idx, (sample, category) in enumerate(zip(samples, sample_categories)):
        f.write(f"{sample.name:<40} {category:<10}")
        for desc in descriptors:
            mge = results_dict[desc][sample.name]['mge']
            f.write(f" {mge:>15.6f}")
        f.write("\n")
    
    # Summary statistics
    f.write("\n" + "-"*100 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("-"*100 + "\n")
    for desc in descriptors:
        mge_vals = [results_dict[desc][s.name]['mge'] for s in samples]
        mge_arr = np.array(mge_vals)
        f.write(f"\n{desc.upper()}:\n")
        f.write(f"  Mean MGE:  {mge_arr.mean():.6f}\n")
        f.write(f"  Std Dev:   {mge_arr.std():.6f}\n")
        f.write(f"  Min MGE:   {mge_arr.min():.6f}\n")
        f.write(f"  Max MGE:   {mge_arr.max():.6f}\n")

print(f"Complete results saved to: {results_file}\n")

# Print summary
with open(results_file, "r") as f:
    content = f.read()

print(content)
