from pfm_py.dataset.shrec16 import Shrec16
from pfm_py.options import Options
import pfm_py.main as main

import torch
import numpy as np
import os
import argparse
import json
import io
from contextlib import redirect_stdout
import open3d as o3d
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine

def run(mesh_pair, opts: Options):
    mesh_M = o3d.io.read_triangle_mesh(mesh_pair.full_mesh)
    mesh_N = o3d.io.read_triangle_mesh(mesh_pair.partial_mesh)

    vert_M, triv_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
    vert_N, triv_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)
    M = ManifoldMesh(vert_M, triv_M, opts, compute_geo=True)
    N = ManifoldMesh(vert_N, triv_N, opts, compute_geo=False)

    # Non-deterministically flip eigenvectors
    for i in range(M.evecs.shape[1]):
        if torch.rand(1, device=M.evecs.device) < 0.5:
            M.evecs[:, i] = -M.evecs[:, i]
    for i in range(N.evecs.shape[1]):
        if torch.rand(1, device=N.evecs.device) < 0.5:
            N.evecs[:, i] = -N.evecs[:, i]

    C, v, matches = match_and_refine(M, N, opts)
    C, v, matches = C.numpy(force=True), v.numpy(force=True), matches.numpy(force=True)

    gt_matches = np.loadtxt(mesh_pair.ground_truth, dtype=float).astype(int) - 1
    geodesics_M = M.compute_geodesic_matrix()
    dist_method_geo = None
    mean_geodesic_error = float('nan')
    dist_method_geo = np.array([geodesics_M[gt_matches[i], matches[i]] for i in range(len(matches))])
    dist_method_geo = dist_method_geo / np.sqrt(M.area)
    mean_geodesic_error = dist_method_geo.mean()
    return mean_geodesic_error

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description='Partial Functions Map - Randomness Statistics',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=""
)

parser.add_argument(
    '--data-path',
    type=str,
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
opts.enforce_determinism = False

# Collect 10 samples (5 cuts, 5 holes)
sample_iterator = Shrec16(data_path, ["cuts", "holes"]).__iter__()

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
results_json = os.path.join(target_path, "randomness_statistics.json")
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

relative_std_list = []
relative_min_list = []
relative_max_list = []
sample_records = []

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
            mge = run(sample, opts)
        
        mge_values.append(mge)
        print(f"MGE={mge:.6f}")
    
    mge_values = np.array(mge_values)
    
    # Calculate statistics
    min_mge = np.min(mge_values)
    max_mge = np.max(mge_values)
    mean_mge = np.mean(mge_values)
    std_mge = np.std(mge_values, ddof=1)  # ddof=1 for sample std
    width_mge = max_mge - min_mge
    if mean_mge != 0:
        rel_std = std_mge / mean_mge
        rel_min = (mean_mge - min_mge) / mean_mge
        rel_max = (max_mge - mean_mge) / mean_mge
    else:
        rel_std = float('nan')
        rel_min = float('nan')
        rel_max = float('nan')
    relative_std_list.append(rel_std)
    relative_min_list.append(rel_min)
    relative_max_list.append(rel_max)
    
    # Print to console
    print(f"\nStatistics for {sample.name}:")
    print(f"  Min MGE:      {min_mge:.6f}")
    print(f"  Max MGE:      {max_mge:.6f}")
    print(f"  Mean MGE:     {mean_mge:.6f}")
    print(f"  Std Dev:      {std_mge:.6f}")
    print(f"  Width (max-min): {width_mge:.6f}")
    print(f"  Rel StdDev (std/mean): {rel_std:.6f}")
    print(f"  Rel Min ((mean-min)/mean): {rel_min:.6f}")
    print(f"  Rel Max ((max-mean)/mean): {rel_max:.6f}")
    
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
        f.write(f"  Rel StdDev (std/mean): {rel_std:.6f}\n")
        f.write(f"  Rel Min ((mean-min)/mean): {rel_min:.6f}\n")
        f.write(f"  Rel Max ((max-mean)/mean): {rel_max:.6f}\n")
        f.flush()

    sample_records.append({
        "name": sample.name,
        "category": category,
        "mge_values": [float(x) for x in mge_values.tolist()],
        "min_mge": float(min_mge),
        "max_mge": float(max_mge),
        "mean_mge": float(mean_mge),
        "std_mge": float(std_mge),
        "rel_std": float(rel_std),
        "rel_min": float(rel_min),
        "rel_max": float(rel_max),
    })
    
    print(f"✓ Results written to {results_file}")

print("\n" + "="*80)
print("ALL SAMPLES PROCESSED")
print("="*80)
print(f"Complete results saved to: {results_file}")

# Print final summary
avg_rel_std = float(np.nanmean(relative_std_list)) if relative_std_list else float('nan')
avg_rel_min = float(np.nanmean(relative_min_list)) if relative_min_list else float('nan')
avg_rel_max = float(np.nanmean(relative_max_list)) if relative_max_list else float('nan')

with open(results_file, "a") as f:
    f.write("\n" + "="*80 + "\n")
    f.write("RELATIVE SUMMARY (AVERAGED OVER SAMPLES)\n")
    f.write("="*80 + "\n")
    f.write(f"  Avg Rel StdDev (std/mean): {avg_rel_std:.6f}\n")
    f.write(f"  Avg Rel Min ((mean-min)/mean): {avg_rel_min:.6f}\n")
    f.write(f"  Avg Rel Max ((max-mean)/mean): {avg_rel_max:.6f}\n")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Avg Rel StdDev (std/mean): {avg_rel_std:.6f}")
print(f"Avg Rel Min ((mean-min)/mean): {avg_rel_min:.6f}")
print(f"Avg Rel Max ((max-mean)/mean): {avg_rel_max:.6f}")

with open(results_file, "r") as f:
    content = f.read()

summary_payload = {
    "num_samples": int(len(samples)),
    "runs_per_sample": 10,
    "samples": sample_records,
    "aggregate": {
        "avg_rel_std": float(avg_rel_std),
        "avg_rel_min": float(avg_rel_min),
        "avg_rel_max": float(avg_rel_max),
    },
}

with open(results_json, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print(f"\nJSON summary saved to: {results_json}")

print("\n" + content)
