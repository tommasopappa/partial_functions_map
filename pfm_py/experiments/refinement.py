from pfm_py.dataset import MeshPair
from pfm_py.dataset.shrec16 import Shrec16
from pfm_py.options import Options
import pfm_py.main as main

import torch
import numpy as np
import os
import argparse
import json
from typing import List, Tuple

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description='Partial Functions Map - Refinement Iteration Study',
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
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Random seed for sample selection (optional)'
)

args = parser.parse_args()

# Data path
data_path = args.data_path
target_path = args.target_path
seed = args.seed

print(f"Data path: {data_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
opts = Options(device)
opts.C_max_iter = 2000
opts.v_max_iter = 2000
opts.early_stopping = False
opts.descriptor_type = "shot"

refine_values = [0, 1, 3, 5, 7]

def _collect_all(data_path: str, folder: str) -> List[MeshPair]:
    iterator = Shrec16(data_path, [folder]).__iter__()
    return list(iterator)

def _shuffle_and_take(samples: List[MeshPair], limit: int, rng: np.random.Generator) -> List[MeshPair]:
    if len(samples) <= limit:
        return samples
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled[:limit]

rng = np.random.default_rng(seed)

cuts_all = _collect_all(data_path, "cuts")
holes_all = _collect_all(data_path, "holes")

cuts_samples = _shuffle_and_take(cuts_all, 15, rng)
holes_samples = _shuffle_and_take(holes_all, 15, rng)
samples: List[Tuple[MeshPair, str]] = [(s, "cuts") for s in cuts_samples] + [(s, "holes") for s in holes_samples]

print(f"\nCollected {len(samples)} samples:")
for i, (s, cat) in enumerate(samples):
    print(f"  {i+1}. {s.name} ({cat})")

results_file = os.path.join(target_path, "refinement_statistics.txt")
results_json = os.path.join(target_path, "refinement_statistics.json")
os.makedirs(target_path, exist_ok=True)

with open(results_file, "w") as f:
    f.write("="*100 + "\n")
    f.write("REFINEMENT ITERATION STUDY: Effect of opts.refine_iters\n")
    f.write("="*100 + "\n")
    f.write("30 samples (15 cuts, 15 holes) × 5 refine_iters values per sample\n")
    f.write(f"refine_iters values: {refine_values}\n")
    f.write("="*100 + "\n\n")

print("\n" + "="*80)
print("TESTING EFFECT OF refine_iters")
print("="*80)
print(f"Testing {len(samples)} samples with refine_iters values {refine_values}\n")

all_mge = []
per_sample_best = []
sample_records = []

for sample_idx, (sample, category) in enumerate(samples):
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx+1}/{len(samples)}: {sample.name} ({category})")
    print(f"{'='*80}")

    mge_by_refine = {}
    for r_idx, r in enumerate(refine_values):
        print(f"  refine_iters={r}...", end=" ", flush=True)
        opts.refine_iters = r
        res = main.run(sample, output_folder=None, opts=opts, target_path=target_path)
        mge = float(res['mean'])
        mge_by_refine[r] = mge
        print(f"MGE={mge:.6f}")

    all_mge.append(mge_by_refine)

    baseline = mge_by_refine[0]
    # choose smallest refine_iters in case of ties
    best_r = min(refine_values, key=lambda rv: (mge_by_refine[rv], refine_values.index(rv)))
    best_mge = mge_by_refine[best_r]

    if np.isfinite(baseline) and baseline > 0:
        improvement_pct = 100.0 * (baseline - best_mge) / baseline
    else:
        improvement_pct = float('nan')

    per_sample_best.append((best_r, best_mge, improvement_pct))
    sample_records.append({
        "name": sample.name,
        "category": category,
        "mge_by_refine": {str(k): float(v) for k, v in mge_by_refine.items()},
        "best_refine_iters": int(best_r),
        "best_mge": float(best_mge),
        "improvement_pct_vs_0": float(improvement_pct),
    })

    print("\nMGE values:")
    for r in refine_values:
        print(f"  refine_iters={r}: {mge_by_refine[r]:.6f}")
    print(f"Best refine_iters={best_r} with MGE={best_mge:.6f}")
    print(f"Relative improvement vs refine_iters=0: {improvement_pct:.2f}%")

    with open(results_file, "a") as f:
        f.write(f"\nSample {sample_idx+1}: {sample.name} ({category})\n")
        f.write(f"{'-'*100}\n")
        for r in refine_values:
            f.write(f"  refine_iters={r}: MGE={mge_by_refine[r]:.6f}\n")
        f.write(f"  Best refine_iters: {best_r}\n")
        f.write(f"  Best MGE: {best_mge:.6f}\n")
        f.write(f"  Relative improvement vs refine_iters=0: {improvement_pct:.2f}%\n")
        f.flush()

print("\n" + "="*80)
print("AGGREGATE SUMMARY")
print("="*80)

baseline_vals = np.array([m[0] for m in all_mge], dtype=float)
baseline_mask = np.isfinite(baseline_vals) & (baseline_vals > 0)
baseline_mean = float(np.mean(baseline_vals[baseline_mask])) if np.any(baseline_mask) else float('nan')

categories = [cat for _, cat in samples]
cuts_mask = np.array([cat == "cuts" for cat in categories], dtype=bool)
holes_mask = np.array([cat == "holes" for cat in categories], dtype=bool)

baseline_mean_cuts = float(np.mean(baseline_vals[baseline_mask & cuts_mask])) if np.any(baseline_mask & cuts_mask) else float('nan')
baseline_mean_holes = float(np.mean(baseline_vals[baseline_mask & holes_mask])) if np.any(baseline_mask & holes_mask) else float('nan')

aggregate_records = []
for r in refine_values:
    best_count = sum(1 for (best_r, _, _) in per_sample_best if best_r == r)
    best_pct = 100.0 * best_count / len(samples) if samples else 0.0
    vals = np.array([m[r] for m in all_mge], dtype=float)
    mask = np.isfinite(vals) & baseline_mask
    if np.any(mask):
        avg_mge = float(np.mean(vals[mask]))
        if np.isfinite(baseline_mean) and baseline_mean > 0:
            avg_improve = 100.0 * (baseline_mean - avg_mge) / baseline_mean
        else:
            avg_improve = float('nan')
    else:
        avg_mge = float('nan')
        avg_improve = float('nan')

    # Category-specific averages
    mask_cuts = mask & cuts_mask
    mask_holes = mask & holes_mask
    if np.any(mask_cuts) and np.isfinite(baseline_mean_cuts) and baseline_mean_cuts > 0:
        avg_mge_cuts = float(np.mean(vals[mask_cuts]))
        avg_improve_cuts = 100.0 * (baseline_mean_cuts - avg_mge_cuts) / baseline_mean_cuts
    else:
        avg_mge_cuts = float('nan')
        avg_improve_cuts = float('nan')

    if np.any(mask_holes) and np.isfinite(baseline_mean_holes) and baseline_mean_holes > 0:
        avg_mge_holes = float(np.mean(vals[mask_holes]))
        avg_improve_holes = 100.0 * (baseline_mean_holes - avg_mge_holes) / baseline_mean_holes
    else:
        avg_mge_holes = float('nan')
        avg_improve_holes = float('nan')

    print(
        f"refine_iters={r}: optimal for {best_pct:.1f}% of samples; "
        f"avg improvement vs 0: {avg_improve:.2f}% "
        f"(cuts {avg_improve_cuts:.2f}%, holes {avg_improve_holes:.2f}%)"
    )

    with open(results_file, "a") as f:
        f.write(f"\nrefine_iters={r}: optimal for {best_pct:.1f}% of samples\n")
        f.write(f"  Avg improvement vs refine_iters=0: {avg_improve:.2f}%\n")
        f.write(f"  Avg improvement vs 0 (cuts): {avg_improve_cuts:.2f}%\n")
        f.write(f"  Avg improvement vs 0 (holes): {avg_improve_holes:.2f}%\n")

    aggregate_records.append({
        "refine_iters": int(r),
        "optimal_pct": float(best_pct),
        "avg_improvement_pct_vs_0": float(avg_improve),
        "avg_improvement_pct_vs_0_cuts": float(avg_improve_cuts),
        "avg_improvement_pct_vs_0_holes": float(avg_improve_holes),
        "avg_mge": float(avg_mge),
        "avg_mge_cuts": float(avg_mge_cuts),
        "avg_mge_holes": float(avg_mge_holes),
    })

summary_payload = {
    "refine_iters_values": [int(r) for r in refine_values],
    "num_samples": int(len(samples)),
    "baseline_refine_iters": 0,
    "seed": seed,
    "samples": sample_records,
    "aggregate": aggregate_records,
}

with open(results_json, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2)

print(f"\nComplete results saved to: {results_file}")
print(f"JSON summary saved to: {results_json}")
