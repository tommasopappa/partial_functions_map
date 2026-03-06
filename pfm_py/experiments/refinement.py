from pfm_py.dataset.mesh_pair import MeshPair
from pfm_py.dataset.shrec16 import Shrec16
from pfm_py.options import Options
import pfm_py.main as main

import torch
import numpy as np
import os
import argparse
import json
from typing import List, Tuple


def format_num(val, decimals=3):
    """Format a number to fixed decimal places, handling NaN/Inf."""
    if not np.isfinite(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def generate_html(sample_records, aggregate_records, refine_values, target_path):
    """Generate HTML report with two tables: by sample and by refinement iteration."""
    html_lines = [
        '<!DOCTYPE html>',
        '<html>',
        '<head>',
        '<meta charset="utf-8" />',
        '<title>Refinement Iteration Study</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }',
        'h1, h2 { color: #333; }',
        'table { border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        'th, td { border: 1px solid #ddd; padding: 10px; text-align: right; }',
        'th { background: #4CAF50; color: white; font-weight: bold; }',
        'tr:nth-child(even) { background: #f9f9f9; }',
        'tr:hover { background: #f0f0f0; }',
        '.sample-name { text-align: left; }',
        '.category { font-size: 0.9em; color: #666; }',
        '.best { background: #c8e6c9; font-weight: bold; }',
        '.improvement-positive { color: green; }',
        '.improvement-negative { color: red; }',
        '</style>',
        '</head>',
        '<body>',
        '<h1>Refinement Iteration Study - SHOT Descriptors</h1>',
        '<p>Full dataset experiment: All cuts + All holes samples from SHREC16</p>',
        '<p>Refinement iterations tested: ' + ', '.join(map(str, refine_values)) + '</p>',
    ]
    
    # TABLE 1: Samples as rows, refinement iterations as columns
    html_lines.extend([
        '<h2>Table 1: Results by Sample</h2>',
        '<p>Mean Geodesic Error (MGE) for each sample across refinement iterations. Best value per sample highlighted.</p>',
        '<table>',
        '<tr><th class="sample-name">Sample</th><th class="sample-name">Category</th>',
    ])
    
    for r in refine_values:
        html_lines.append(f'<th>refine={r}</th>')
    
    html_lines.extend([
        '<th>Best</th>',
        '<th>Improvement %</th>',
        '</tr>',
    ])
    
    for record in sample_records:
        name = record['name']
        category = record['category']
        mge_by_refine = record['mge_by_refine']
        best_r = record['best_refine_iters']
        improvement = record['improvement_pct_vs_0']
        
        html_lines.append(f'<tr><td class="sample-name">{name}</td><td class="category">{category}</td>')
        
        for r in refine_values:
            mge_val = mge_by_refine.get(str(r), float('nan'))
            is_best = (r == best_r)
            css_class = 'best' if is_best else ''
            html_lines.append(f'<td class="{css_class}">{format_num(mge_val)}</td>')
        
        html_lines.append(f'<td>{best_r}</td>')
        improvement_class = 'improvement-positive' if improvement > 0 else 'improvement-negative'
        html_lines.append(f'<td class="{improvement_class}">{format_num(improvement, 2)}%</td>')
        html_lines.append('</tr>')
    
    html_lines.extend([
        '</table>',
        '',
    ])
    
    # TABLE 2: Refinement iterations as rows, aggregate statistics as columns
    html_lines.extend([
        '<h2>Table 2: Aggregate Results by Refinement Iteration</h2>',
        '<p>Summary statistics across all samples for each refinement iteration value.</p>',
        '<table>',
        '<tr>',
        '<th>refine_iters</th>',
        '<th>Optimal For (%)</th>',
        '<th>Avg MGE</th>',
        '<th>Avg Improvement %</th>',
        '<th>Avg MGE (Cuts)</th>',
        '<th>Avg Improvement % (Cuts)</th>',
        '<th>Avg MGE (Holes)</th>',
        '<th>Avg Improvement % (Holes)</th>',
        '</tr>',
    ])
    
    for record in aggregate_records:
        r = record['refine_iters']
        optimal_pct = record['optimal_pct']
        avg_mge = record['avg_mge']
        avg_improve = record['avg_improvement_pct_vs_0']
        avg_mge_cuts = record['avg_mge_cuts']
        avg_improve_cuts = record['avg_improvement_pct_vs_0_cuts']
        avg_mge_holes = record['avg_mge_holes']
        avg_improve_holes = record['avg_improvement_pct_vs_0_holes']
        
        html_lines.append('<tr>')
        html_lines.append(f'<td>{r}</td>')
        html_lines.append(f'<td>{format_num(optimal_pct, 1)}%</td>')
        html_lines.append(f'<td>{format_num(avg_mge)}</td>')
        html_lines.append(f'<td>{format_num(avg_improve, 2)}%</td>')
        html_lines.append(f'<td>{format_num(avg_mge_cuts)}</td>')
        html_lines.append(f'<td>{format_num(avg_improve_cuts, 2)}%</td>')
        html_lines.append(f'<td>{format_num(avg_mge_holes)}</td>')
        html_lines.append(f'<td>{format_num(avg_improve_holes, 2)}%</td>')
        html_lines.append('</tr>')
    
    html_lines.extend([
        '</table>',
        '</body>',
        '</html>',
    ])
    
    html_path = os.path.join(target_path, 'refinement_study.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))
    
    return html_path


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
    default='results/refinement_experiment',
    help='Path to the output results directory'
)
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Random seed (not used when loading existing results)'
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

cuts_all = _collect_all(data_path, "cuts")
holes_all = _collect_all(data_path, "holes")

# Use all samples (no shuffling or limiting)
samples: List[Tuple[MeshPair, str]] = [(s, "cuts") for s in cuts_all] + [(s, "holes") for s in holes_all]

print(f"\nCollected {len(samples)} samples:")
print(f"  Cuts: {len(cuts_all)}")
print(f"  Holes: {len(holes_all)}")

results_json = os.path.join(target_path, "refinement_statistics.json")

# Load existing results to skip already-processed samples
already_processed = set()
existing_sample_records = []
if os.path.exists(results_json):
    try:
        with open(results_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_sample_records = existing_data.get("samples", [])
        already_processed = {record["name"] for record in existing_sample_records}
        print(f"\nFound existing results: {len(already_processed)} samples already processed")
        print(f"Will skip these and continue with remaining samples")
    except Exception as e:
        print(f"\nWarning: Could not load existing results: {e}")
        print("Starting fresh...")

# Filter out already-processed samples
samples_to_process = [(s, cat) for s, cat in samples if s.name not in already_processed]
print(f"\nSamples to process: {len(samples_to_process)}")
print(f"Already completed: {len(already_processed)}")
print(f"Total: {len(samples)}\n")

results_json = os.path.join(target_path, "refinement_statistics.json")
os.makedirs(target_path, exist_ok=True)

print("\n" + "="*80)
print("TESTING EFFECT OF refine_iters")
print("="*80)
print(f"Testing {len(samples_to_process)} samples with refine_iters values {refine_values}\n")

# Initialize with existing data
all_mge = []
per_sample_best = []
sample_records = list(existing_sample_records)

# Build existing aggregate data
for record in existing_sample_records:
    mge_by_refine = {int(k): v for k, v in record["mge_by_refine"].items()}
    all_mge.append(mge_by_refine)
    per_sample_best.append((record["best_refine_iters"], record["best_mge"], record["improvement_pct_vs_0"]))

for sample_idx, (sample, category) in enumerate(samples_to_process):
    print(f"\n{'='*80}")
    print(f"SAMPLE {len(existing_sample_records) + sample_idx+1}/{len(samples)}: {sample.name} ({category})")
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

    # Update JSON after each sample
    # Build full list including both existing and new samples for aggregate calculations
    all_samples = samples  # original full list
    processed_count = len(existing_sample_records) + sample_idx + 1
    
    baseline_vals = np.array([m[0] for m in all_mge], dtype=float)
    baseline_mask = np.isfinite(baseline_vals) & (baseline_vals > 0)
    baseline_mean = float(np.mean(baseline_vals[baseline_mask])) if np.any(baseline_mask) else float('nan')

    # Categories from all processed samples so far
    categories = [rec["category"] for rec in sample_records]
    cuts_mask = np.array([cat == "cuts" for cat in categories], dtype=bool)
    holes_mask = np.array([cat == "holes" for cat in categories], dtype=bool)

    baseline_mean_cuts = float(np.mean(baseline_vals[baseline_mask & cuts_mask])) if np.any(baseline_mask & cuts_mask) else float('nan')
    baseline_mean_holes = float(np.mean(baseline_vals[baseline_mask & holes_mask])) if np.any(baseline_mask & holes_mask) else float('nan')

    aggregate_records = []
    for r in refine_values:
        best_count = sum(1 for (best_r, _, _) in per_sample_best if best_r == r)
        best_pct = 100.0 * best_count / len(per_sample_best) if per_sample_best else 0.0
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
        "num_samples_processed": processed_count,
        "num_samples_total": len(all_samples),
        "baseline_refine_iters": 0,
        "seed": seed,
        "samples": sample_records,
        "aggregate": aggregate_records,
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Generate HTML report
    generate_html(sample_records, aggregate_records, refine_values, target_path)

print("\n" + "="*80)
print("STUDY COMPLETE")
print("="*80)

print(f"\nJSON results saved to: {results_json}")
print(f"HTML report saved to: {os.path.join(target_path, 'refinement_study.html')}")
