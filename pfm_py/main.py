from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options
from pfm_py.web_viewer import generate_interactive_view
from pfm_py.dataset.mesh_pair import MeshPair
from pfm_py.dataset.shrec16 import Shrec16
from pfm_py.visualizations import (
    create_functional_map_visualization as _create_functional_map_visualization,
    create_color_pullback_visualization as _create_color_pullback_visualization,
    create_functional_map_heatmap as _create_functional_map_heatmap,
)

import os
# Prefer software rasterization when no GPU drivers/EGL are available (helps avoid Open3D segfaults)
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import torch
import open3d as o3d
import numpy as np
import argparse
import json


def create_functional_map_visualization(vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder):
    return _create_functional_map_visualization(
        vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder
    )


def create_color_pullback_visualization(vert_M, vert_N, triv_M, triv_N, matches, gt_matches, opts, output_folder):
    return _create_color_pullback_visualization(
        vert_M, vert_N, triv_M, triv_N, matches, gt_matches, opts, output_folder
    )


def run(mesh_data, output_folder, opts: Options, target_path = None):
    print('#'*60)
    print(f"Running `{mesh_data.name}` ...")
    print('#'*60)

    mesh_M = o3d.io.read_triangle_mesh(mesh_data.full_mesh)
    mesh_N = o3d.io.read_triangle_mesh(mesh_data.partial_mesh)

    vert_M, triv_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
    vert_N, triv_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)
    M = ManifoldMesh(vert_M, triv_M, opts, compute_geo=True)
    N = ManifoldMesh(vert_N, triv_N, opts, compute_geo=False)

    C, v, matches = match_and_refine(M, N, opts)
    C, v, matches = C.numpy(force=True), v.numpy(force=True), matches.numpy(force=True)

    gt_matches = None
    if mesh_data.ground_truth and os.path.exists(mesh_data.ground_truth):
        try:
            gt_matches = np.loadtxt(mesh_data.ground_truth, dtype=float).astype(int) - 1
        except Exception as e:
            print(f"Warning: could not read ground truth at {mesh_data.ground_truth}: {e}")

    geodesics_M = M.compute_geodesic_matrix()
    dist_method_geo = None
    mean_geodesic_error = float('nan')
    if gt_matches is not None:
        dist_method_geo = np.array([geodesics_M[gt_matches[i], matches[i]] for i in range(len(matches))])
        dist_method_geo = dist_method_geo / np.sqrt(M.area)
        mean_geodesic_error = dist_method_geo.mean()
        print(f"Mean geodesic error: {mean_geodesic_error:.6f}")
    else:
        print("No ground truth provided; skipping geodesic error computation.")

    if output_folder is None:
        return {
            'mean': float(mean_geodesic_error),
            'functional_map': None,
            'color_pullback': None,
            'output_folder': None,
            'descriptor': opts.descriptor_type,
            'matches': matches,
            'gt_matches': gt_matches,
        }

    os.makedirs(output_folder, exist_ok=True)
    
    # Save pointwise correspondences to correspondences_{descriptor}.vts
    correspondence_path = os.path.join(output_folder, f'correspondences_{opts.descriptor_type}.vts')
    np.savetxt(correspondence_path, matches.astype(int) + 1, fmt='%d')
    print(f"Wrote correspondences to {correspondence_path}")
    
    # Create visualizations using helper functions
    functional_map_path = create_functional_map_visualization(
        vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder
    )

    # Save functional map matrix C as a separate standalone heatmap PNG
    try:
        _create_functional_map_heatmap(C, opts, output_folder)
    except Exception as e:
        print(f"Warning: failed to save functional map heatmap: {e}")
    
    color_pullback_path = create_color_pullback_visualization(
        vert_M, vert_N, triv_M, triv_N, matches, gt_matches, opts, output_folder
    )
    
    print()
    print()

    # prepare relative paths for returned result
    try:
        functional_map_rel = os.path.relpath(functional_map_path, start=target_path)
        color_pullback_rel = os.path.relpath(color_pullback_path, start=target_path)
    except Exception:
        functional_map_rel = functional_map_path
        color_pullback_rel = color_pullback_path

    # return result dict (do not append to global here)
    return {
        'mean': float(mean_geodesic_error),
        'functional_map': functional_map_rel,
        'color_pullback': color_pullback_rel,
        'output_folder': output_folder,
        'descriptor': opts.descriptor_type,
        'matches': matches,
        'gt_matches': gt_matches,
    }

def write_summary_html(summary_results, target_path, descriptor_list, include_visuals: bool = True):
    """Write/overwrite the HTML meshes summary from `summary_results` into `target_path/meshes_summary.html`.
    This function is safe to call repeatedly (it overwrites the previous file).
    """
    os.makedirs(target_path, exist_ok=True)
    html_path = os.path.join(target_path, 'meshes_summary.html')
    rows = sorted(summary_results, key=lambda x: x['name'])

    # compute summary statistics for top summary table
    def _mean_key(d: str) -> str:
        return f"mean_{d}"

    def safe_mean(arr):
        if arr.size == 0:
            return float('nan')
        return float(np.mean(arr))

    overall_means = {
        d: safe_mean(np.array([r.get(_mean_key(d)) for r in rows], dtype=float)) if rows else float('nan')
        for d in descriptor_list
    }

    cuts_rows = [r for r in rows if r.get('folder') == 'cuts']
    holes_rows = [r for r in rows if r.get('folder') == 'holes']

    cuts_count = len(cuts_rows)
    holes_count = len(holes_rows)
    total_count = len(rows)

    cuts_means = {
        d: safe_mean(np.array([r.get(_mean_key(d)) for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
        for d in descriptor_list
    }
    holes_means = {
        d: safe_mean(np.array([r.get(_mean_key(d)) for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')
        for d in descriptor_list
    }

    html_lines = [
        '<!doctype html>',
        '<html>',
        '<head>',
        '<meta charset="utf-8" />',
        '<title>Meshes Summary</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; padding: 20px; }',
        'table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
        'th, td { border: 1px solid #ddd; padding: 8px; }',
        'th { background: #f4f4f4; text-align: left; }',
        'tr:nth-child(even) { background: #fbfbfb; }',
        '</style>',
        '</head>',
        '<body>',
        '<h1>Meshes Summary</h1>',
        '<h2>Dataset Statistics</h2>',
        '<table>',
        '<tr><th>Category</th><th>Count</th>' + ''.join([f'<th>Mean Geodesic Error ({d.upper()})</th>' for d in descriptor_list]) + '</tr>',
        f'<tr><td>Cuts</td><td>{cuts_count}</td>' + ''.join([f'<td>{cuts_means[d]:.6f}</td>' for d in descriptor_list]) + '</tr>',
        f'<tr><td>Holes</td><td>{holes_count}</td>' + ''.join([f'<td>{holes_means[d]:.6f}</td>' for d in descriptor_list]) + '</tr>',
        f'<tr><td>Entire dataset</td><td>{total_count}</td>' + ''.join([f'<td>{overall_means[d]:.6f}</td>' for d in descriptor_list]) + '</tr>',
        '</table>',
        '<hr/>',
        '<h2>Individual Mesh Results</h2>',
        '<table>',
        '<tr><th>Name</th><th>Best</th>'
        + ''.join([f'<th>Mean Geodesic Error ({d.upper()})</th>' for d in descriptor_list])
        + (''.join([f'<th>{d.upper()} Visualizations</th>' for d in descriptor_list]) if include_visuals else '')
        + '</tr>'
    ]

    for r in rows:
        means = {d: r.get(_mean_key(d)) if r.get(_mean_key(d)) is not None else float('nan') for d in descriptor_list}

        # Determine best descriptor (lowest error)
        descriptor_errors = {d.upper(): means[d] for d in descriptor_list}
        valid_errors = {k: v for k, v in descriptor_errors.items() if not np.isnan(v)}
        best_descriptor = min(valid_errors, key=valid_errors.get) if valid_errors else 'N/A'

        row_cells = [f'<td>{r["name"]}</td>', f'<td>{best_descriptor}</td>']
        row_cells.extend([f'<td>{means[d]:.6f}</td>' for d in descriptor_list])

        if include_visuals:
            vis_cells = []
            for d in descriptor_list:
                links = []
                fm_key = f'functional_map_{d}'
                cp_key = f'color_pullback_{d}'
                iv_key = f'interactive_view_{d}'
                if r.get(fm_key):
                    links.append(f'<a href="{r[fm_key]}" target="_blank">functional_map_visualization_{d}</a>')
                if r.get(cp_key):
                    links.append(f'<a href="{r[cp_key]}" target="_blank">color_pullback_{d}</a>')
                if r.get(iv_key):
                    links.append(f'<a href="{r[iv_key]}" target="_blank">interactive_view_{d}</a>')
                vis_cells.append(f'<td>{" | ".join(links) if links else ""}</td>')
            row_cells.extend(vis_cells)

        html_lines.append('<tr>' + ''.join(row_cells) + '</tr>')

    html_lines.extend(['</table>', '</body>', '</html>'])

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_lines))

    print(f"Wrote HTML summary to {html_path}")


def load_state(state_path):
    """Load persisted state.json if it exists, else return empty structure."""
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'processed_samples' in data:
                return data
        except Exception as e:
            print(f"Warning: could not read state file {state_path}: {e}")
    return {'processed_samples': {}}


def save_state(state, state_path):
    """Persist state to disk as JSON."""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    try:
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write state file {state_path}: {e}")

def main():
        
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Partial Functions Map - 3D shape matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Single-run with user-provided meshes
    python main.py --full-mesh /path/to/full.off --partial-mesh /path/to/partial.off --shot --target-path results

    # Single-run with GT
    python main.py --full-mesh /path/to/full.off --partial-mesh /path/to/partial.off --gt-path /path/to/corres.vts --target-path results

    # Dataset mode (uses defaults for SHREC16 directory layout)
    python main.py --fpfh --target-path results
        """
    )
    parser.add_argument(
        '--fpfh',
        action='store_true',
        help='Use FPFH descriptors (default)'
    )
    parser.add_argument(
        '--shot',
        action='store_true',
        help='Use SHOT descriptors'
    )
    parser.add_argument(
        '--dino',
        action='store_true',
        help='Use DINO descriptors (requires pytorch3d and internet to download model)'
    )
    parser.add_argument(
        '--dinov3', '--dino3',
        dest='dinov3',
        action='store_true',
        help='Use DINOv3 descriptors (requires pytorch3d, transformers and internet to download model)'
    )
    parser.add_argument(
        '--desc',
        action='append',
        help='Descriptor to run (can be provided multiple times, e.g. --desc shot --desc fpfh)'
    )
    parser.add_argument(
        '--target-path',
        type=str,
        default='results',
        help='Path to the output results directory (default: results)'
    )

    # Web viewer: generate HTML + JSON assets and link in summary
    parser.add_argument(
        '--web-view',
        action='store_true',
        help='Generate a web-based interactive viewer (HTML + JSON) in the result folder and link it in the summary'
    )

    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Skip generation of visualizations (functional map and color pullback images)'
    )

    # Single-run paths: user-provided full/partial mesh (optional GT)
    parser.add_argument(
        '--full-mesh',
        type=str,
        help='Path to the full mesh (.off) or a directory containing full meshes (auto-resolved from partial name)'
    )
    parser.add_argument(
        '--partial-mesh',
        type=str,
        help='Path to the partial mesh (.off) to match from'
    )
    parser.add_argument(
        '--gt-path',
        type=str,
        help='Optional path to ground-truth correspondences (.vts). If omitted, GT-based metrics/visualizations are skipped'
    )
    parser.add_argument(
        '--shrec16',
        type=str,
        help='Path to SHREC16 root (contains cuts/holes/null). Required for dataset mode.'
    )

    # Optional optimization/iteration overrides
    parser.add_argument(
        '--v-max-iter',
        dest='v_max_iter',
        type=int,
        help='Override Options.v_max_iter (default: 2000)'
    )
    parser.add_argument(
        '--C-max-iter',
        dest='C_max_iter',
        type=int,
        help='Override Options.C_max_iter (default: 2000)'
    )
    parser.add_argument(
        '--max-outer-iter',
        dest='max_outer_iter',
        type=int,
        help='Override Options.max_outer_iter (default: 7)'
    )
    parser.add_argument(
        '--refine-iters',
        dest='refine_iters',
        type=int,
        help='Override refinement iterations (default: same value as --max-outer-iter / Options.max_outer_iter)'
    )
    parser.add_argument(
        '--early-stopping',
        dest='early_stopping',
        action='store_true',
        help='Enable early stopping in C and v optimization steps'
    )

    # Dataset mode uses internal defaults, no CLI args needed

    args = parser.parse_args()

    # Viewer mode will be handled after matching to show post-match colors

    # Determine descriptor types to use (allow multiple)
    selected_types = []
    if args.fpfh:
        selected_types.append("fpfh")
    if args.shot:
        selected_types.append("shot")
    if args.dino:
        selected_types.append("dino")
    if getattr(args, 'dinov3', False):
        selected_types.append("dinov3")
    if args.desc:
        selected_types.extend(args.desc)
    # De-duplicate while preserving order
    _seen = set()
    selected_types = [t for t in selected_types if not (t in _seen or _seen.add(t))]
    if not selected_types:
        selected_types = ["fpfh"]

    target_path = args.target_path
    state_path = os.path.join(target_path, 'state.txt')

    print("Using descriptors: " + ", ".join([t.upper() for t in selected_types]))
    if args.full_mesh and args.partial_mesh:
        print(f"Single-run: full={args.full_mesh}, partial={args.partial_mesh}, gt={args.gt_path or 'None'}")
    else:
        if not args.shrec16:
            parser.error("Dataset mode requires --shrec16 <path> when --full-mesh/--partial-mesh are not provided.")
        print(f"Dataset mode: SHREC16 root={args.shrec16}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    opts = Options(device)
    # Apply user-provided iteration overrides if any
    if getattr(args, 'v_max_iter', None) is not None:
        opts.v_max_iter = args.v_max_iter
    if getattr(args, 'C_max_iter', None) is not None:
        opts.C_max_iter = args.C_max_iter
    if getattr(args, 'max_outer_iter', None) is not None:
        opts.max_outer_iter = args.max_outer_iter
    if getattr(args, 'refine_iters', None) is not None:
        opts.refine_iters = args.refine_iters
    else:
        # Default behavior: if --refine-iters is omitted, mirror max_outer_iter
        # (including user override via --max-outer-iter, if provided).
        opts.refine_iters = opts.max_outer_iter
    if getattr(args, 'early_stopping', False):
        opts.early_stopping = True
    
    # load persisted state (processed samples) and initialize summary_results from it
    state = load_state(state_path)
    processed_samples = state.get('processed_samples', {})
    summary_results = list(processed_samples.values())

    # If user provided explicit meshes, run a single job; otherwise iterate dataset
    if args.full_mesh and args.partial_mesh:
        single_name = os.path.splitext(os.path.basename(args.partial_mesh))[0]
        # Resolve full mesh: allow file or directory input
        _full_path = args.full_mesh
        if os.path.isdir(_full_path):
            parts = single_name.split('_')
            candidates = []
            if len(parts) >= 2:
                candidates.append('_'.join(parts[1:]) + '.off')  # e.g., horse_shape_14.off
                candidates.append(parts[1] + '.off')              # e.g., horse.off
            candidates.append(single_name + '.off')               # fallback: full same as partial basename
            found = None
            for c in candidates:
                cp = os.path.join(_full_path, c)
                if os.path.exists(cp):
                    found = cp
                    break
            if not found:
                raise FileNotFoundError(f"Could not resolve full mesh file under directory {_full_path} for partial {single_name} (tried: {', '.join(candidates)})")
            _full_path = found

        mesh_data = MeshPair(
            name=single_name,
            full_mesh=_full_path,
            partial_mesh=args.partial_mesh,
            ground_truth=(args.gt_path if args.gt_path else None)
        )
        result_path = os.path.join(target_path, single_name)

        entry = processed_samples.get(single_name)
        if entry is None:
            entry = {
                'name': single_name,
                'output_folder': result_path,
                'folder': 'single',
            }
            summary_results.append(entry)
        else:
            # Keep path/folder up to date
            entry['output_folder'] = result_path
            entry['folder'] = 'single'

        # Run only descriptors not yet present in state for this sample
        _types_to_run = [d for d in selected_types if entry.get(f'mean_{d}') is None]
        if not _types_to_run:
            print(f"Skipping {single_name}: requested descriptors already processed in state")
        else:
            _results = {}
            for _dt in _types_to_run:
                opts.descriptor_type = _dt
                _results[_dt] = run(
                    mesh_data,
                    None if args.no_vis else result_path,
                    opts,
                    target_path
                )

            for _dt in _types_to_run:
                res = _results.get(_dt) or {}
                entry[f'mean_{_dt}'] = res.get('mean')
                entry[f'functional_map_{_dt}'] = res.get('functional_map')
                entry[f'color_pullback_{_dt}'] = res.get('color_pullback')

            processed_samples[single_name] = entry
            state['processed_samples'] = processed_samples
            save_state(state, state_path)
            write_summary_html(summary_results, target_path, selected_types, include_visuals=(not args.no_vis))

            # If web viewer requested, generate HTML using the already-computed matches
            if args.web_view:
                for _dt in _types_to_run:
                    try:
                        viewer_res = _results.get(_dt)
                        if viewer_res is None:
                            raise RuntimeError(f"No results found for descriptor '{_dt}' to generate web view.")
                        matches_arr = viewer_res.get('matches')
                        gt_matches_arr = viewer_res.get('gt_matches')
                        if matches_arr is None:
                            raise RuntimeError("Matches are missing from run() results; cannot generate web view.")
                        html_path = generate_interactive_view(
                            mesh_data.full_mesh,
                            mesh_data.partial_mesh,
                            matches_arr,
                            gt_matches_arr,
                            result_path
                        )
                        try:
                            html_rel = os.path.relpath(html_path, start=target_path)
                        except Exception:
                            html_rel = html_path
                        key_name = f'interactive_view_{_dt}'
                        entry[key_name] = html_rel
                        processed_samples[single_name] = entry
                        state['processed_samples'] = processed_samples
                        save_state(state, state_path)
                        write_summary_html(summary_results, target_path, selected_types, include_visuals=(not args.no_vis))
                        print(f"Web viewer generated: {html_path}")
                    except Exception as e:
                        print(f"Web viewer generation failed (no recompute): {e}")

            # Popup viewer deprecated; web-based viewer available via --web-view
    else:
        def _infer_folder(path: str) -> str:
            parts = os.path.normpath(path).split(os.sep)
            if 'cuts' in parts:
                return 'cuts'
            if 'holes' in parts:
                return 'holes'
            return 'unknown'

        dataset = Shrec16(args.shrec16)
        for mesh_data in dataset:
            folder = _infer_folder(mesh_data.partial_mesh)
            result_path = os.path.join(target_path, folder, mesh_data.name)

            entry = processed_samples.get(mesh_data.name)
            if entry is None:
                entry = {
                    'name': mesh_data.name,
                    'output_folder': result_path,
                    'folder': folder,
                }
                summary_results.append(entry)
            else:
                entry['output_folder'] = result_path
                entry['folder'] = folder

            # Decide descriptors to run (only missing for this sample)
            _types_to_run = [d for d in selected_types if entry.get(f'mean_{d}') is None]
            if not _types_to_run:
                continue

            _results = {}
            for _dt in _types_to_run:
                opts.descriptor_type = _dt
                _results[_dt] = run(
                    mesh_data,
                    None if args.no_vis else result_path,
                    opts,
                    target_path
                )

            # aggregate into one summary entry (only fill those run)
            for _dt in _types_to_run:
                res = _results.get(_dt) or {}
                entry[f'mean_{_dt}'] = res.get('mean')
                entry[f'functional_map_{_dt}'] = res.get('functional_map')
                entry[f'color_pullback_{_dt}'] = res.get('color_pullback')

            processed_samples[mesh_data.name] = entry
            state['processed_samples'] = processed_samples
            # write incremental HTML summary after each processed mesh
            save_state(state, state_path)
            write_summary_html(summary_results, target_path, selected_types, include_visuals=(not args.no_vis))

        # In dataset mode, viewer is less defined; skipping popup to avoid repeated openings

if __name__ == "__main__":
    main()