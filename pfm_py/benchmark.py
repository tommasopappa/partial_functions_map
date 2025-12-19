"""
Benchmark script for partial functional maps on a representative sample.
Runs DINO, SHOT, FPFH and stores results including DINO missing features.
Computes:
  - Argmax (feature NN)
  - FM (standard functional maps from Diff3F)
  - PFM (partial functional maps pipeline)
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from dataclasses import dataclass
from scipy.sparse.csgraph import dijkstra

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options

# Try to import Diff3F's compute_surface_map
DIFF3F_AVAILABLE = False

# Add Diffusion-3D-Features to path
_diff3f_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Diffusion-3D-Features'),
    os.path.join(os.getcwd(), 'Diffusion-3D-Features'),
]
for _p in _diff3f_paths:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        print(f"Added {_p} to path")

try:
    from functional_map import compute_surface_map
    DIFF3F_AVAILABLE = True
    print("Diff3F functional_map available")
except ImportError as e:
    print(f"WARNING: Diff3F not found ({e}), FM results will be skipped")

@dataclass
class MeshPair:
    name: str
    full_mesh: str
    partial_mesh: str
    ground_truth: str
    folder: str


def get_representative_sample(data_path):
    """Select a representative sample: 2 cuts + 2 holes across different animals."""
    samples = [
        ("cuts", "cuts_cat_shape_1"),
        ("cuts", "cuts_dog_shape_3"),
        ("holes", "holes_cat_shape_10"),
        ("holes", "holes_horse_shape_5"),
    ]
    
    pairs = []
    for folder, name in samples:
        parts = name.split('_')
        animal = parts[1]
        pair = MeshPair(
            name=name,
            full_mesh=f"{data_path}/SHREC16/null/off/{animal}.off",
            partial_mesh=f"{data_path}/SHREC16/{folder}/off/{name}.off",
            ground_truth=f"{data_path}/SHREC16/{folder}/corres/{name}.vts",
            folder=folder,
        )
        if os.path.exists(pair.partial_mesh) and os.path.exists(pair.ground_truth):
            pairs.append(pair)
        else:
            print(f"Skipping {name} - files not found")
    return pairs


def compute_geodesic_matrix(vertices, faces):
    n = len(vertices)
    edges = set()
    for f in faces:
        edges.add(tuple(sorted([f[0], f[1]])))
        edges.add(tuple(sorted([f[1], f[2]])))
        edges.add(tuple(sorted([f[2], f[0]])))
    
    graph = np.full((n, n), np.inf)
    for i, j in edges:
        d = np.linalg.norm(vertices[i] - vertices[j])
        graph[i, j] = d
        graph[j, i] = d
    return dijkstra(graph, directed=False)


def find_boundary_edges(triangles):
    from collections import defaultdict
    edge_count = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[edge] += 1
    return [e for e, c in edge_count.items() if c == 1]


def compute_argmax_correspondences(desc_M, desc_N):
    """Simple nearest neighbor matching in feature space."""
    desc_M_norm = desc_M / (torch.norm(desc_M, dim=1, keepdim=True) + 1e-8)
    desc_N_norm = desc_N / (torch.norm(desc_N, dim=1, keepdim=True) + 1e-8)
    similarity = desc_N_norm @ desc_M_norm.T
    return torch.argmax(similarity, dim=1).cpu().numpy()


def icp_baseline(v_M, v_N, gt_correspondences, geo_dist_M, area_M):
    """ICP baseline for comparison."""
    import scipy.spatial
    
    pcd_M = o3d.geometry.PointCloud()
    pcd_M.points = o3d.utility.Vector3dVector(v_M)
    pcd_N = o3d.geometry.PointCloud()
    pcd_N.points = o3d.utility.Vector3dVector(v_N)
    
    pcd_M.estimate_normals()
    pcd_N.estimate_normals()
    
    radius = 0.05 * np.linalg.norm(v_M.max(0) - v_M.min(0))
    
    pcd_M_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_M, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    pcd_N_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_N, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_N, pcd_M, pcd_N_fpfh, pcd_M_fpfh, True, 0.05,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    icp = o3d.pipelines.registration.registration_icp(
        pcd_N, pcd_M, 0.02, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    v_N_aligned = (icp.transformation[:3, :3] @ v_N.T + icp.transformation[:3, 3:4]).T
    tree = scipy.spatial.cKDTree(v_M)
    _, icp_corr = tree.query(v_N_aligned)
    
    dist_icp = np.array([geo_dist_M[gt_correspondences[i], icp_corr[i]] / area_M
                         for i in range(len(v_N))])
    return icp_corr, dist_icp


def create_comparison_figure(v_M, v_N, f_M, f_N, gt_corr, argmax_corr, fm_corr, pfm_corr, icp_corr,
                              dist_argmax, dist_fm, dist_pfm, dist_icp, method_name, output_path):
    """5-row comparison: GT, Argmax, FM, PFM, ICP."""
    v_N_vis = v_N - v_N.mean(0)
    v_M_vis = v_M - v_M.mean(0)
    
    bbox_min, bbox_max = v_N_vis.min(0), v_N_vis.max(0)
    bbox_range = (bbox_max - bbox_min).max()
    bbox_center = (bbox_max + bbox_min) / 2
    lim = bbox_range / 2 + 0.1 * bbox_range
    
    boundary_N = find_boundary_edges(f_N)
    boundary_M = find_boundary_edges(f_M)
    
    v_N_norm = (v_N - v_N.min(0)) / (v_N.max(0) - v_N.min(0))
    source = v_N_norm[:, 0]
    
    colors_gt = np.zeros(len(v_M)); colors_gt[gt_corr] = source
    colors_argmax = np.zeros(len(v_M)); colors_argmax[argmax_corr] = source
    colors_fm = np.zeros(len(v_M)); colors_fm[fm_corr] = source if fm_corr is not None else 0
    colors_pfm = np.zeros(len(v_M)); colors_pfm[pfm_corr] = source
    colors_icp = np.zeros(len(v_M)); colors_icp[icp_corr] = source
    
    cmap_v = plt.get_cmap("viridis")
    cmap_e = plt.get_cmap("coolwarm")
    
    def face_colors(vert_colors, faces):
        return cmap_v(vert_colors)[faces].mean(axis=1)[:, :3]
    
    def error_colors(errors, faces, vmax):
        return cmap_e(np.clip(errors / vmax, 0, 1))[faces].mean(axis=1)[:, :3]
    
    poly_N = [v_N_vis[f] for f in f_N]
    poly_M = [v_M_vis[f] for f in f_M]
    
    fig = plt.figure(figsize=(18, 25))
    ls = mpl.colors.LightSource(azdeg=315, altdeg=45)
    
    def setup_ax(ax):
        ax.set_xlim([bbox_center[0]-lim, bbox_center[0]+lim])
        ax.set_ylim([bbox_center[1]-lim, bbox_center[1]+lim])
        ax.set_zlim([bbox_center[2]-lim, bbox_center[2]+lim])
        try: ax.set_box_aspect([1,1,1])
        except: pass
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
    
    def draw_boundary(ax, verts, edges):
        for e in edges:
            pts = verts[list(e)]
            ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', lw=1.5)
    
    dist_gt = np.zeros(len(v_N))
    vmax = max(np.percentile(dist_argmax, 95), 
               np.percentile(dist_fm, 95) if dist_fm is not None else 0,
               np.percentile(dist_pfm, 95), 
               np.percentile(dist_icp, 95), 0.1)
    
    rows = [
        ("GT", colors_gt, dist_gt, 0.1),
        (f"{method_name}", colors_argmax, dist_argmax, vmax),
        (f"{method_name}+FM", colors_fm, dist_fm if dist_fm is not None else dist_gt, vmax),
        (f"{method_name}+PFM", colors_pfm, dist_pfm, vmax),
        ("ICP", colors_icp, dist_icp, vmax),
    ]
    
    for row_idx, (label, colors_M, dist, err_vmax) in enumerate(rows):
        ax1 = fig.add_subplot(5, 3, row_idx*3 + 1, projection='3d')
        pc1 = Poly3DCollection(poly_N, facecolors=face_colors(source, f_N), 
                                linewidths=0, alpha=1.0, shade=True, lightsource=ls)
        ax1.add_collection3d(pc1)
        draw_boundary(ax1, v_N_vis, boundary_N)
        ax1.set_title("N: Source", fontweight='bold')
        setup_ax(ax1)
        
        ax2 = fig.add_subplot(5, 3, row_idx*3 + 2, projection='3d')
        pc2 = Poly3DCollection(poly_M, facecolors=face_colors(colors_M, f_M),
                                linewidths=0, alpha=1.0, shade=True, lightsource=ls)
        ax2.add_collection3d(pc2)
        draw_boundary(ax2, v_M_vis, boundary_M)
        ax2.set_title(f"{label} Transfer", fontweight='bold')
        setup_ax(ax2)
        
        ax3 = fig.add_subplot(5, 3, row_idx*3 + 3, projection='3d')
        pc3 = Poly3DCollection(poly_N, facecolors=error_colors(dist, f_N, err_vmax),
                                linewidths=0, alpha=1.0, shade=True, lightsource=ls)
        ax3.add_collection3d(pc3)
        draw_boundary(ax3, v_N_vis, boundary_N)
        ax3.set_title(f"{label} Error (mean={dist.mean():.4f})", fontweight='bold')
        setup_ax(ax3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def run_single(pair: MeshPair, opts: Options, output_dir: str):
    """Run all descriptors on a single mesh pair."""
    os.makedirs(output_dir, exist_ok=True)
    
    mesh_M = o3d.io.read_triangle_mesh(pair.full_mesh)
    mesh_N = o3d.io.read_triangle_mesh(pair.partial_mesh)
    
    v_M, f_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
    v_N, f_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)
    
    print(f"\n{'='*60}")
    print(f"Processing: {pair.name}")
    print(f"Full: {len(v_M)} verts, Partial: {len(v_N)} verts")
    print(f"{'='*60}")
    
    gt_corr = np.loadtxt(pair.ground_truth, dtype=float).astype(int) - 1
    
    print("Computing geodesic matrix...")
    geo_M = compute_geodesic_matrix(v_M, f_M)
    area_M = np.sqrt(0.5 * np.linalg.norm(
        np.cross(v_M[f_M[:,1]] - v_M[f_M[:,0]], v_M[f_M[:,2]] - v_M[f_M[:,0]]), axis=1).sum())
    
    print("Running ICP baseline...")
    icp_corr, dist_icp = icp_baseline(v_M, v_N, gt_corr, geo_M, area_M)
    
    results = {
        'name': pair.name,
        'folder': pair.folder,
        'n_verts_M': len(v_M),
        'n_verts_N': len(v_N),
        'icp_mean_error': float(dist_icp.mean()),
    }
    
    for desc_type in ['dino', 'shot', 'fpfh']:
        print(f"\n--- Running {desc_type.upper()} ---")
        opts.descriptor_type = desc_type
        
        # Build meshes and compute descriptors
        M = ManifoldMesh(v_M, f_M, opts, compute_geo=True)
        N = ManifoldMesh(v_N, f_N, opts, compute_geo=False)
        
        desc_M = M.compute_descriptors(opts)
        desc_N = N.compute_descriptors(opts)
        
        # Capture DINO missing features
        if desc_type == 'dino':
            results['dino_missing_M'] = getattr(M, 'dino_n_missing', 0)
            results['dino_missing_N'] = getattr(N, 'dino_n_missing', 0)
        
        # 1. Argmax correspondences (simple feature NN)
        print(f"Computing {desc_type.upper()} argmax correspondences...")
        argmax_corr = compute_argmax_correspondences(desc_M, desc_N)
        dist_argmax = np.array([geo_M[gt_corr[i], argmax_corr[i]] / area_M for i in range(len(v_N))])
        
        # 2. FM correspondences (standard functional maps from Diff3F)
        fm_corr = None
        dist_fm = None
        if DIFF3F_AVAILABLE:
            print(f"Computing {desc_type.upper()}+FM correspondences...")
            G_M = desc_M.cpu().numpy()
            F_N = desc_N.cpu().numpy()
            fm_corr = compute_surface_map(pair.full_mesh, pair.partial_mesh, G_M, F_N).cpu().numpy()
            dist_fm = np.array([geo_M[gt_corr[i], fm_corr[i]] / area_M for i in range(len(v_N))])
        else:
            print(f"Skipping {desc_type.upper()}+FM (Diff3F not available)")
        
        # 3. PFM correspondences (partial functional maps pipeline)
        print(f"Computing {desc_type.upper()}+PFM correspondences...")
        M2 = ManifoldMesh(v_M, f_M, opts, compute_geo=True)
        N2 = ManifoldMesh(v_N, f_N, opts, compute_geo=False)
        C, v, pfm_corr = match_and_refine(M2, N2, opts)
        pfm_corr = pfm_corr.numpy(force=True)
        dist_pfm = np.array([geo_M[gt_corr[i], pfm_corr[i]] / area_M for i in range(len(v_N))])
        
        # Store results
        results[f'{desc_type}_argmax_error'] = float(dist_argmax.mean())
        results[f'{desc_type}_fm_error'] = float(dist_fm.mean()) if dist_fm is not None else None
        results[f'{desc_type}_pfm_error'] = float(dist_pfm.mean())
        
        print(f"  {desc_type.upper()}:      {dist_argmax.mean():.4f}")
        if dist_fm is not None:
            print(f"  {desc_type.upper()}+FM:   {dist_fm.mean():.4f}")
        print(f"  {desc_type.upper()}+PFM:  {dist_pfm.mean():.4f}")
        
        # Create comparison figure
        fig_path = os.path.join(output_dir, f"{pair.name}_{desc_type}_comparison.png")
        create_comparison_figure(v_M, v_N, f_M, f_N, gt_corr, argmax_corr, fm_corr, pfm_corr, icp_corr,
                                  dist_argmax, dist_fm, dist_pfm, dist_icp, desc_type.upper(), fig_path)
        results[f'{desc_type}_figure'] = fig_path
    
    return results


def write_summary(all_results, output_dir):
    """Write JSON and markdown summary."""
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")
    
    md_path = os.path.join(output_dir, 'benchmark_results.md')
    
    # Check if FM results exist
    has_fm = any(r.get('dino_fm_error') is not None for r in all_results)
    
    if has_fm:
        header = "| Mesh | Folder | DINO | DINO+FM | DINO+PFM | SHOT | SHOT+FM | SHOT+PFM | FPFH | FPFH+FM | FPFH+PFM | ICP | DINO Missing (M/N) |"
        separator = "|------|--------|------|---------|----------|------|---------|----------|------|---------|----------|-----|-------------------|"
    else:
        header = "| Mesh | Folder | DINO | DINO+PFM | SHOT | SHOT+PFM | FPFH | FPFH+PFM | ICP | DINO Missing (M/N) |"
        separator = "|------|--------|------|----------|------|----------|------|----------|-----|-------------------|"
    
    lines = [
        "# Benchmark Results",
        "",
        "## Summary Table",
        "",
        header,
        separator,
    ]
    
    for r in all_results:
        dino_miss = f"{r.get('dino_missing_M', '-')}/{r.get('dino_missing_N', '-')}"
        
        if has_fm:
            dino_fm = f"{r.get('dino_fm_error', 0):.4f}" if r.get('dino_fm_error') is not None else "-"
            shot_fm = f"{r.get('shot_fm_error', 0):.4f}" if r.get('shot_fm_error') is not None else "-"
            fpfh_fm = f"{r.get('fpfh_fm_error', 0):.4f}" if r.get('fpfh_fm_error') is not None else "-"
            lines.append(
                f"| {r['name']} | {r['folder']} | "
                f"{r.get('dino_argmax_error', 0):.4f} | {dino_fm} | {r.get('dino_pfm_error', 0):.4f} | "
                f"{r.get('shot_argmax_error', 0):.4f} | {shot_fm} | {r.get('shot_pfm_error', 0):.4f} | "
                f"{r.get('fpfh_argmax_error', 0):.4f} | {fpfh_fm} | {r.get('fpfh_pfm_error', 0):.4f} | "
                f"{r.get('icp_mean_error', 0):.4f} | {dino_miss} |"
            )
        else:
            lines.append(
                f"| {r['name']} | {r['folder']} | "
                f"{r.get('dino_argmax_error', 0):.4f} | {r.get('dino_pfm_error', 0):.4f} | "
                f"{r.get('shot_argmax_error', 0):.4f} | {r.get('shot_pfm_error', 0):.4f} | "
                f"{r.get('fpfh_argmax_error', 0):.4f} | {r.get('fpfh_pfm_error', 0):.4f} | "
                f"{r.get('icp_mean_error', 0):.4f} | {dino_miss} |"
            )
    
    # Averages
    n = len(all_results)
    if n > 0:
        def avg(key): 
            vals = [r.get(key) for r in all_results if r.get(key) is not None]
            return np.mean(vals) if vals else 0
        
        if has_fm:
            lines.append(
                f"| **Average** | - | "
                f"**{avg('dino_argmax_error'):.4f}** | **{avg('dino_fm_error'):.4f}** | **{avg('dino_pfm_error'):.4f}** | "
                f"**{avg('shot_argmax_error'):.4f}** | **{avg('shot_fm_error'):.4f}** | **{avg('shot_pfm_error'):.4f}** | "
                f"**{avg('fpfh_argmax_error'):.4f}** | **{avg('fpfh_fm_error'):.4f}** | **{avg('fpfh_pfm_error'):.4f}** | "
                f"**{avg('icp_mean_error'):.4f}** | - |"
            )
        else:
            lines.append(
                f"| **Average** | - | "
                f"**{avg('dino_argmax_error'):.4f}** | **{avg('dino_pfm_error'):.4f}** | "
                f"**{avg('shot_argmax_error'):.4f}** | **{avg('shot_pfm_error'):.4f}** | "
                f"**{avg('fpfh_argmax_error'):.4f}** | **{avg('fpfh_pfm_error'):.4f}** | "
                f"**{avg('icp_mean_error'):.4f}** | - |"
            )
    
    lines.extend([
        "",
        "## Methods",
        "",
        "- **DINO/SHOT/FPFH**: Simple nearest neighbor matching in feature space (argmax cosine similarity)",
        "- **+FM**: Standard Functional Maps (from Diffusion-3D-Features)",
        "- **+PFM**: Partial Functional Maps pipeline (C optimization + spectral ICP refinement)",
        "- **ICP**: Iterative Closest Point baseline",
        "",
        "## Notes",
        "",
        "- All errors are mean geodesic error normalized by sqrt(area)",
        "- DINO Missing shows vertices without feature coverage on M/N",
    ])
    
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {md_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark PFM on representative sample')
    parser.add_argument('--data-path', type=str, default='/usr/prakt/w0010/SAVHA/shape_data')
    parser.add_argument('--output', type=str, default='benchmark_results')
    parser.add_argument('--diff3f-path', type=str, default=None,
                        help='Path to Diffusion-3D-Features repo')
    args = parser.parse_args()
    
    # Add Diff3F to path if specified
    if args.diff3f_path and os.path.isdir(args.diff3f_path):
        sys.path.insert(0, args.diff3f_path)
        try:
            from functional_map import compute_surface_map
            DIFF3F_AVAILABLE = True
            print(f"Loaded Diff3F from {args.diff3f_path}")
        except ImportError:
            pass
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Diff3F available: {DIFF3F_AVAILABLE}")
    
    opts = Options(device)
    pairs = get_representative_sample(args.data_path)
    
    if not pairs:
        print("No valid mesh pairs found!")
        exit(1)
    
    print(f"Running benchmark on {len(pairs)} mesh pairs")
    
    all_results = []
    for pair in pairs:
        output_dir = os.path.join(args.output, pair.folder, pair.name)
        result = run_single(pair, opts, output_dir)
        all_results.append(result)
    
    write_summary(all_results, args.output)
    print("\nBenchmark complete!")
