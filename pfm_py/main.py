from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options
from pfm_py.web_viewer import generate_interactive_view

import torch
import open3d as o3d
import numpy as np
import os
import argparse
import json

from dataclasses import dataclass

@dataclass
class TestMeshData:
    name: str
    full_mesh: str
    partial_mesh: str
    ground_truth: str

def create_functional_map_visualization(vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder):
    """Create and save the functional map visualization showing source function, ground truth transfer, and method transfer."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib as mpl

    def create_full_colormap(n):
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 1, n))[:, :3]
        return colors
    
    def find_boundary_edges(triangles):
        """Find boundary edges (edges that appear only once in the mesh)."""
        from collections import defaultdict
        edge_count = defaultdict(int)
        for tri in triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                edge_count[edge] += 1
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    # Center and downscale meshes using a shared centroid and common scale
    all_verts = np.vstack([vert_M, vert_N])
    shared_center = all_verts.mean(0)
    ranges = all_verts.max(0) - all_verts.min(0)
    max_range = float(ranges.max())
    scale = (1.0 / max_range) if max_range > 1e-12 else 1.0
    v_N_vis = (vert_N - shared_center) * scale
    v_M_vis = (vert_M - shared_center) * scale

    # Find boundary edges
    boundary_edges_N = find_boundary_edges(triv_N)
    boundary_edges_M = find_boundary_edges(triv_M)

    def set_axes_clean(ax):
        # Fit perfectly into a fixed cube without coordinate systems
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_axis_off()

    # --- Paper colormap on N and push-forward RGB via the functional map ---
    def create_paper_colormap(verts_A: np.ndarray, verts_B: np.ndarray) -> np.ndarray:
        """RGB = normalized XYZ of A using min/max computed on B (create_colormap(B,B))."""
        mins = verts_B.min(axis=0)
        maxs = verts_B.max(axis=0)
        denom = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
        return (verts_A - mins) / denom

    # Source colors on N (create_colormap(N,N))
    colors_N = np.clip(create_paper_colormap(vert_N, vert_N), 0.0, 1.0)

    # Ground truth RGB transfer to M (vertex-wise assignment)
    has_gt = gt_matches is not None and len(gt_matches) == vert_N.shape[0]
    facecols_gt = None
    if has_gt:
        colors_M_gt = np.zeros((vert_M.shape[0], 3), dtype=float)
        colors_M_gt[gt_matches] = colors_N
        facecols_gt = colors_M_gt[triv_M].mean(axis=1)

    # Push-forward each RGB component via C: a_M = C (E_N^T S_N f_N), then f_M = E_M a_M
    evecs_N_T = N.evecs.T.numpy(force=True)
    mass_N = N.S.numpy(force=True)
    evecs_M = M.evecs.numpy(force=True)
    colors_M_method = np.zeros((vert_M.shape[0], 3), dtype=float)
    for ch in range(3):
        fN = colors_N[:, ch]
        aN = evecs_N_T @ (mass_N * fN)
        aM = C @ aN
        fM = evecs_M @ aM
        # normalize per-channel for visualization stability
        fM = (fM - fM.min()) / (fM.max() - fM.min() + 1e-10)
        colors_M_method[:, ch] = fM

    # Prepare mesh polygons and face colors
    poly_N = [v_N_vis[f] for f in triv_N]
    poly_M = [v_M_vis[f] for f in triv_M]

    # Map vertex colors to face colors (average of vertex colors)
    cmap_viridis = plt.get_cmap("viridis")
    cmap_coolwarm = plt.get_cmap("coolwarm")
    cmap_gray = plt.get_cmap("gray")

    facecols_source = colors_N[triv_N].mean(axis=1)
    facecols_method = colors_M_method[triv_M].mean(axis=1)
    
    facecols_method_error = None
    error_cb_cfg = None
    if dist_method_geo is not None and dist_method_geo.size > 0:
        vmax_error = np.percentile(dist_method_geo, 95)
        dist_method_geo_norm = np.clip(dist_method_geo / vmax_error, 0, 1)
        dist_method_geo_colors = cmap_coolwarm(dist_method_geo_norm)[:, :3]
        facecols_method_error = dist_method_geo_colors[triv_N].mean(axis=1)
        error_cb_cfg = (cmap_coolwarm, plt.Normalize(vmin=0, vmax=vmax_error))

    # Soft membership function v visualization (grayscale, bright=1 dark=0)
    v_cb_cfg = None
    facecols_v = None
    if v is not None:
        try:
            v_arr = np.asarray(v, dtype=float).reshape(-1)
            if v_arr.size == vert_N.shape[0]:
                vnorm = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min() + 1e-12)
                vcols = cmap_gray(vnorm)[:, :3]
                facecols_v = vcols[triv_N].mean(axis=1)
                v_cb_cfg = (cmap_gray, plt.Normalize(vmin=0, vmax=1))
            elif v_arr.size == vert_M.shape[0]:
                vnorm = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min() + 1e-12)
                vcols = cmap_gray(vnorm)[:, :3]
                facecols_v = vcols[triv_M].mean(axis=1)
                v_cb_cfg = (cmap_gray, plt.Normalize(vmin=0, vmax=1))
        except Exception:
            facecols_v = None
            v_cb_cfg = None

    # --- Build dynamic panel list ---
    panels = []
    panels.append(("N: Paper Colormap (RGB)", poly_N, facecols_source, boundary_edges_N, v_N_vis, None))
    if has_gt and facecols_gt is not None:
        panels.append(("GROUND TRUTH Transfer", poly_M, facecols_gt, boundary_edges_M, v_M_vis, None))
    panels.append(("METHOD Push-forward (RGB)", poly_M, facecols_method, boundary_edges_M, v_M_vis, None))
    if facecols_method_error is not None and error_cb_cfg is not None:
        panels.append((f"Method Error (mean = {dist_method_geo.mean():.4f})", poly_N, facecols_method_error, boundary_edges_N, v_N_vis, error_cb_cfg))
    if facecols_v is not None and v_cb_cfg is not None:
        # Decide which geometry to show for v based on its size
        if v_cb_cfg and v_arr.size == vert_N.shape[0]:
            panels.append(("Soft Membership v (N)", poly_N, facecols_v, boundary_edges_N, v_N_vis, v_cb_cfg))
        elif v_cb_cfg and v_arr.size == vert_M.shape[0]:
            panels.append(("Soft Membership v (M)", poly_M, facecols_v, boundary_edges_M, v_M_vis, v_cb_cfg))

    # Layout: up to 3 columns per row
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    boundary_line_width = 0
    opacity = 1.0
    # Render panels
    for idx, (title, polys, facecols, boundary_edges, verts_vis, cb_cfg) in enumerate(panels, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
        pc = Poly3DCollection(polys, facecolors=facecols, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
        ax.add_collection3d(pc)
        for edge in boundary_edges:
            pts = verts_vis[list(edge)]
            ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
        ax.set_title(title)
        set_axes_clean(ax)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        # Add colorbar when configuration is provided (error or membership)
        if cb_cfg is not None:
            cmap_cb, norm_cb = cb_cfg
            sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.6)

    plt.tight_layout()
    functional_map_fname = f"functional_map_visualization_{opts.descriptor_type}.png"
    functional_map_path = os.path.join(output_folder, functional_map_fname)
    plt.savefig(functional_map_path, dpi=300)
    print(f"Saved visualization to {functional_map_path}")
    
    return functional_map_path


def create_color_pullback_visualization(vert_M, vert_N, triv_M, triv_N, matches, gt_matches, opts, output_folder):
    """Create and save the color pullback visualization showing full and partial meshes with method and ground truth pullbacks."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib as mpl

    def find_boundary_edges(triangles):
        """Find boundary edges (edges that appear only once in the mesh). Kept for optional outlines (not used)."""
        from collections import defaultdict
        edge_count = defaultdict(int)
        for tri in triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                edge_count[edge] += 1
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    def _build_vertex_adjacency(num_verts: int, faces: np.ndarray):
        """Build vertex adjacency list from triangle faces."""
        neighbors = [set() for _ in range(num_verts)]
        f_int = faces.astype(int)
        for f in f_int:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))
        return [list(s) for s in neighbors]

    def _smooth_vertex_colors(faces: np.ndarray, colors: np.ndarray, iters: int = 2, alpha: float = 0.5) -> np.ndarray:
        """Laplacian-like smoothing for vertex colors using neighbor averaging.
        - faces: (m,3) triangle indices
        - colors: (n,3) RGB in [0,1]
        - iters: number of smoothing iterations
        - alpha: blend factor toward neighbor mean (0=no change, 1=replace)
        """
        n = int(colors.shape[0])
        adj = _build_vertex_adjacency(n, faces)
        cur = colors.astype(float).copy()
        for _ in range(max(0, int(iters))):
            nxt = cur.copy()
            for i, nbrs in enumerate(adj):
                if not nbrs:
                    continue
                avg = cur[nbrs].mean(axis=0)
                nxt[i] = (1.0 - alpha) * cur[i] + alpha * avg
            cur = np.clip(nxt, 0.0, 1.0)
        return cur

    # Shared centering and uniform scaling so both meshes fit neatly
    all_verts = np.vstack([vert_M, vert_N])
    shared_center = all_verts.mean(0)
    ranges = all_verts.max(0) - all_verts.min(0)
    max_range = float(ranges.max())
    scale = (1.0 / max_range) if max_range > 1e-12 else 1.0
    v_M_vis = (vert_M - shared_center) * scale
    v_N_vis = (vert_N - shared_center) * scale

    # Paper colormap: per-vertex RGB = normalized XYZ of M using joint min/max over M and N
    def create_paper_colormap(vert_M_arr, vert_N_arr):
        mins = np.minimum(vert_M_arr.min(axis=0), vert_N_arr.min(axis=0))
        maxs = np.maximum(vert_M_arr.max(axis=0), vert_N_arr.max(axis=0))
        denom = maxs - mins
        denom = np.where(denom > 1e-12, denom, 1.0)
        return (vert_M_arr - mins) / denom
    colors_M = create_paper_colormap(vert_M, vert_N)
    # Slightly dim base colors to improve specular visibility
    colors_M = np.clip(colors_M * 0.90, 0.0, 1.0)
    # Transfer to N via matches (method pullback)
    colors_N_method = colors_M[matches]
    has_gt = gt_matches is not None and len(gt_matches) == vert_N.shape[0]
    colors_N_gt = None
    if has_gt:
        colors_N_gt = colors_M[gt_matches]

    # Smooth vertex colors (no triangle subdivision) for M and N
    colors_M_s = _smooth_vertex_colors(triv_M, colors_M, iters=2, alpha=0.5)
    colors_N_method_s = _smooth_vertex_colors(triv_N, colors_N_method, iters=2, alpha=0.5)

    # compute face polygons and per-face colors (average vertex colors per face) on original meshes
    poly_M = [v_M_vis[f] for f in triv_M]
    facecols_M = colors_M_s[triv_M].mean(axis=1)

    poly_N_method = [v_N_vis[f] for f in triv_N]
    facecols_N_method = colors_N_method_s[triv_N].mean(axis=1)
    facecols_N_gt = None
    if has_gt and colors_N_gt is not None:
        colors_N_gt_s = _smooth_vertex_colors(triv_N, colors_N_gt, iters=2, alpha=0.5)
        poly_N_gt = [v_N_vis[f] for f in triv_N]
        facecols_N_gt = colors_N_gt_s[triv_N].mean(axis=1)

    # Specular + diffuse shading per face (Blinn-Phong approximation)
    def compute_face_normals(verts_vis, triangles):
        normals = []
        for f in triangles:
            a, b, c = verts_vis[f]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 1.0])
            normals.append(n)
        return np.array(normals)

    light_dir = np.array([0.577, 0.577, 0.577])  # normalized diagonal light
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0])
    view_dir = view_dir / np.linalg.norm(view_dir)
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec)
    shininess = 128.0
    ambient = 0.25
    kd = 0.85
    ks = 0.80

    normals_M = compute_face_normals(v_M_vis, triv_M)
    normals_Nm = compute_face_normals(v_N_vis, triv_N)
    normals_Ng = None
    if has_gt and colors_N_gt is not None:
        normals_Ng = compute_face_normals(v_N_vis, triv_N)

    diffuse_M = np.maximum(normals_M @ light_dir, 0.0)
    spec_M = np.maximum(normals_M @ half_vec, 0.0) ** shininess
    shading_M = ambient + kd * diffuse_M[:, None]
    facecols_M_shaded = np.clip(facecols_M * shading_M + ks * spec_M[:, None], 0.0, 1.0)

    diffuse_Nm = np.maximum(normals_Nm @ light_dir, 0.0)
    spec_Nm = np.maximum(normals_Nm @ half_vec, 0.0) ** shininess
    shading_Nm = ambient + kd * diffuse_Nm[:, None]
    facecols_N_method_shaded = np.clip(facecols_N_method * shading_Nm + ks * spec_Nm[:, None], 0.0, 1.0)
    facecols_N_gt_shaded = None
    if facecols_N_gt is not None and normals_Ng is not None:
        diffuse_Ng = np.maximum(normals_Ng @ light_dir, 0.0)
        spec_Ng = np.maximum(normals_Ng @ half_vec, 0.0) ** shininess
        shading_Ng = ambient + kd * diffuse_Ng[:, None]
        facecols_N_gt_shaded = np.clip(facecols_N_gt * shading_Ng + ks * spec_Ng[:, None], 0.0, 1.0)

    def set_axes_clean(ax):
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_axis_off()

    # create figure: left = full mesh, middle = GT pullback, right = method pullback
    fig = plt.figure(figsize=(24, 9))
    boundary_line_width = 0
    opacity = 0.85

    # Full mesh (continuous)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    pc1 = Poly3DCollection(poly_M, facecolors=facecols_M_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=False)
    ax1.add_collection3d(pc1)
    # No boundary edges to keep surfaces clean
    ax1.set_title("Full Mesh (M)\nsmooth colors", pad=20)
    set_axes_clean(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(False)

    # Partial mesh with ground truth pullback
    if facecols_N_gt is not None:
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        pc2 = Poly3DCollection(poly_N_gt, facecolors=facecols_N_gt_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=False)
        ax2.add_collection3d(pc2)
        # No boundary edges to keep surfaces clean
        ax2.set_title("Partial Mesh (N)\nGround Truth Pullback", pad=20)
        set_axes_clean(ax2)
        ax2.view_init(elev=20, azim=45)
        ax2.grid(False)

    # Partial mesh with method pullback
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    pc3 = Poly3DCollection(poly_N_method, facecolors=facecols_N_method_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=False)
    ax3.add_collection3d(pc3)
    # No boundary edges to keep surfaces clean
    ax3.set_title("Partial Mesh (N)\nMethod Pullback", pad=20)
    set_axes_clean(ax3)
    ax3.view_init(elev=20, azim=45)
    ax3.grid(False)

    plt.tight_layout(pad=2.0)
    color_pullback_fname = f"color_pullback_{opts.descriptor_type}.png"
    color_pullback_path = os.path.join(output_folder, color_pullback_fname)
    plt.savefig(color_pullback_path, dpi=300)
    print(f"Saved: {color_pullback_path}")
    
    return color_pullback_path


def run(mesh_data, output_folder, opts: Options, target_path):
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
        return mean_geodesic_error

    os.makedirs(output_folder, exist_ok=True)
    # Create visualizations using helper functions
    functional_map_path = create_functional_map_visualization(
        vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder
    )
    
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

def write_summary_html(summary_results, target_path):
    """Write/overwrite the HTML meshes summary from `summary_results` into `target_path/meshes_summary.html`.
    This function is safe to call repeatedly (it overwrites the previous file).
    """
    os.makedirs(target_path, exist_ok=True)
    html_path = os.path.join(target_path, 'meshes_summary.html')
    rows = sorted(summary_results, key=lambda x: x['name'])

    # compute summary statistics for top summary table
    dino_vals = np.array([r.get('mean_dino') for r in rows], dtype=float) if rows else np.array([], dtype=float)
    dinov3_vals = np.array([r.get('mean_dinov3') for r in rows], dtype=float) if rows else np.array([], dtype=float)
    shot_vals = np.array([r.get('mean_shot') for r in rows], dtype=float) if rows else np.array([], dtype=float)
    fpfh_vals = np.array([r.get('mean_fpfh') for r in rows], dtype=float) if rows else np.array([], dtype=float)

    def safe_mean(arr):
        if arr.size == 0:
            return float('nan')
        return float(np.mean(arr))

    overall_dino = safe_mean(dino_vals)
    overall_dinov3 = safe_mean(dinov3_vals)
    overall_shot = safe_mean(shot_vals)
    overall_fpfh = safe_mean(fpfh_vals)

    cuts_rows = [r for r in rows if r.get('folder') == 'cuts']
    holes_rows = [r for r in rows if r.get('folder') == 'holes']

    cuts_count = len(cuts_rows)
    holes_count = len(holes_rows)
    total_count = len(rows)

    cuts_dino = safe_mean(np.array([r.get('mean_dino') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    cuts_dinov3 = safe_mean(np.array([r.get('mean_dinov3') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    cuts_shot = safe_mean(np.array([r.get('mean_shot') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    cuts_fpfh = safe_mean(np.array([r.get('mean_fpfh') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    holes_dino = safe_mean(np.array([r.get('mean_dino') for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')
    holes_dinov3 = safe_mean(np.array([r.get('mean_dinov3') for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')
    holes_shot = safe_mean(np.array([r.get('mean_shot') for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')
    holes_fpfh = safe_mean(np.array([r.get('mean_fpfh') for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')

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
        '<tr><th>Category</th><th>Count</th><th>Mean Geodesic Error (DINO)</th><th>Mean Geodesic Error (DINOv3)</th><th>Mean Geodesic Error (SHOT)</th><th>Mean Geodesic Error (FPFH)</th></tr>',
        f'<tr><td>Cuts</td><td>{cuts_count}</td><td>{cuts_dino:.6f}</td><td>{cuts_dinov3:.6f}</td><td>{cuts_shot:.6f}</td><td>{cuts_fpfh:.6f}</td></tr>',
        f'<tr><td>Holes</td><td>{holes_count}</td><td>{holes_dino:.6f}</td><td>{holes_dinov3:.6f}</td><td>{holes_shot:.6f}</td><td>{holes_fpfh:.6f}</td></tr>',
        f'<tr><td>Entire dataset</td><td>{total_count}</td><td>{overall_dino:.6f}</td><td>{overall_dinov3:.6f}</td><td>{overall_shot:.6f}</td><td>{overall_fpfh:.6f}</td></tr>',
        '</table>',
        '<hr/>',
        '<h2>Individual Mesh Results</h2>',
        '<table>',
        '<tr><th>Name</th><th>Best</th><th>Mean Geodesic Error (DINO)</th><th>Mean Geodesic Error (DINOv3)</th><th>Mean Geodesic Error (SHOT)</th><th>Mean Geodesic Error (FPFH)</th><th>DINO Visualizations</th><th>DINOv3 Visualizations</th><th>SHOT Visualizations</th><th>FPFH Visualizations</th></tr>'
    ]

    for r in rows:
        dino_links = []
        dinov3_links = []
        shot_links = []
        fpfh_links = []
        if r.get('functional_map_dino'):
            dino_links.append(f'<a href="{r["functional_map_dino"]}" target="_blank">functional_map_visualization_dino</a>')
        if r.get('color_pullback_dino'):
            dino_links.append(f'<a href="{r["color_pullback_dino"]}" target="_blank">color_pullback_dino</a>')
        if r.get('functional_map_dinov3'):
            dinov3_links.append(f'<a href="{r["functional_map_dinov3"]}" target="_blank">functional_map_visualization_dinov3</a>')
        if r.get('color_pullback_dinov3'):
            dinov3_links.append(f'<a href="{r["color_pullback_dinov3"]}" target="_blank">color_pullback_dinov3</a>')
        if r.get('functional_map_shot'):
            shot_links.append(f'<a href="{r["functional_map_shot"]}" target="_blank">functional_map_visualization_shot</a>')
        if r.get('color_pullback_shot'):
            shot_links.append(f'<a href="{r["color_pullback_shot"]}" target="_blank">color_pullback_shot</a>')
        if r.get('interactive_view_shot'):
            shot_links.append(f'<a href="{r["interactive_view_shot"]}" target="_blank">interactive_view_shot</a>')
        if r.get('functional_map_fpfh'):
            fpfh_links.append(f'<a href="{r["functional_map_fpfh"]}" target="_blank">functional_map_visualization_fpfh</a>')
        if r.get('color_pullback_fpfh'):
            fpfh_links.append(f'<a href="{r["color_pullback_fpfh"]}" target="_blank">color_pullback_fpfh</a>')
        if r.get('interactive_view_fpfh'):
            fpfh_links.append(f'<a href="{r["interactive_view_fpfh"]}" target="_blank">interactive_view_fpfh</a>')
        if r.get('interactive_view_dino'):
            dino_links.append(f'<a href="{r["interactive_view_dino"]}" target="_blank">interactive_view_dino</a>')
        if r.get('interactive_view_dinov3'):
            dinov3_links.append(f'<a href="{r["interactive_view_dinov3"]}" target="_blank">interactive_view_dinov3</a>')

        dino_html = ' | '.join(dino_links) if dino_links else ''
        dinov3_html = ' | '.join(dinov3_links) if dinov3_links else ''
        shot_html = ' | '.join(shot_links) if shot_links else ''
        fpfh_html = ' | '.join(fpfh_links) if fpfh_links else ''

        mean_dino = r.get('mean_dino') if r.get('mean_dino') is not None else float('nan')
        mean_dinov3 = r.get('mean_dinov3') if r.get('mean_dinov3') is not None else float('nan')
        mean_shot = r.get('mean_shot') if r.get('mean_shot') is not None else float('nan')
        mean_fpfh = r.get('mean_fpfh') if r.get('mean_fpfh') is not None else float('nan')

        # Determine best descriptor (lowest error)
        descriptor_errors = {
            'DINO': mean_dino,
            'DINOv3': mean_dinov3,
            'SHOT': mean_shot,
            'FPFH': mean_fpfh
        }
        valid_errors = {k: v for k, v in descriptor_errors.items() if not np.isnan(v)}
        best_descriptor = min(valid_errors, key=valid_errors.get) if valid_errors else 'N/A'

        html_lines.append(f'<tr><td>{r["name"]}</td><td>{best_descriptor}</td><td>{mean_dino:.6f}</td><td>{mean_dinov3:.6f}</td><td>{mean_shot:.6f}</td><td>{mean_fpfh:.6f}</td><td>{dino_html}</td><td>{dinov3_html}</td><td>{shot_html}</td><td>{fpfh_html}</td></tr>')

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

    # Single-run with GT and benchmark all descriptors
    python main.py --full-mesh /path/to/full.off --partial-mesh /path/to/partial.off --gt-path /path/to/corres.vts --benchmark --target-path results

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
        '--target-path',
        type=str,
        default='results',
        help='Path to the output results directory (default: results)'
    )

    # (Removed) Popup viewer flag was deprecated in favor of web-based viewer

    # Web viewer: generate HTML + JSON assets and link in summary
    parser.add_argument(
        '--web-view',
        action='store_true',
        help='Generate a web-based interactive viewer (HTML + JSON) in the result folder and link it in the summary'
    )

    # Benchmark mode: run all descriptors for comparison
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run all descriptors (DINO, DINOv3, SHOT, FPFH) and write comparative summary'
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

    # Dataset mode uses internal defaults, no CLI args needed

    args = parser.parse_args()

    # Viewer mode will be handled after matching to show post-match colors

    # Determine the descriptor type to use
    descriptor_type = "fpfh"  # Default value
    if args.shot:
        descriptor_type = "shot"
    elif args.fpfh:
        descriptor_type = "fpfh"
    elif args.dino:
        descriptor_type = "dino"
    elif getattr(args, 'dinov3', False):
        descriptor_type = "dinov3"

    target_path = args.target_path
    state_path = os.path.join(target_path, 'state.txt')

    print(f"Using descriptor: {descriptor_type.upper()}")
    # Defaults for dataset mode (used when single-run paths are not provided)
    FULL_MESH_DIR_DEFAULT = '/usr/prakt/w0010/SAVHA/shape_data/SHREC16/null/off'
    PARTIAL_ROOT_DIR_DEFAULT = '/usr/prakt/w0010/SAVHA/shape_data/SHREC16'

    if args.full_mesh and args.partial_mesh:
        print(f"Single-run: full={args.full_mesh}, partial={args.partial_mesh}, gt={args.gt_path or 'None'}")
    else:
        print(f"Dataset defaults in use: full_dir={FULL_MESH_DIR_DEFAULT}, partial_root={PARTIAL_ROOT_DIR_DEFAULT}")

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

        mesh_data = TestMeshData(
            name=single_name,
            full_mesh=_full_path,
            partial_mesh=args.partial_mesh,
            ground_truth=(args.gt_path if args.gt_path else None)
        )
        result_path = os.path.join(target_path, single_name)

        if single_name in processed_samples:
            print(f"Skipping {single_name}: already processed (state)")
        else:
            if args.benchmark:
                _types_to_run = ['dino', 'dinov3', 'shot', 'fpfh']
            else:
                _types_to_run = [descriptor_type]

            _results = {}
            for _dt in _types_to_run:
                opts.descriptor_type = _dt
                _results[_dt] = run(mesh_data, result_path, opts, target_path)

            entry = {
                'name': single_name,
                'mean_dino': (_results.get('dino') or {}).get('mean'),
                'mean_dinov3': (_results.get('dinov3') or {}).get('mean'),
                'mean_shot': (_results.get('shot') or {}).get('mean'),
                'mean_fpfh': (_results.get('fpfh') or {}).get('mean'),
                'functional_map_dino': (_results.get('dino') or {}).get('functional_map'),
                'color_pullback_dino': (_results.get('dino') or {}).get('color_pullback'),
                'functional_map_dinov3': (_results.get('dinov3') or {}).get('functional_map'),
                'color_pullback_dinov3': (_results.get('dinov3') or {}).get('color_pullback'),
                'functional_map_shot': (_results.get('shot') or {}).get('functional_map'),
                'color_pullback_shot': (_results.get('shot') or {}).get('color_pullback'),
                'functional_map_fpfh': (_results.get('fpfh') or {}).get('functional_map'),
                'color_pullback_fpfh': (_results.get('fpfh') or {}).get('color_pullback'),
                'output_folder': result_path,
                'folder': 'single',
            }
            summary_results.append(entry)
            processed_samples[single_name] = entry
            state['processed_samples'] = processed_samples
            save_state(state, state_path)
            write_summary_html(summary_results, target_path)

            # If web viewer requested, generate HTML using the already-computed matches
            if args.web_view:
                try:
                    viewer_res = _results.get(descriptor_type)
                    if viewer_res is None:
                        raise RuntimeError(f"No results found for descriptor '{descriptor_type}' to generate web view.")
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
                    key_name = f'interactive_view_{descriptor_type}'
                    entry[key_name] = html_rel
                    processed_samples[single_name] = entry
                    state['processed_samples'] = processed_samples
                    save_state(state, state_path)
                    write_summary_html(summary_results, target_path)
                    print(f"Web viewer generated: {html_path}")
                except Exception as e:
                    print(f"Web viewer generation failed (no recompute): {e}")

            # Popup viewer deprecated; web-based viewer available via --web-view
    else:
        partial_folders = ["cuts", "holes"]
        for folder in partial_folders:
            partial_off_dir = os.path.join(PARTIAL_ROOT_DIR_DEFAULT, folder, 'off')
            partial_files = os.listdir(partial_off_dir)
            i = 0
            for partial_file in partial_files:
                # remove extension safely
                partial_mesh_name = os.path.splitext(partial_file)[0]

                # safe extraction of the full mesh name from the partial's filename
                parts = partial_mesh_name.split('_')
                if len(parts) >= 2:
                    full_mesh_name = parts[1]
                else:
                    full_mesh_name = partial_mesh_name
                mesh_data = TestMeshData(
                    name=partial_mesh_name,
                    full_mesh=os.path.join(FULL_MESH_DIR_DEFAULT, f"{full_mesh_name}.off"),
                    partial_mesh=os.path.join(PARTIAL_ROOT_DIR_DEFAULT, folder, 'off', partial_file),
                    ground_truth=os.path.join(PARTIAL_ROOT_DIR_DEFAULT, folder, 'corres', f"{partial_mesh_name}.vts")
                )
                result_path = os.path.join(target_path, folder, partial_mesh_name)

                # skip if already processed (from persisted state)
                if partial_mesh_name in processed_samples:
                    continue
                # Process all samples

                # Decide descriptors to run
                if args.benchmark:
                    _types_to_run = ['dino', 'dinov3', 'shot', 'fpfh']
                else:
                    _types_to_run = [descriptor_type]

                _results = {}
                for _dt in _types_to_run:
                    opts.descriptor_type = _dt
                    _results[_dt] = run(mesh_data, result_path, opts, target_path)

                # aggregate into one summary entry (only fill those run)
                entry = {
                    'name': partial_mesh_name,
                    'mean_dino': (_results.get('dino') or {}).get('mean'),
                    'mean_dinov3': (_results.get('dinov3') or {}).get('mean'),
                    'mean_shot': (_results.get('shot') or {}).get('mean'),
                    'mean_fpfh': (_results.get('fpfh') or {}).get('mean'),
                    'functional_map_dino': (_results.get('dino') or {}).get('functional_map'),
                    'color_pullback_dino': (_results.get('dino') or {}).get('color_pullback'),
                    'functional_map_dinov3': (_results.get('dinov3') or {}).get('functional_map'),
                    'color_pullback_dinov3': (_results.get('dinov3') or {}).get('color_pullback'),
                    'functional_map_shot': (_results.get('shot') or {}).get('functional_map'),
                    'color_pullback_shot': (_results.get('shot') or {}).get('color_pullback'),
                    'functional_map_fpfh': (_results.get('fpfh') or {}).get('functional_map'),
                    'color_pullback_fpfh': (_results.get('fpfh') or {}).get('color_pullback'),
                    'output_folder': result_path,
                    'folder': folder,
                }
                summary_results.append(entry)
                processed_samples[partial_mesh_name] = entry
                state['processed_samples'] = processed_samples
                # write incremental HTML summary after each processed mesh
                save_state(state, state_path)
                write_summary_html(summary_results, target_path)
                
                i += 1
                if i == 50:
                    break

            # In dataset mode, viewer is less defined; skipping popup to avoid repeated openings

if __name__ == "__main__":
    main()