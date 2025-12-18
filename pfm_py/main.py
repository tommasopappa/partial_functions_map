from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options

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

def create_functional_map_visualization(vert_M, vert_N, triv_M, triv_N, M, N, C, matches, gt_matches, dist_method_geo, opts, output_folder):
    """Create and save the functional map visualization showing source function, ground truth transfer, and method transfer."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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

    # Center shapes for visualization
    v_N_vis = vert_N - vert_N.mean(0)
    v_M_vis = vert_M - vert_M.mean(0)
    
    # Find boundary edges
    boundary_edges_N = find_boundary_edges(triv_N)
    boundary_edges_M = find_boundary_edges(triv_M)

    bbox_min = v_N_vis.min(0)
    bbox_max = v_N_vis.max(0)
    bbox_range = (bbox_max - bbox_min).max()
    bbox_center = (bbox_max + bbox_min) / 2
    margin = 0.1 * bbox_range
    lim = bbox_range / 2 + margin

    def set_axes_equal(ax):
        ax.set_xlim([bbox_center[0] - lim, bbox_center[0] + lim])
        ax.set_ylim([bbox_center[1] - lim, bbox_center[1] + lim])
        ax.set_zlim([bbox_center[2] - lim, bbox_center[2] + lim])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    # Compute source function on N
    v_N_norm = (vert_N - vert_N.min(0)) / (vert_N.max(0) - vert_N.min(0))
    source_fun = v_N_norm[:, 0]

    # Colors transferred via ground truth + method
    colors_gt = np.zeros(vert_M.shape[0])
    colors_gt[gt_matches] = source_fun

    colors_method = np.zeros(vert_M.shape[0])
    colors_method = N.evecs.T.numpy(force=True) @ (N.S.numpy(force=True) * source_fun)
    colors_method = C @ colors_method
    colors_method = M.evecs.numpy(force=True) @ colors_method

    dist_gt_geo = np.zeros_like(dist_method_geo)
    
    # Prepare mesh polygons and face colors
    poly_N = [v_N_vis[f] for f in triv_N]
    poly_M = [v_M_vis[f] for f in triv_M]
    
    # Map vertex colors to face colors (average of vertex colors)
    cmap_viridis = plt.get_cmap("viridis")
    cmap_coolwarm = plt.get_cmap("coolwarm")
    
    source_fun_colors = cmap_viridis((source_fun - source_fun.min()) / (source_fun.max() - source_fun.min()))[:, :3]
    facecols_source = source_fun_colors[triv_N].mean(axis=1)
    
    colors_gt_norm = (colors_gt - colors_gt.min()) / (colors_gt.max() - colors_gt.min() + 1e-10)
    colors_gt_rgb = cmap_viridis(colors_gt_norm)[:, :3]
    facecols_gt = colors_gt_rgb[triv_M].mean(axis=1)
    
    colors_method_norm = (colors_method - colors_method.min()) / (colors_method.max() - colors_method.min() + 1e-10)
    colors_method_rgb = cmap_viridis(colors_method_norm)[:, :3]
    facecols_method = colors_method_rgb[triv_M].mean(axis=1)
    
    dist_gt_geo_colors = cmap_coolwarm(dist_gt_geo / 0.1)[:, :3]
    facecols_gt_error = dist_gt_geo_colors[triv_N].mean(axis=1)
    
    vmax_error = np.percentile(dist_method_geo, 95)
    dist_method_geo_norm = np.clip(dist_method_geo / vmax_error, 0, 1)
    dist_method_geo_colors = cmap_coolwarm(dist_method_geo_norm)[:, :3]
    facecols_method_error = dist_method_geo_colors[triv_N].mean(axis=1)

    # --- Make figure (2 rows) ---

    fig = plt.figure(figsize=(14, 10))
    boundary_line_width = 2
    opacity = 1.0

    # ============== ROW 1: Source + GT Transfer ==============

    # Source function on N
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    pc1 = Poly3DCollection(poly_N, facecolors=facecols_source, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax1.add_collection3d(pc1)
    for edge in boundary_edges_N:
        pts = v_N_vis[list(edge)]
        ax1.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax1.set_title("N: Source Function")
    set_axes_equal(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(False)

    # GT transfer onto M
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    pc2 = Poly3DCollection(poly_M, facecolors=facecols_gt, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax2.add_collection3d(pc2)
    for edge in boundary_edges_M:
        pts = v_M_vis[list(edge)]
        ax2.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax2.set_title("GROUND TRUTH Transfer")
    set_axes_equal(ax2)
    ax2.view_init(elev=20, azim=45)
    ax2.grid(False)

    # GT error (always 0)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    pc3 = Poly3DCollection(poly_N, facecolors=facecols_gt_error, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax3.add_collection3d(pc3)
    for edge in boundary_edges_N:
        pts = v_N_vis[list(edge)]
        ax3.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax3.set_title(f"GT Error (mean = {dist_gt_geo.mean():.4f})")
    set_axes_equal(ax3)
    ax3.view_init(elev=20, azim=45)
    ax3.grid(False)
    # Add colorbar
    sm3 = plt.cm.ScalarMappable(cmap=cmap_coolwarm, norm=plt.Normalize(vmin=0, vmax=0.1))
    sm3.set_array([])
    plt.colorbar(sm3, ax=ax3, shrink=0.6)

    # ============== ROW 2: Method Transfer + Errors ==============

    # Method transfer to M
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    pc4 = Poly3DCollection(poly_M, facecolors=facecols_method, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax4.add_collection3d(pc4)
    for edge in boundary_edges_M:
        pts = v_M_vis[list(edge)]
        ax4.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax4.set_title("METHOD Transfer")
    set_axes_equal(ax4)
    ax4.view_init(elev=20, azim=45)
    ax4.grid(False)

    # Error heatmap on N
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    pc5 = Poly3DCollection(poly_N, facecolors=facecols_method_error, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax5.add_collection3d(pc5)
    for edge in boundary_edges_N:
        pts = v_N_vis[list(edge)]
        ax5.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax5.set_title(f"Method Error (mean = {dist_method_geo.mean():.4f})")
    set_axes_equal(ax5)
    ax5.view_init(elev=20, azim=45)
    ax5.grid(False)
    # Add colorbar
    sm5 = plt.cm.ScalarMappable(cmap=cmap_coolwarm, norm=plt.Normalize(vmin=0, vmax=vmax_error))
    sm5.set_array([])
    plt.colorbar(sm5, ax=ax5, shrink=0.6)

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

    # create vertex colormap on M and transfer to N by indexing (pullback)
    colors_M = create_full_colormap(vert_M.shape[0])   # (n_M,3)
    colors_N_method = colors_M[matches]                # (n_N,3) method pullback
    colors_N_gt = colors_M[gt_matches]                 # (n_N,3) ground truth pullback

    # prepare centered geometry (use same centering as first block)
    v_M_vis = vert_M - vert_N.mean(0)   # center both shapes on partial-shape center (keeps views consistent)
    v_N_vis = vert_N - vert_N.mean(0)
    
    # Find boundary edges
    boundary_edges_M = find_boundary_edges(triv_M)
    boundary_edges_N = find_boundary_edges(triv_N)

    # compute face polygons and per-face colors (average vertex colors per face)
    poly_M = [v_M_vis[f] for f in triv_M]
    facecols_M = colors_M[triv_M].mean(axis=1)

    poly_N = [v_N_vis[f] for f in triv_N]
    facecols_N_method = colors_N_method[triv_N].mean(axis=1)
    facecols_N_gt = colors_N_gt[triv_N].mean(axis=1)

    # ensure bounding box and aspect come from the same bbox used in first block (recompute if needed)
    bbox_min = v_N_vis.min(0)
    bbox_max = v_N_vis.max(0)
    bbox_range = (bbox_max - bbox_min).max()
    bbox_center = (bbox_max + bbox_min) / 2
    margin = 0.1 * bbox_range
    lim = bbox_range / 2 + margin

    def set_axes_equal_local(ax):
        ax.set_xlim([bbox_center[0] - lim, bbox_center[0] + lim])
        ax.set_ylim([bbox_center[1] - lim, bbox_center[1] + lim])
        ax.set_zlim([bbox_center[2] - lim, bbox_center[2] + lim])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    # create figure: left = full mesh, middle = GT pullback, right = method pullback
    fig = plt.figure(figsize=(18, 7))
    boundary_line_width = 2
    opacity = 1.0

    # Full mesh (continuous)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    pc1 = Poly3DCollection(poly_M, facecolors=facecols_M, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax1.add_collection3d(pc1)
    # Plot boundary edges in black
    for edge in boundary_edges_M:
        pts = v_M_vis[list(edge)]
        ax1.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax1.set_title("Full Mesh (M)\ncontinuous colors", pad=20)
    set_axes_equal_local(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.grid(False)

    # Partial mesh with ground truth pullback
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    pc2 = Poly3DCollection(poly_N, facecolors=facecols_N_gt, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax2.add_collection3d(pc2)
    # Plot boundary edges in black
    for edge in boundary_edges_N:
        pts = v_N_vis[list(edge)]
        ax2.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax2.set_title("Partial Mesh (N)\nGround Truth Pullback", pad=20)
    set_axes_equal_local(ax2)
    ax2.view_init(elev=20, azim=45)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.grid(False)

    # Partial mesh with method pullback
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    pc3 = Poly3DCollection(poly_N, facecolors=facecols_N_method, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax3.add_collection3d(pc3)
    # Plot boundary edges in black
    for edge in boundary_edges_N:
        pts = v_N_vis[list(edge)]
        ax3.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
    ax3.set_title("Partial Mesh (N)\nMethod Pullback", pad=20)
    set_axes_equal_local(ax3)
    ax3.view_init(elev=20, azim=45)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.grid(False)

    plt.tight_layout(pad=2.0)
    color_pullback_fname = f"color_pullback_{opts.descriptor_type}.png"
    color_pullback_path = os.path.join(output_folder, color_pullback_fname)
    plt.savefig(color_pullback_path, dpi=300)
    print(f"Saved: {color_pullback_path}")
    
    return color_pullback_path


def run(mesh_data, output_folder, opts: Options):
    os.makedirs(output_folder, exist_ok=True)

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

    gt_matches = np.loadtxt(mesh_data.ground_truth, dtype=float).astype(int) - 1

    geodesics_M = M.compute_geodesic_matrix()
    dist_method_geo = np.array([geodesics_M[gt_matches[i], matches[i]] for i in range(len(matches))])
    dist_method_geo = dist_method_geo / np.sqrt(M.area)
    mean_geodesic_error = dist_method_geo.mean()
    print(f"Mean geodesic error: {mean_geodesic_error:.6f}")

    # Create visualizations using helper functions
    functional_map_path = create_functional_map_visualization(
        vert_M, vert_N, triv_M, triv_N, M, N, C, matches, gt_matches, dist_method_geo, opts, output_folder
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
    shot_vals = np.array([r.get('mean_shot') for r in rows], dtype=float) if rows else np.array([], dtype=float)
    fpfh_vals = np.array([r.get('mean_fpfh') for r in rows], dtype=float) if rows else np.array([], dtype=float)

    def safe_mean(arr):
        if arr.size == 0:
            return float('nan')
        return float(np.mean(arr))

    overall_dino = safe_mean(dino_vals)
    overall_shot = safe_mean(shot_vals)
    overall_fpfh = safe_mean(fpfh_vals)

    cuts_rows = [r for r in rows if r.get('folder') == 'cuts']
    holes_rows = [r for r in rows if r.get('folder') == 'holes']

    cuts_count = len(cuts_rows)
    holes_count = len(holes_rows)
    total_count = len(rows)

    cuts_dino = safe_mean(np.array([r.get('mean_dino') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    cuts_shot = safe_mean(np.array([r.get('mean_shot') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    cuts_fpfh = safe_mean(np.array([r.get('mean_fpfh') for r in cuts_rows], dtype=float)) if cuts_count > 0 else float('nan')
    holes_dino = safe_mean(np.array([r.get('mean_dino') for r in holes_rows], dtype=float)) if holes_count > 0 else float('nan')
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
        '<tr><th>Category</th><th>Count</th><th>Mean Geodesic Error (DINO)</th><th>Mean Geodesic Error (SHOT)</th><th>Mean Geodesic Error (FPFH)</th></tr>',
        f'<tr><td>Cuts</td><td>{cuts_count}</td><td>{cuts_dino:.6f}</td><td>{cuts_shot:.6f}</td><td>{cuts_fpfh:.6f}</td></tr>',
        f'<tr><td>Holes</td><td>{holes_count}</td><td>{holes_dino:.6f}</td><td>{holes_shot:.6f}</td><td>{holes_fpfh:.6f}</td></tr>',
        f'<tr><td>Entire dataset</td><td>{total_count}</td><td>{overall_dino:.6f}</td><td>{overall_shot:.6f}</td><td>{overall_fpfh:.6f}</td></tr>',
        '</table>',
        '<hr/>',
        '<h2>Individual Mesh Results</h2>',
        '<table>',
        '<tr><th>Name</th><th>Mean Geodesic Error (DINO)</th><th>Mean Geodesic Error (SHOT)</th><th>Mean Geodesic Error (FPFH)</th><th>DINO Visualizations</th><th>SHOT Visualizations</th><th>FPFH Visualizations</th></tr>'
    ]

    for r in rows:
        dino_links = []
        shot_links = []
        fpfh_links = []
        if r.get('functional_map_dino'):
            dino_links.append(f'<a href="{r["functional_map_dino"]}" target="_blank">functional_map_visualization_dino</a>')
        if r.get('color_pullback_dino'):
            dino_links.append(f'<a href="{r["color_pullback_dino"]}" target="_blank">color_pullback_dino</a>')
        if r.get('functional_map_shot'):
            shot_links.append(f'<a href="{r["functional_map_shot"]}" target="_blank">functional_map_visualization_shot</a>')
        if r.get('color_pullback_shot'):
            shot_links.append(f'<a href="{r["color_pullback_shot"]}" target="_blank">color_pullback_shot</a>')
        if r.get('functional_map_fpfh'):
            fpfh_links.append(f'<a href="{r["functional_map_fpfh"]}" target="_blank">functional_map_visualization_fpfh</a>')
        if r.get('color_pullback_fpfh'):
            fpfh_links.append(f'<a href="{r["color_pullback_fpfh"]}" target="_blank">color_pullback_fpfh</a>')

        dino_html = ' | '.join(dino_links) if dino_links else ''
        shot_html = ' | '.join(shot_links) if shot_links else ''
        fpfh_html = ' | '.join(fpfh_links) if fpfh_links else ''

        mean_dino = r.get('mean_dino') if r.get('mean_dino') is not None else float('nan')
        mean_shot = r.get('mean_shot') if r.get('mean_shot') is not None else float('nan')
        mean_fpfh = r.get('mean_fpfh') if r.get('mean_fpfh') is not None else float('nan')

        html_lines.append(f'<tr><td>{r["name"]}</td><td>{mean_dino:.6f}</td><td>{mean_shot:.6f}</td><td>{mean_fpfh:.6f}</td><td>{dino_html}</td><td>{shot_html}</td><td>{fpfh_html}</td></tr>')

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

# Command-line argument parsing
parser = argparse.ArgumentParser(
    description='Partial Functions Map - 3D shape matching',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python main.py --fpfh                                    # Use FPFH descriptors (default)
  python main.py --shot                                    # Use SHOT descriptors
  python main.py --data-path /path/to/data --shot         # Specify custom data path
  python main.py --fpfh --data-path ~/data/shapes         # FPFH with custom path
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
    '--data-path',
    type=str,
    default='/usr/prakt/w0010/SAVHA/shape_data',
    help='Path to the shape data directory (default: /usr/prakt/w0010/SAVHA/shape_data)'
)
parser.add_argument(
    '--target-path',
    type=str,
    default='results',
    help='Path to the output results directory (default: results)'
)

args = parser.parse_args()

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

# Data path
data_path = args.data_path
target_path = args.target_path
state_path = os.path.join(target_path, 'state.txt')

print(f"Using descriptor: {descriptor_type.upper()}")
print(f"Data path: {data_path}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
opts = Options(device)
 
# load persisted state (processed samples) and initialize summary_results from it
state = load_state(state_path)
processed_samples = state.get('processed_samples', {})
summary_results = list(processed_samples.values())

partial_folders = ["cuts", "holes"]
for folder in partial_folders:
    partial_files = os.listdir(data_path + "/SHREC16/" + folder + "/off")
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
            full_mesh=data_path + f"/SHREC16/null/off/{full_mesh_name}.off",
            partial_mesh=data_path + f"/SHREC16/{folder}/off/{partial_file}",
            ground_truth=data_path + f"/SHREC16/{folder}/corres/{partial_mesh_name}.vts"
        )
        result_path = f"{target_path}/{folder}/{partial_mesh_name}"

        # skip if already processed (from persisted state)
        if partial_mesh_name in processed_samples:
            continue
        if partial_mesh_name != "holes_cat_shape_10" and partial_mesh_name != "cuts_cat_shape_10":
            continue

        # run with DINO, SHOT, and FPFH
        opts.descriptor_type = 'dino'
        res_dino = run(mesh_data, result_path, opts)

        opts.descriptor_type = 'shot'
        res_shot = run(mesh_data, result_path, opts)

        opts.descriptor_type = 'dinov3'
        res_fpfh = run(mesh_data, result_path, opts)

        # aggregate into one summary entry
        entry = {
            'name': partial_mesh_name,
            'mean_dino': res_dino.get('mean'),
            'mean_shot': res_shot.get('mean'),
            'mean_fpfh': res_fpfh.get('mean'),
            'functional_map_dino': res_dino.get('functional_map'),
            'color_pullback_dino': res_dino.get('color_pullback'),
            'functional_map_shot': res_shot.get('functional_map'),
            'color_pullback_shot': res_shot.get('color_pullback'),
            'functional_map_fpfh': res_fpfh.get('functional_map'),
            'color_pullback_fpfh': res_fpfh.get('color_pullback'),
            'output_folder': result_path,
            'folder': folder,
        }
        summary_results.append(entry)
        processed_samples[partial_mesh_name] = entry
        state['processed_samples'] = processed_samples
        # write incremental HTML summary after each processed mesh
        save_state(state, state_path)
        write_summary_html(summary_results, target_path)