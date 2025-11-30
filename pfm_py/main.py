from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options

import torch
import open3d as o3d
import numpy as np
import os
import argparse

from dataclasses import dataclass

@dataclass
class TestMeshData:
    name: str
    full_mesh: str
    partial_mesh: str
    ground_truth: str

target_path = 'results'
cat_holes = TestMeshData(
    name='cat_holes_10',
    full_mesh='SHREC16/null/off/cat.off',
    partial_mesh='SHREC16/holes/off/holes_cat_shape_10.off',
    ground_truth='SHREC16/holes/corres/holes_cat_shape_10.vts'
)
victoria_cut = TestMeshData(
    name='victoria_cuts_1',
    full_mesh='SHREC16/null/off/victoria.off',
    partial_mesh='SHREC16/cuts/off/cuts_victoria_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_victoria_shape_1.vts'
)
michael_cut = TestMeshData(
    name='michael_cuts_1',
    full_mesh='SHREC16/null/off/michael.off',
    partial_mesh='SHREC16/cuts/off/cuts_michael_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_michael_shape_1.vts'
)
horse_cut = TestMeshData(
    name='horse_cuts_1',
    full_mesh='SHREC16/null/off/horse.off',
    partial_mesh='SHREC16/cuts/off/cuts_horse_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_horse_shape_1.vts'
)
centaur_cut = TestMeshData(
    name='centaur_cuts_1',
    full_mesh='SHREC16/null/off/centaur.off',
    partial_mesh='SHREC16/cuts/off/cuts_centaur_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_centaur_shape_1.vts'
)
cat_cut = TestMeshData(
    name='cat_cuts_1',
    full_mesh='SHREC16/null/off/cat.off',
    partial_mesh='SHREC16/cuts/off/cuts_cat_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_cat_shape_1.vts'
)
dog_cut = TestMeshData(
    name='dog_cuts_1',
    full_mesh='SHREC16/null/off/dog.off',
    partial_mesh='SHREC16/cuts/off/cuts_dog_shape_1.off',
    ground_truth='SHREC16/cuts/corres/cuts_dog_shape_1.vts'
)
experiments = [victoria_cut, michael_cut, horse_cut, centaur_cut, cat_cut, dog_cut, cat_holes]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

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
    '--data-path',
    type=str,
    default='/usr/prakt/w0010/SAVHA/shape_data',
    help='Path to the shape data directory (default: /usr/prakt/w0010/SAVHA/shape_data)'
)

args = parser.parse_args()

# Determine the descriptor type to use
descriptor_type = "fpfh"  # Default value
if args.shot:
    descriptor_type = "shot"
elif args.fpfh:
    descriptor_type = "fpfh"

# Data path
data_path = args.data_path

print(f"Using descriptor: {descriptor_type.upper()}")
print(f"Data path: {data_path}")
print()

for test in experiments:
    output_folder = target_path + '/' + test.name
    os.makedirs(output_folder, exist_ok=True)

    print('#'*60)
    print(f"Running `{test.name}` ...")
    print('#'*60)
    opts = Options(device, descriptor_type=descriptor_type)

    mesh_M = o3d.io.read_triangle_mesh(data_path + '/' + test.full_mesh)
    mesh_N = o3d.io.read_triangle_mesh(data_path + '/' + test.partial_mesh)

    vert_M, triv_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
    vert_N, triv_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)
    M = ManifoldMesh(vert_M, triv_M, opts, compute_geo=True)
    N = ManifoldMesh(vert_N, triv_N, opts, compute_geo=False)

    C, v, matches = match_and_refine(M, N, opts)
    C, v, matches = C.numpy(force=True), v.numpy(force=True), matches.numpy(force=True)

    gt_matches = np.loadtxt(data_path + '/' + test.ground_truth, dtype=float).astype(int) - 1

    geodesics_M = M.compute_geodesic_matrix()
    dist_method_geo = np.array([geodesics_M[gt_matches[i], matches[i]] for i in range(len(matches))])
    dist_method_geo = dist_method_geo / np.sqrt(M.area)
    mean_geodesic_error = dist_method_geo.mean()
    print(f"Mean geodesic error: {mean_geodesic_error:.6f}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # --- Prepare visualization data ---

    def create_full_colormap(n):
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 1, n))[:, :3]
        return colors

    # Center shapes for visualization
    v_N_vis = vert_N - vert_N.mean(0)
    v_M_vis = vert_M - vert_M.mean(0)

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
        ax.set_box_aspect([1, 1, 1])

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


    # --- Make figure (2 rows) ---

    fig = plt.figure(figsize=(14, 10))

    # ============== ROW 1: Source + GT Transfer ==============

    # Source function on N
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
                c=source_fun, cmap="viridis", s=20)
    ax1.set_title("N: Source Function")
    set_axes_equal(ax1)
    ax1.view_init(elev=20, azim=45)

    # GT transfer onto M
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(v_M_vis[:,0], v_M_vis[:,1], v_M_vis[:,2],
                c=colors_gt, cmap="viridis", s=10)
    ax2.set_title("GROUND TRUTH Transfer")
    set_axes_equal(ax2)
    ax2.view_init(elev=20, azim=45)

    # GT error (always 0)
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    sc3 = ax3.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
                    c=dist_gt_geo, cmap="coolwarm", s=20, vmin=0, vmax=0.1)
    ax3.set_title(f"GT Error (mean = {dist_gt_geo.mean():.4f})")
    set_axes_equal(ax3)
    plt.colorbar(sc3, ax=ax3, shrink=0.6)
    ax3.view_init(elev=20, azim=45)

    # ============== ROW 2: Method Transfer + Errors ==============

    # Method transfer to M
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(v_M_vis[:,0], v_M_vis[:,1], v_M_vis[:,2],
                c=colors_method, cmap="viridis", s=10)
    ax4.set_title("METHOD Transfer")
    set_axes_equal(ax4)
    ax4.view_init(elev=20, azim=45)

    # Error heatmap on N
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    vmax = np.percentile(dist_method_geo, 95)
    sc5 = ax5.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
                    c=dist_method_geo, cmap="coolwarm", s=20,
                    vmin=0, vmax=vmax)
    ax5.set_title(f"Method Error (mean = {dist_method_geo.mean():.4f})")
    set_axes_equal(ax5)
    plt.colorbar(sc5, ax=ax5, shrink=0.6)
    ax5.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/pfm_visualization.png", dpi=300)
    print(f"Saved visualization to {output_folder}/pfm_visualization.png")


    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib as mpl

    # create vertex colormap on M and transfer to N by indexing
    colors_M = create_full_colormap(vert_M.shape[0])   # (n_M,3)
    colors_N = colors_M[matches]                       # (n_N,3)

    # prepare centered geometry (use same centering as first block)
    v_M_vis = vert_M - vert_N.mean(0)   # center both shapes on partial-shape center (keeps views consistent)
    v_N_vis = vert_N - vert_N.mean(0)

    # compute face polygons and per-face colors (average vertex colors per face)
    poly_M = [v_M_vis[f] for f in triv_M]
    facecols_M = colors_M[triv_M].mean(axis=1)

    poly_N = [v_N_vis[f] for f in triv_N]
    facecols_N = colors_N[triv_N].mean(axis=1)

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

    # create figure: left = continuous full mesh, right = continuous partial mesh
    fig = plt.figure(figsize=(14, 6))

    # Full mesh (continuous)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    pc1 = Poly3DCollection(poly_M, facecolors=facecols_M, linewidths=0, edgecolor=None)
    ax1.add_collection3d(pc1)
    ax1.set_title("Full Mesh (M) — continuous colors")
    set_axes_equal_local(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.grid(False)

    # Partial mesh (continuous, indexed colors)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pc2 = Poly3DCollection(poly_N, facecolors=facecols_N, linewidths=0, edgecolor=None)
    ax2.add_collection3d(pc2)
    ax2.set_title("Partial Mesh (N) — colors via matches indexing")
    set_axes_equal_local(ax2)
    ax2.view_init(elev=20, azim=45)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/indexed_color_transfer.png", dpi=300)
    print(f"Saved: {output_folder}/indexed_color_transfer.png")
    print()
    print()