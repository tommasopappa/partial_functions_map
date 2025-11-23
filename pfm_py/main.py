from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options

import torch
import open3d as o3d
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

opts = Options(device)

mesh_M = o3d.io.read_triangle_mesh('/usr/prakt/w0012/SAVHA/shape_data/SHREC16/null/off/cat.off')
mesh_N = o3d.io.read_triangle_mesh('/usr/prakt/w0012/SAVHA/shape_data/SHREC16/holes/off/holes_cat_shape_10.off')

vert_M, triv_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
vert_N, triv_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)
M = ManifoldMesh(vert_M, triv_M, opts, compute_geo=True)
N = ManifoldMesh(vert_N, triv_N, opts, compute_geo=False)

C, v, matches = match_and_refine(M, N, opts)
C, v, matches = C.numpy(force=True), v.numpy(force=True), matches.numpy(force=True)

gt_matches = np.loadtxt('/usr/prakt/w0012/SAVHA/shape_data/SHREC16/holes/corres/holes_cat_shape_10.vts', dtype=float).astype(int) - 1

geodesics_M = M.compute_geodesic_matrix()
dist_method_geo = np.array([geodesics_M[gt_matches[i], matches[i]] for i in range(len(matches))])
dist_method_geo = dist_method_geo / np.sqrt(M.area)
mean_geodesic_error = dist_method_geo.mean()
print(f"Mean geodesic error: {mean_geodesic_error}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Prepare visualization data ---

# Center shapes for visualization
v_N_vis = vert_N - vert_N.mean(0)
v_M_vis = vert_M - vert_M.mean(0)

# Compute source function on N
v_N_norm = (vert_N - vert_N.min(0)) / (vert_N.max(0) - vert_N.min(0))
source_fun = v_N_norm[:, 0]

# Colors transferred via ground truth + method
colors_gt = np.zeros(vert_M.shape[0])
colors_gt[gt_matches] = source_fun

colors_method = np.zeros(vert_M.shape[0])
colors_method[matches] = source_fun

dist_gt_geo = np.zeros_like(dist_method_geo)


# --- Make figure (2 rows) ---

fig = plt.figure(figsize=(14, 10))

# ============== ROW 1: Source + GT Transfer ==============

# Source function on N
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
            c=source_fun, cmap="viridis", s=20)
ax1.set_title("N: Source Function")
ax1.view_init(elev=20, azim=45)

# GT transfer onto M
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.scatter(v_M_vis[:,0], v_M_vis[:,1], v_M_vis[:,2],
            c=colors_gt, cmap="viridis", s=10)
ax2.set_title("GROUND TRUTH Transfer")
ax2.view_init(elev=20, azim=45)

# GT error (always 0)
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
sc3 = ax3.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
                  c=dist_gt_geo, cmap="coolwarm", s=20, vmin=0, vmax=0.1)
ax3.set_title(f"GT Error (mean = {dist_gt_geo.mean():.4f})")
plt.colorbar(sc3, ax=ax3, shrink=0.6)
ax3.view_init(elev=20, azim=45)

# ============== ROW 2: Method Transfer + Errors ==============

# Method transfer to M
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.scatter(v_M_vis[:,0], v_M_vis[:,1], v_M_vis[:,2],
            c=colors_method, cmap="viridis", s=10)
ax4.set_title("METHOD Transfer")
ax4.view_init(elev=20, azim=45)

# Error heatmap on N
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
vmax = np.percentile(dist_method_geo, 95)
sc5 = ax5.scatter(v_N_vis[:,0], v_N_vis[:,1], v_N_vis[:,2],
                  c=dist_method_geo, cmap="coolwarm", s=20,
                  vmin=0, vmax=vmax)
ax5.set_title(f"Method Error (mean = {dist_method_geo.mean():.4f})")
plt.colorbar(sc5, ax=ax5, shrink=0.6)
ax5.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("pfm_visualization.png", dpi=300)
print("Saved visualization to pfm_visualization.png")
