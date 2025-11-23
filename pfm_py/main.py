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

C, v, m = match_and_refine(M, N, opts)