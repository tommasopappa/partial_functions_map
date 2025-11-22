from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

def compute_geodesic_descriptors(M : ManifoldMesh, N : ManifoldMesh, matches, opts: Options):
    v_N, f_N = N.vert.numpy(force=True), N.triv.numpy(force=True)
    v_M, f_M = M.vert.numpy(force=True), M.triv.numpy(force=True)
    fps_indices = _fps_euclidean(v_N, opts.refine_fps)
    func_M, func_N = _compute_indicator_functions(v_M, v_N, f_M, f_N, fps_indices, matches, opts.fps_variance)
    func_M = torch.tensor(func_M, dtype=torch.float32, device=opts.device)
    func_N = torch.tensor(func_N, dtype=torch.float32, device=opts.device)
    return func_M, func_N

def _fps_euclidean(vertices : np.ndarray, n_samples, start_idx=0):
    n_vert = len(vertices)

    """Farthest point sampling"""
    if n_samples >= n_vert:
        return np.arange(n_vert)

    fps_indices = [start_idx]
    dists = euclidean_distances(vertices[[start_idx]], vertices).squeeze()

    for _ in range(n_samples - 1):
        new_idx = np.argmax(dists)
        fps_indices.append(new_idx)
        new_dists = euclidean_distances(vertices[[new_idx]], vertices).squeeze()
        dists = np.minimum(dists, new_dists)

    return np.array(fps_indices)

def _compute_geodesic_distances_mesh(vert : np.ndarray, triv : np.ndarray, source_indices):
    """Compute geodesic distances on a mesh from source vertices"""
    # Build edge list from faces
    edges = set()
    for face in triv:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            edges.add(tuple(sorted([v1, v2])))

    # Create sparse adjacency matrix
    row, col, data = [], [], []
    for v1, v2 in edges:
        dist = np.linalg.norm(vert[v1] - vert[v2])
        row.extend([v1, v2])
        col.extend([v2, v1])
        data.extend([dist, dist])

    adj_matrix = csr_matrix((data, (row, col)), shape=(len(vert), len(vert)))

    # Compute geodesic distances using Dijkstra
    distances = dijkstra(adj_matrix, indices=source_indices, directed=False)

    return distances

def _compute_indicator_functions(v_M, v_N, f_M, f_N, fps_indices, matches, variance):
    """Compute indicator functions using geodesic distances"""
    n_fps = len(fps_indices)

    # Get corresponding points on M
    fps_matches_M = [matches[idx] for idx in fps_indices]

    print(f"  Computing geodesic distances on N...")
    geo_dists_N = _compute_geodesic_distances_mesh(v_N, f_N, fps_indices)

    print(f"  Computing geodesic distances on M...")
    geo_dists_M = _compute_geodesic_distances_mesh(v_M, f_M, fps_matches_M)

    # Convert to indicator functions
    F = np.exp(-0.5 * variance * geo_dists_N.T**2)
    G = np.exp(-0.5 * variance * geo_dists_M.T**2)

    return F, G