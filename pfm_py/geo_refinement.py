from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

def compute_geodesic_descriptors(M : ManifoldMesh, N : ManifoldMesh, matches, opts: Options):
    """Compute geodesic-based indicator descriptors on M and N.

    Samples `opts.fps_n_sample_points` farthest points (FPS) on N, maps them to M via
    `matches`, then computes geodesic distance fields and converts them into Gaussian
    indicator functions on both meshes.
    For every sample point P on N, we obtain
        1. a descriptor function f: N -> ℝ s.t. for any point Q of N,
            f(Q) is measure of the geodesic distance of Q from P
        2. a corresponding descriptor g: M -> ℝ s.t. for any point Q' of M, 
            g(Q') is a measure of the geodesic distance of Q' from P',
            where P' is the match of P on M according to `matches`.

    Parameters:
        M (ManifoldMesh): Full target mesh
        N (ManifoldMesh): Partial source mesh
        matches (torch.Tensor | np.ndarray): Array-like of length N.n_vert mapping each vertex
            of N to a corresponding vertex index on M (int indices)
        opts (Options): Options and hyperparameters.

    Returns:
        (torch.Tensor, torch.Tensor):
            - func_M: shape (M.n_vert, n_samples) indicator functions on M
            - func_N: shape (N.n_vert, n_samples) indicator functions on N
    """
    v_N, f_N = N.vert.numpy(force=True), N.triv.numpy(force=True)
    v_M, f_M = M.vert.numpy(force=True), M.triv.numpy(force=True)
    
    fps_variance = opts.geo_descriptor_variance * M.area # scale-dependent, see doc in options.py
    scale_factor = np.sqrt(M.area / 17500)
    fps_variance = 0.7 * scale_factor

    fps_indices = _fps_euclidean(v_N, opts.fps_n_sample_points)
    func_M, func_N = _compute_indicator_functions(v_M, v_N, f_M, f_N, fps_indices, matches.numpy(force=True), fps_variance)
    func_M = torch.tensor(func_M, dtype=torch.float32, device=opts.device)
    func_N = torch.tensor(func_N, dtype=torch.float32, device=opts.device)
    return func_M, func_N

def _fps_euclidean(vertices : np.ndarray, n_samples, start_idx=0):
    """Farthest point sampling (FPS) on vertex positions.

    Selects `n_samples` points by iteratively choosing the farthest vertex from the set
    of already selected vertices under Euclidean distance.

    Parameters:
        vertices (np.ndarray): shape (n_vert, 3) vertex coordinates
        n_samples (int): desired number of samples (clamped to n_vert)
        start_idx (int): vertex index of the initial sample point (default: 0)

    Returns:
        np.ndarray: shape (n_samples,) selected vertex indices
    """
    n_vert = len(vertices)
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
    """Approximates geodesic distances on a mesh from source vertices.

    Builds an undirected graph from triangle edges with edge weights equal to Euclidean
    edge lengths, then runs Dijkstra from each source index.

    Parameters:
        vert (np.ndarray): shape (n_vert, 3) vertex coordinates
        triv (np.ndarray): shape (n_faces, 3) triangle vertex indices
        source_indices (array-like): indices of source vertices on which distances are computed

    Returns:
        np.ndarray: shape (n_sources, n_vert) geodesic distances from each source to all vertices
    """
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
    """Compute Gaussian indicator functions from geodesic distances.

    For each FPS point on N (and its match on M), compute geodesic distance fields and
    convert them to indicator functions via a Gaussian of variance `variance`.

    Parameters:
        v_M (np.ndarray): shape (M.n_vert, 3) vertex coordinates on M
        v_N (np.ndarray): shape (N.n_vert, 3) vertex coordinates on N
        f_M (np.ndarray): shape (M.n_faces, 3) triangles on M
        f_N (np.ndarray): shape (N.n_faces, 3) triangles on N
        fps_indices (np.ndarray): shape (n_samples,) FPS vertex indices on N
        matches (np.ndarray): shape (N.n_vert,) mapping N vertex → M vertex index
        variance (float): Gaussian variance parameter controlling indicator spread

    Returns:
        (np.ndarray, np.ndarray):
            - G: shape (M.n_vert, n_samples) indicator functions on M
            - F: shape (N.n_vert, n_samples) indicator functions on N
    """

    # Get corresponding points on M
    fps_matches_M = [matches[idx] for idx in fps_indices]

    print(f"  Computing geodesic distances on N...")
    geo_dists_N = _compute_geodesic_distances_mesh(v_N, f_N, fps_indices)

    print(f"  Computing geodesic distances on M...")
    geo_dists_M = _compute_geodesic_distances_mesh(v_M, f_M, fps_matches_M)

    # Convert to indicator functions
    F = np.exp(-0.5 * variance * geo_dists_N.T**2)
    G = np.exp(-0.5 * variance * geo_dists_M.T**2)

    return G, F