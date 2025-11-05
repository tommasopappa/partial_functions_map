# pfm_py/mesh_utils.py
from __future__ import annotations
import numpy as np


# ---------- Basic geometry utilities ----------

def face_normals(V: np.ndarray, F: np.ndarray, unit: bool = True) -> np.ndarray:
    """
    Compute per-face normal vectors.

    Parameters
    ----------
    V : (n, 3) array
        Vertex coordinates.
    F : (m, 3) int array
        Triangle face indices.
    unit : bool, default=True
        If True, return normalized unit normals.

    Returns
    -------
    N : (m, 3) array
        Normal vector of each triangle face.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = np.cross(v1 - v0, v2 - v0)  # unnormalized face normal
    if unit:
        l = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N = N / l
    return N


def tri_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute area of each triangular face.

    Formula: A_f = 0.5 * || (v1 - v0) x (v2 - v0) ||

    Returns
    -------
    A : (m,) array
        Area of each face.
    """
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def vertex_areas_barycentric(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex area using barycentric rule:
    each face area is equally distributed to its three vertices.

    Often used for constructing a diagonal mass matrix.

    Returns
    -------
    va : (n,) array
        Area associated with each vertex.
    """
    A = tri_areas(V, F)
    n = V.shape[0]
    va = np.zeros(n, dtype=V.dtype)
    np.add.at(va, F[:, 0], A / 3.0)
    np.add.at(va, F[:, 1], A / 3.0)
    np.add.at(va, F[:, 2], A / 3.0)
    return va


def normalize_unit_area(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Scale mesh so that the total surface area ≈ 1.

    Useful to remove scale differences before computing Laplace–Beltrami spectra.

    Returns
    -------
    V_scaled : (n,3) array
        Scaled vertex positions.
    scale : float
        The scale factor applied to vertices.
    """
    A_total = float(tri_areas(V, F).sum())
    if A_total <= 0:
        raise ValueError("Mesh area is non-positive. Check face orientation or degeneracy.")
    scale = (1.0 / A_total) ** 0.5  # because area scales with length^2
    return V * scale, scale


def center_vertices(V: np.ndarray, method: str = "barycenter") -> np.ndarray:
    """
    Translate vertices so that the mesh is centered at the origin.

    Parameters
    ----------
    method : {'barycenter', 'median'}
        How to compute the center point.
        'barycenter' = mean of all vertices (default)
        'median' = median, more robust to outliers

    Returns
    -------
    V_centered : (n,3) array
        Centered vertex coordinates.
    """
    if method == "median":
        c = np.median(V, axis=0)
    else:
        c = V.mean(axis=0)
    return V - c


# ---------- Connectivity utilities ----------

def vertex_vertex_adjacency(F: np.ndarray, n_vertices: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute undirected vertex–vertex adjacency (edge list form).

    Returns pairs (I, J) such that an undirected edge i--j exists
    (no self-loops, no duplicates, always i < j).

    Returns
    -------
    I, J : arrays of shape (num_edges,)
        Indices of adjacent vertices.
    """
    if n_vertices is None:
        n_vertices = int(F.max()) + 1
    E = np.vstack([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]],
    ])
    # undirected, remove duplicates and self-loops
    E = np.sort(E, axis=1)
    E = E[E[:, 0] != E[:, 1]]
    E = np.unique(E, axis=0)
    return E[:, 0], E[:, 1]


def remove_isolated_vertices(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove isolated (unused) vertices that are not referenced by any face.

    Returns
    -------
    V_new : (n_new, 3)
        Filtered vertex array.
    F_new : (m, 3)
        Faces with updated vertex indices.
    idx_map : (n_new,)
        Mapping new_index -> old_index
    """
    used = np.zeros(len(V), dtype=bool)
    used[F.reshape(-1)] = True
    idx_map = np.flatnonzero(used)
    new_index = -np.ones(len(V), dtype=int)
    new_index[idx_map] = np.arange(len(idx_map))
    F_new = new_index[F]
    V_new = V[idx_map]
    return V_new, F_new, idx_map