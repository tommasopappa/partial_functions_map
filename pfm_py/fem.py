# pfm_py/fem.py
from __future__ import annotations
import numpy as np
from scipy import sparse
from .mesh_utils import tri_areas


def mass_matrix_barycentric(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    """
    Construct a diagonal (barycentric / lumped) mass matrix.

    Each triangle area A_f is equally distributed to its three vertices.
    For a vertex i, the diagonal entry is:
        M_ii = sum_{f incident to i} A_f / 3

    This is widely used in spectral geometry and functional maps:
    - simple
    - symmetric positive definite (SPD)
    - very efficient in storage and computation

    Parameters
    ----------
    V : (n, 3) array
        Vertex coordinates.
    F : (m, 3) int array
        Triangle vertex indices.

    Returns
    -------
    M : (n, n) csr_matrix
        Diagonal lumped mass matrix.
    """
    A = tri_areas(V, F)                # (m,)
    n = V.shape[0]

    # Repeat each face's vertices to build diagonal contributions
    I = np.hstack([F[:, 0], F[:, 1], F[:, 2]])
    J = I                              # diagonal entries only
    vals = np.hstack([A / 3.0, A / 3.0, A / 3.0])

    M = sparse.coo_matrix((vals, (I, J)), shape=(n, n))
    return M.tocsr()


def mass_matrix_full(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    """
    Construct the full (consistent) FEM mass matrix.

    For each triangle f = (i, j, k) with area A_f, we add the local matrix:

        (A_f / 12) * [[2, 1, 1],
                      [1, 2, 1],
                      [1, 1, 2]]

    to the global mass matrix M at rows/cols (i, j, k).

    This is the classical "consistent" mass matrix from linear FEM.
    It is slightly more accurate as an approximation of the continuous
    L2 inner product, but:
    - has off-diagonal entries
    - is a bit more expensive to store and use

    For most spectral methods / functional maps, the lumped (diagonal)
    version is sufficient and more convenient. Use this if:
    - you need a closer match to some existing MATLAB implementation
      based on the same formula
    - or you care about higher FEM consistency.

    Parameters
    ----------
    V : (n, 3) array
        Vertex coordinates.
    F : (m, 3) int array
        Triangle vertex indices.

    Returns
    -------
    M : (n, n) csr_matrix
        Full consistent mass matrix.
    """
    A = tri_areas(V, F)          # (m,)
    n = V.shape[0]
    m = F.shape[0]

    # For each face (i,j,k), we need a 3x3 block:
    # rows:    [i,i,i, j,j,j, k,k,k]
    # cols:    [i,j,k, i,j,k, i,j,k]
    rows = np.repeat(F, 3, axis=1).ravel()
    cols = np.tile(F, 3).ravel()

    # Local 3x3 pattern flattened row-major:
    # [[2,1,1],
    #  [1,2,1],
    #  [1,1,2]] -> [2,1,1, 1,2,1, 1,1,2]
    base = np.array([2, 1, 1,
                     1, 2, 1,
                     1, 1, 2], dtype=float)

    # For each face f, scale this pattern by (A_f / 12)
    # Result shape: (m * 9,)
    vals = ((A / 12.0)[:, None] * base[None, :]).ravel()

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return M.tocsr()