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
