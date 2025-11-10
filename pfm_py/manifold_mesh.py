import numpy as np
from numpy import ndarray

class ManifoldMesh:
    def __init__(self, vertices : ndarray, triangles : ndarray, k : int):
        """
        A simple container for mesh/manifold data.

        Parameters
        ----------
        vertices : np.ndarray, shape (n_vert, 3)
            Vertex coordinates.
        triangles : np.ndarray, shape (n_faces, 3)
            Triangle indices (0-based).
        k : int
            Number of eigenvectors/eigenvalues to compute
        """
        self.vert = vertices        # Vertex positions
        self.triv = triangles       # Triangle indices
        self.n_vert = vertices.shape[0]  # Number of vertices
        self.k = k

        self.compute_eigendecomposition()
        self.compute_S()

    # TODO
    def compute_eigendecomposition(self):
        """
        Computes the first k eigenvectors/eigenvalues of the Laplacian.
        
        evecs: np.ndarray, shape (n_vert, k), one column per eigenvector.

        evals: np.ndarray, shape (k).
        """
        self.evecs : ndarray = np.array([])
        self.evals : ndarray = np.array([])

    # TODO
    def compute_S(self):
        """
        Computes the diagonal mass density matrix S, represented as an ndarray of shape (n_vert) containing the diagonal elements. 
        """
        self.S : ndarray = np.array([])