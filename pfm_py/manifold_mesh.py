import torch
import robust_laplacian
import scipy.sparse.linalg as sla
import numpy as np
import open3d as o3d

from pfm_py.options import Options

ALMOST_ZERO = 1e-10

class ManifoldMesh:
    def __init__(self, vertices, triangles, opts: Options, compute_geo=False):
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
        self.vert = torch.tensor(vertices, dtype=torch.float32, device=opts.device)    
        self.triv = torch.tensor(triangles, dtype=torch.long, device=opts.device)
        self.n_vert = vertices.shape[0]

        L, S = robust_laplacian.mesh_laplacian(vertices, triangles, mollify_factor=1e-5)
        L, S = L.tocsr(), S.tocsr() # CSR for efficiency
        evals, evecs = sla.eigsh(L, k=opts.n_eigen, M=S, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15) # type: ignore
        for i in range(opts.n_eigen): # Normalize eigenvectors w.r.t mass matrix
            evecs[:, i] = evecs[:, i] / np.sqrt(evecs[:, i].T @ S @ evecs[:, i])
        self.evals = torch.tensor(evals, dtype=torch.float32, device=opts.device)
        self.evecs = torch.tensor(evecs, dtype=torch.float32, device=opts.device)
        self.S = torch.tensor(S.diagonal(), dtype=torch.float32, device=opts.device)
        self.area = torch.sum(self.S)

        if compute_geo:
            self._compute_geometry()

    def compute_fpfh_features(self, opts: Options):
        radius = 0.04 * self.area
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vert.numpy(force=True))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=radius*2))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        return torch.tensor(fpfh.data.T, dtype=torch.float32, device=opts.device)
        
    def _compute_geometry(self):
        i, j, k = self.triv[:, 0], self.triv[:, 1], self.triv[:, 2]
        x1, x2, x3 = self.vert[i], self.vert[j], self.vert[k]
        e1, e2 = x2 - x1, x3 - x1

        self.E = torch.sum(e1 * e1, dim=1)
        self.F = torch.sum(e1 * e2, dim=1)
        self.G = torch.sum(e2 * e2, dim=1)
        self.det = torch.sqrt(torch.abs(self.E * self.G - self.F * self.F) + ALMOST_ZERO)

        """"
        e3 = x3 - x2
        edges = torch.cat([e1, e2, e3])
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        v1, v2 = self.vert[edges[:, 0]], self.vert[edges[:, 1]]
        edge_lengths = torch.sqrt(torch.sum((v2 - v1)**2, dim=1) + ALMOST_ZERO)
        self.avg_edge_length = edge_lengths.mean()
        """

    def partial_area(self, v):
        return torch.sum(v * self.S)