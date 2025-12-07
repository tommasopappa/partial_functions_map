import torch
import robust_laplacian
import scipy.sparse.linalg as sla
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import dijkstra

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
        self.area = torch.sum(self.S).item()

        if compute_geo:
            self._compute_geometry()

    def compute_fpfh_features(self, opts: Options):
        radius = 0.04 * self.area
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vert.numpy(force=True))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=radius*2))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        return torch.tensor(fpfh.data.T, dtype=torch.float32, device=opts.device)

    def compute_descriptors(self, opts: Options):
        """Compute descriptors based on the descriptor_type in options."""
        if opts.descriptor_type.lower() == "shot":
            return self.compute_shot_descriptors(opts)
        elif opts.descriptor_type.lower() == "fpfh":
            return self.compute_fpfh_features(opts)
        else:
            raise ValueError(f"Unknown descriptor type: {opts.descriptor_type}. Choose 'shot' or 'fpfh'.")
        
    def compute_shot_descriptors(self, opts: Options, radius=None, n_bins=10,
                                 min_neighbors=10, local_rf_radius=None, query_idx=None):
        """Compute SHOT descriptors for vertices."""
        from pfm_py.shot import SHOTParams, SHOTDescriptor
        
        vertices = self.vert.numpy(force=True)
        faces = self.triv.numpy(force=True)
        # ------- 0. Compute scale-aware radius (if not provided) -------
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
        total_area = np.sum(triangle_areas)

        # If radius not provided, compute automatically from total mesh area
        if radius is None:
            radius = opts.shot_radius_scale * np.sqrt(total_area)
            print(f"[SHOT] Using auto-scaled radius = {radius:.6f} (scale={opts.shot_radius_scale})")

        # Local RF radius default (scale factor comes from options)
        if local_rf_radius is None:
            local_rf_radius = radius * opts.shot_local_rf_factor
        normals = None
        
        # ------- 1. Normal vectors (with direction consistency) -------
        if faces is not None:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

            # Key: First unify triangle normal directions, then recompute vertex normals
            mesh.orient_triangles()                # Make triangle normals consistent
            mesh.compute_vertex_normals()          # Update vertex normals accordingly

            normals = np.asarray(mesh.vertex_normals, dtype=float)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            pcd.estimate_normals()
            # Key: Make local normal directions consistent (k can be adjusted, e.g., 20–30)
            pcd.orient_normals_consistent_tangent_plane(k=20)

            normals = np.asarray(pcd.normals, dtype=float)

        # ------- 2. SHOT parameters (using paper-based parameter structure) -------
        params = SHOTParams(
            radius=radius,
            localRFradius=local_rf_radius,
            bins=n_bins,
            doubleVolumes=True,
            useInterpolation=True,
            useNormalization=True,
            minNeighbors=min_neighbors
        )

        # ------- 3. Create SHOT descriptor instance -------
        shot = SHOTDescriptor(params)
        shot.set_data(vertices, normals, faces=faces)

        # ---- DEBUG: print neighbor count at some sample point ----
        # Select a test point, e.g., vertex 5
        idx_test = min(5, len(vertices) - 1)
        ni, dists = shot.nearest_neighbors_with_dist(idx_test, radius)
        print(f"[DEBUG SHOT] radius={radius:.4f}, neighbors for point {idx_test} = {len(ni)}")

        # ------- 4. Compute all descriptors -------
        if query_idx is None:
            desc_all = shot.describe_all()
        else:
            query_idx = np.asarray(query_idx, dtype=int)
            desc_all = shot.describe_all(query_idx=query_idx)

        return torch.tensor(desc_all, dtype=torch.float32, device=opts.device)

   

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
    
    def compute_geodesic_matrix(self):
        vertices, faces = self.vert.numpy(force=True), self.triv.numpy(force=True)
        n = self.n_vert
        edges = set()
        for face in faces:
            edges.add(tuple(sorted([face[0], face[1]])))
            edges.add(tuple(sorted([face[1], face[2]])))
            edges.add(tuple(sorted([face[2], face[0]])))

        graph = np.full((n, n), np.inf)
        for i, j in edges:
            dist = np.linalg.norm(vertices[i] - vertices[j])
            graph[i, j] = dist
            graph[j, i] = dist

        return dijkstra(graph, directed=False)