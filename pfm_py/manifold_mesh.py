import torch
import robust_laplacian
import scipy.sparse.linalg as sla
import numpy as np
import os
import open3d as o3d
from scipy.sparse.csgraph import dijkstra

from pfm_py.options import Options
from pfm_py import dino as dino_module, dinov3 as dinov3_module

ALMOST_ZERO = 1e-10

class ManifoldMesh:
    """Represents a compact Riemannian 2-manifold approximated via discrete triangulation.
    All functions on the mesh are represented as arrays of scalar values at vertices and are assumed to behave 
    approximately linearly on each triangle.

    Core attributes provide:
        - Geometric information: vertex coordinates, triangulation, surface area
        - Spectral basis: eigenvalues and eigenfunctions of Laplace-Beltrami operator
        - Mass matrix: area weights for integration and inner products
        - Riemannian metric: first fundamental form coefficients (optional)
    
    The space L2(M) of square-integrable functions on the manifold M is endowed with the inner product
    ⟨f, g⟩_L2 = ∫_M f(x) * g(x) dA. The eigenfunctions of the Laplace-Beltrami operator on M form an 
    orthonormal basis of L_2(M). We work in the subspace of L_2(M) spanned by the `n_eigen`
    lowest-eigenvalue eigenfunctions. Projecting a function into this subspace yields a good low-dimensional
    approximation. The coefficients of f w.r.t eigenfunction φ_i is given by ⟨f, φ_i⟩_L2.

    The mass matrix S is a diagonal matrix of shape (n_vert, n_vert); we only store its diagonal as a vector of 
    shape (n_vert,). Integrals of scalar functions can be approximated via ∫_M f dA ≈ Σ_i f[i] * S[i].
    
    Attributes:
        vert (torch.Tensor): Vertex coordinates, shape (n_vert, 3)
        triv (torch.Tensor): Triangle indices, shape (n_faces, 3)
        n_vert (int): Number of vertices
        evals (torch.Tensor): Eigenvalues of Laplace-Beltrami (truncated), shape (n_eigen,).
                             Forms the basis for spectral approximation.
        evecs (torch.Tensor): Eigenfunctions (orthonormal basis for L2(M)), shape (n_vert, n_eigen).
                             Each column is a mass-normalized eigenfunction. These form an
                             orthonormal basis for the L2 function space on the mesh.
        S (torch.Tensor): Mass matrix diagonal (area elements), shape (n_vert,).
                         Used for numerical quadrature and L2 inner products.
        area (float): Total surface area of the mesh
        E, F, G (torch.Tensor): First fundamental form coefficients (Riemannian metric tensor),
                               shape (n_faces,). Only computed if compute_geo=True.
        det (torch.Tensor): Metric determinant (area scaling factor), shape (n_faces,).
                           Only computed if compute_geo=True.
    """
    def __init__(self, vertices, triangles, opts: Options, compute_geo=False):
        """Initialize a ManifoldMesh from vertex and triangle arrays.
        
        Constructs a triangulated mesh and precomputes spectral information:
        computes the Laplace-Beltrami operator and its first opts.n_eigen eigenpairs,
        which provide an orthonormal basis for function space on the mesh. All data
        is transferred to the device specified in opts.
        
        The initialization also:
        - Computes the mass matrix (area elements for integration)
        - Mass-normalizes eigenfunctions for orthonormality
        - Enforces sign consistency on eigenfunctions
        - Optionally computes first fundamental form coefficients
        
        Parameters
        ----------
        vertices : np.ndarray, shape (n_vert, 3)
            Vertex coordinates in 3D space.
        triangles : np.ndarray, shape (n_faces, 3)
            Triangle indices, 0-based, defining the mesh topology.
        opts : Options
            Options object specifying device, n_eigen (number of eigenfunctions to compute).
        compute_geo : bool, optional
            If True, compute first fundamental form coefficients E, F, G (Riemannian metric).
            Default False.
        """
        self.vert = torch.tensor(vertices, dtype=torch.float32, device=opts.device)    
        self.triv = torch.tensor(triangles, dtype=torch.long, device=opts.device)
        self.n_vert = vertices.shape[0]

        # Compute Laplace-Beltrami operator L and mass matrix S
        L, S = robust_laplacian.mesh_laplacian(vertices, triangles, mollify_factor=1e-5)
        L, S = L.tocsr(), S.tocsr() # CSR for efficiency
        
        # np.random.seed(42)
        # v0 = np.random.randn(L.shape[0])

        evals, evecs = sla.eigsh(L, k=opts.n_eigen, M=S, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15) # type: ignore
        
        for i in range(opts.n_eigen): # Normalize eigenvectors w.r.t mass matrix
            evecs[:, i] = evecs[:, i] / np.sqrt(evecs[:, i].T @ S @ evecs[:, i])
        
        # Enforce sign consistency: for each eigenvector, ensure the highest-absolute-value
        # component is positive. This resolves arbitrary sign flips from eigendecomposition.
        for i in range(opts.n_eigen):
            max_abs_idx = np.argmax(np.abs(evecs[:, i]))
            if evecs[max_abs_idx, i] < 0:
                evecs[:, i] = -evecs[:, i]

        self.evals = torch.tensor(evals, dtype=torch.float32, device=opts.device)
        self.evecs = torch.tensor(evecs, dtype=torch.float32, device=opts.device)
        self.S = torch.tensor(S.diagonal(), dtype=torch.float32, device=opts.device)
        self.area = torch.sum(self.S).item()

        if compute_geo:
            self.compute_geometry()

    def compute_fpfh_descriptors(self, opts: Options):
        """Compute FPFH (Fast Point Feature Histogram) descriptors for all vertices.
        
        FPFH descriptors are local geometric features computed in a two-step process:
        first computing PFH (Point Feature Histogram) for each point and its neighbors,
        then computing FPFH by combining PFH values.
        
        Args:
            opts: Options object containing device specification.
        
        Returns:
            Descriptor matrix of shape (n_vert, feat_dim), where each row contains
            the FPFH feature vector for the corresponding vertex. Values represent
            distributions of angles and distances between vertex normals and positions
            in a local neighborhood.
        """
        radius = 0.04 * self.area
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vert.numpy(force=True))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=radius*2))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamRadius(radius=radius))
        return torch.tensor(fpfh.data.T, dtype=torch.float32, device=opts.device)

    def compute_descriptors(self, opts: Options):
        """Compute local geometric descriptors for all vertices based on descriptor_type option.
        
        Supports multiple descriptor types: SHOT (Signature of Histograms of Orientations),
        FPFH (Fast Point Feature Histogram), DINO and DINOv3 (learned deep features).
        Functions are represented as descriptor values at each vertex.
        
        Args:
            opts: Options object with descriptor_type field specifying which descriptor to use.
        
        Returns:
            Descriptor matrix of shape (n_vert, feat_dim), where each row represents
            functions (descriptors) at one vertex. feat_dim depends on descriptor type.
        
        Raises:
            ValueError: If opts.descriptor_type is not one of the supported types.
        """
        if opts.descriptor_type.lower() == "shot":
            return self.compute_shot_descriptors(opts)
        elif opts.descriptor_type.lower() == "fpfh":
            return self.compute_fpfh_descriptors(opts)
        elif opts.descriptor_type.lower() == "dino":
            verts = self.vert.clone().detach()
            faces = self.triv.clone().detach()
            feats, n_missing = dino_module.compute_dino_features(verts, faces)
            self.dino_n_missing = n_missing
            return torch.tensor(feats, dtype=torch.float32, device=opts.device)
        elif opts.descriptor_type.lower() == "dinov3":
            verts = self.vert.clone().detach()
            faces = self.triv.clone().detach()
            feats, n_missing = dinov3_module.get_shape_dinov3_features(verts, faces)
            self.dino_n_missing = n_missing
            return torch.tensor(feats, dtype=torch.float32, device=opts.device)
        else:
            raise ValueError(f"Unknown descriptor type: {opts.descriptor_type}. Choose 'shot', 'fpfh', 'shot_fpfh', 'dino' or 'dinov3'.")
        
    def compute_shot_descriptors(self, opts: Options, radius=0.05, n_bins=10,
                                 min_neighbors=10, local_rf_radius=None, query_idx=None):
        """Compute SHOT (Signature of Histograms of Orientations) descriptors for vertices.
        
        SHOT descriptors encode local geometric shape information by building histograms
        of angles and distances in a spherical neighborhood around each point. The descriptor
        is computed in a local reference frame aligned with the surface normal.
        
        Args:
            opts: Options object containing device specification.
            radius: Search radius for neighborhood in world coordinates. Default 0.05.
            n_bins: Number of bins for histogram quantization. Default 10.
            min_neighbors: Minimum number of neighbors required for a valid descriptor.
                          Default 10.
            local_rf_radius: Radius for local reference frame computation. If None,
                           defaults to 1.5 * radius.
            query_idx: If provided, only compute descriptors for these vertex indices.
                      If None, compute for all vertices. Shape (n_query,) or None.
        
        Returns:
            Descriptor matrix of shape (n_vert, feat_dim) or (n_query, feat_dim).
            Each row contains the SHOT histogram for one vertex, representing local
            geometric features at that location.
        """
        from pfm_py.shot import SHOTParams, SHOTDescriptor
        
        vertices = self.vert.numpy(force=True)
        faces = self.triv.numpy(force=True)
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
        if local_rf_radius is None:
            local_rf_radius = radius * 1.5

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

    def compute_geometry(self):
        """Compute first fundamental form (Riemannian metric) coefficients for each triangle.
        
        For each triangle with vertices x1, x2, x3 parameterized by coordinates (u, v),
        the first fundamental form is the 2×2 Riemannian metric tensor:
            [E  F]
            [F  G]
        where E, F, G are computed from the edge vectors e1 = x2 - x1 and e2 = x3 - x1:
            E = e1 · e1  (squared length of first edge)
            F = e1 · e2  (dot product of edges)
            G = e2 · e2  (squared length of second edge)
        
        We also store det = sqrt(E*G - F²), the square root of the metric tensor determinant.
        
        Stores results as instance attributes:
            self.E: shape (n_faces,), metric component (squared first edge length)
            self.F: shape (n_faces,), metric component (edge cross term)
            self.G: shape (n_faces,), metric component (squared second edge length)
            self.det: shape (n_faces,), sqrt(|E*G - F²|), the area scaling factor
        """
        i, j, k = self.triv[:, 0], self.triv[:, 1], self.triv[:, 2]
        x1, x2, x3 = self.vert[i], self.vert[j], self.vert[k]
        e1, e2 = x2 - x1, x3 - x1

        self.E = torch.sum(e1 * e1, dim=1)
        self.F = torch.sum(e1 * e2, dim=1)
        self.G = torch.sum(e2 * e2, dim=1)
        self.det = torch.sqrt(torch.abs(self.E * self.G - self.F * self.F) + ALMOST_ZERO)

    def integrate_function(self, f):
        """Compute the integral of a scalar function over the mesh.
        
        Approximates the surface integral ∫_M f dA using quadrature with vertex-based
        basis functions. The mass matrix diagonal (self.S) stores area elements,
        and the integral is computed as: ∫ f = Σ_i f[i] * S[i].

        Args:
            f: Scalar function represented as values at vertices, shape (n_vert,).
        
        Returns:
            Scalar float representing the integral of f over the mesh surface.
        """
        return torch.sum(f * self.S)
    
    def scalar_product(self, f, g):
        """Compute the L2 inner product of two scalar functions over the mesh.
        
        Computes ⟨f, g⟩_L2 = ∫_M f(x) * g(x) dA using quadrature:
        ⟨f, g⟩_L2 = Σ_i f[i] * g[i] * S[i]
        
        Args:
            f: First scalar function, shape (n_vert,).
            g: Second scalar function, shape (n_vert,).
        
        Returns:
            Scalar float representing the L2 inner product ⟨f, g⟩_L2.
        """
        return self.integrate_function(f * g)

    def partial_area(self, v):
        """Compute the soft weighted area of the mesh using a membership function.
        
        This is equivalent to computing the integral of the membership function v
        over the mesh: ∫_M v(x) dA.
        
        Args:
            v: Soft membership function, shape (n_vert,). Values should be in [0, 1]
               indicating the strength of membership at each vertex.
        
        Returns:
            Scalar float representing the weighted area: sum_i v[i] * S[i],
            where S[i] is the mass matrix diagonal (local area element) at vertex i.
        """
        return self.integrate_function(v)
    
    def compute_geodesic_matrix(self):
        """Compute geodesic distances between all pairs of vertices.
        
        Uses Dijkstra's algorithm on the mesh edge graph to compute shortest path
        distances along the surface. First constructs the graph by extracting edges
        from the triangle mesh and computing Euclidean distances along edges,
        then applies Dijkstra's algorithm to find geodesic distances.
        
        Returns:
            Geodesic distance matrix of shape (n_vert, n_vert), where each entry [i, j]
            contains the shortest path distance along the mesh surface from vertex i
            to vertex j. The matrix is symmetric and has zeros on the diagonal.
        """
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