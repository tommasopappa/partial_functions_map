import numpy as np
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

ALMOST_ZERO = 1e-12

def optimize_v(M : ManifoldMesh, N : ManifoldMesh, func_M, func_N, C, opts : Options):
    """Optimize the soft membership function v via gradient descent.
    
    Finds the soft membership function v that best explains which regions of mesh M
    correspond to mesh N. The optimization solves:
    
        min_v L_data(v) + μ₁*L_area(v) + μ₂*L_MS(v)
    
    where v ∈ ℝ^(n_M) is unbounded, then projected into [0,1] via η(v) for evaluation.
    Uses Adam optimizer with early stopping (if enabled, terminates when loss does not 
    significantly decrease for opts.v_patience_iters consecutive iterations)
    
    Args:
        M: Full target mesh as ManifoldMesh
        N: Partial source mesh as ManifoldMesh
        func_M: Descriptor matrix on M, shape (M.n_vert, feat_dim)
        func_N: Descriptor matrix on N, shape (N.n_vert, feat_dim)
        C: Functional map matrix, shape (opts.n_eigen, opts.n_eigen)
        opts: Hyperparameters and options
    
    Returns:
        v_opt: Projected membership function, shape (M.n_vert,) with values in [0,1].
               v_opt[i] indicates the membership strength of vertex i of M in the partial shape.
    
    **Spectral coordinate conversion:**
    To represent functions in the spectral basis, functions are projected onto eigenfunctions.
    A function f's spectral coefficient for eigenfunction φ_j is computed as: c_j = ⟨f, φ_j⟩_L2,
    which is implemented as the dot product f · φ_j scaled by the mass matrix S (see ManifoldMesh.scalar_product).
    """
    # Precompute spectral terms (independent of v, reused in every loss evaluation)
    # func_N_pushforward: project func_N to spectral coordinates on N, then map through C to M's eigenspace
    func_N_pushforward = C @ N.evecs.T @ (N.S.unsqueeze(1) * func_N)
    # M_spectral_projector: projection operator that converts functions on M to spectral coordinates
    M_spectral_projector = M.evecs.T * M.S.unsqueeze(0)

    # Initialize v as constant one function on N, pushed forward through functional map C
    constant_one = torch.ones(N.n_vert, dtype=torch.float32, device=opts.device)
    v0 = M.evecs @ C @ N.evecs.T @ (N.S * constant_one)
    perturb = torch.ones_like(v0) # currently unused (all ones)

    v = torch.nn.Parameter(v0)
    optimizer = torch.optim.Adam([v], lr=opts.v_lr)
    
    # Early stopping mechanism
    # Stops optimization if loss hasn't improved for v_patience_iters iterations
    # This prevents overfitting and avoids wasting computation on plateaus
    best_loss = None
    best_v = None
    patience_counter = 0
    
    for i in range(opts.v_max_iter):
        optimizer.zero_grad()
        loss = v_loss(i, M, N, func_N_pushforward, M_spectral_projector, func_M, v, perturb, opts)
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % 100 == 0:
            print(f"  Iter {i+1}/{opts.v_max_iter}, Loss: {loss.item():.6f}")
        
        # Early stopping: terminate if loss doesn't significantly decrease for patience iterations
        if opts.early_stopping:
            if best_loss is None or (best_loss - loss.item()) / max(abs(best_loss), ALMOST_ZERO) > opts.early_stopping_tol:
                best_loss = loss.item()
                best_v = v.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= opts.patience_iters:
                print(f"  Early stopping at iter {i+1}: Loss has not decreased for {opts.patience_iters} iterations")
                break

    v_opt = best_v if best_v is not None else v.detach().clone()
    return eta(v_opt)

def v_loss(i, M : ManifoldMesh, N : ManifoldMesh, func_N_pushforward, M_spectral_projector, func_M, v, perturb, opts : Options):
    r"""
    Compute the total loss function for soft membership function optimization.
    
     Combines three weighted loss terms:
     1. **Data term**: L2,1 norm measuring descriptor alignment via functional map
         L_data = || func_N_pushforward - M_spectral_projector @ (η(v) ⊙ func_M) ||_{2,1}
     2. **Area term**: Quadratic penalty ensuring soft region covers appropriate area  
         L_area = (N.area - M.partial_area(η(v)))^2
     3. **Mumford-Shah regularization**: Edge-preserving total variation
    
    Args:
        M (ManifoldMesh): Full target mesh
        N (ManifoldMesh): Partial source mesh
        func_N_pushforward (torch.Tensor): Functional map applied to spectral coefficients of func_N.
                           Represents what M's descriptors should look like to match N,
                           in M's eigenspace. Shape (n_eigen_M, feat_dim).
        M_spectral_projector (torch.Tensor): Projection operator that converts vertex functions on M to
                            spectral coefficients in M's eigenbasis. Equivalent to
                            M.evecs.T @ (M.S * func). Shape (n_eigen_M, n_M).
        func_M (torch.Tensor): Descriptor functions on M, shape (n_M, feat_dim)
        v (torch.nn.Parameter): Unbounded membership function, shape (n_M,)
        perturb (torch.Tensor): Perturbation weights, shape (n_M,)
        opts (Options): Hyperparameters and options
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    bounded_v = eta(v) # maps v into [0,1]
    # Softly restrict descriptors on M to N using current soft membership function
    func_M_restrict = bounded_v.unsqueeze(1) * func_M
    # Project restricted descriptors to spectral coefficients and compare with func_N_pushforward
    diff = func_N_pushforward - M_spectral_projector @ func_M_restrict
    # Data term: L2,1 norm measures descriptor mismatch between partial and full mesh
    data_term = l21_norm(diff)

    # Area term: penalize mismatch between N's area and the weighted area on M
    area_term = (N.area - M.partial_area(bounded_v))**2

    reg_term = mumford_shah_cost(M, v, perturb, opts)

    # if i == 0 or (i + 1) % 100 == 0:
        # print(f"Data term: {data_term.item():.6f}")
        # print(f"Area term: {opts.mu1 * area_term.item():.6f}")
        # print(f"Mumford-Shah reg term: {opts.mu2 * reg_term.item():.6f}")
    return data_term + opts.mu1 * area_term + opts.mu2 * reg_term

def l21_norm(matrix):
    """Compute the L2,1 norm (mixed norm) of a matrix.
    
    For matrix X of shape (n, m), computes:
    
        ||X||_{2,1} = Σ_{j=1}^m ||X[:, j]||_2
    
    This is the sum of L2 norms of columns. Used as the data fidelity term because:
    - Robust to outliers (L2 norm per column, not per element)
    - Promotes sparsity in dimensions where all columns are small
    - Natural for multi-feature descriptors where each feature dimension is either
      well-matched or completely mismatched across samples
    
    Args:
        matrix: Tensor of shape (n, m) where m is the feature dimension
    
    Returns:
        Scalar L2,1 norm ≥ 0
    """
    return torch.sum(torch.sqrt(torch.sum(matrix**2, dim=0) + ALMOST_ZERO))

def eta(t):
    """Sigmoid-like transformation mapping ℝ into [0, 1].
    
    This allows optimization over unconstrained v while keeping
    the transformed membership function η(v) in the valid range [0, 1]. The function is given by
        η(t) = 0.5 * tanh(6(t - 0.5)) + 0.5.
    
    Args:
        t: Unbounded input, shape arbitrary (scalar, vector, or tensor)
    
    Returns:
        Projected values in [0, 1], same shape as input
    """
    return 0.5 * torch.tanh(6*(t - 0.5)) + 0.5

def mumford_shah_cost(M : ManifoldMesh, v, perturb, opts: Options):
    """Compute edge-preserving Mumford-Shah regularization for membership function.
    
    Implements a smooth approximation of the Mumford-Shah functional:
    
        L_MS = ∫_M |∇v| * ξ(v) dA
    
    where ξ(v) is a Gaussian approximation to the Dirac delta at tv_mean.
    This regularization penalizes gradients only near the membership transition, producing
    piecewise-smooth membership functions with sharp boundaries.
    For details/derivations, see the PFM paper.
    
    Args:
        M: Mesh with geometry (E, F, G, det) and triangulation (triv)
        v: Unbounded membership function, shape (M.n_vert,)
        perturb: Vertex weights (vertex-wise scaling), shape (M.n_vert,). Currently all ones.
        opts: Options and hyperparameters
    
    Returns:
        Scalar regularization cost ≥ 0. Penalizes gradients near the membership transition.
    """
    # Note that η(0.5) = 0.5, so the default value tv_mean = 0.5 makes sense despite this function
    # being passed the unbounded v instead of the bounded η(v). 
    tv_mean = opts.tv_mean
    tv_sigma = opts.tv_sigma * np.sqrt(M.area) # tv_sigma is scale-dependent, see doc in options.py

    # Gaussian approximation to Dirac delta: ξ(v) = exp(-(v-v_mean)²/(2*σ²))
    # This is localized around v ≈ v_mean (the target membership density)
    xi = torch.exp(-(v - tv_mean)**2 / (2 * tv_sigma**2))
    xi = xi * perturb

    # === NUMERICAL INTEGRATION SCHEME ===
    # Integrate ∫_M |∇v| * ξ(v) dA over the triangulated mesh using barycentric quadrature:
    #   ∫_M |∇v| * ξ(v) dA ≈ Σ_T (ξ₁ + ξ₂ + ξ₃)/6 * |∇v|_T
    # where T ranges over triangles, ξᵢ are ξ values at triangle vertices,
    # |∇v|_T is the gradient magnitude (constant on linear mesh).
    # The factor 1/6 = (1/3 from vertex averaging) * (1/2 from triangle integration weight).
    
    # Extract vertex indices and values for all triangles
    i, j, k = M.triv[:, 0], M.triv[:, 1], M.triv[:, 2]
    v1, v2, v3 = v[i], v[j], v[k]
    
    # Compute gradient magnitude on each triangle using Riemannian metric (E, F, G coefficients)
    # For linear v on triangle: ∇v = v_alpha * e1 + v_beta * e2
    # where e1, e2 are edge basis vectors. |∇v|² = v_alpha² * G - 2*v_alpha*v_beta * F + v_beta² * E
    v_alpha, v_beta = v2 - v1, v3 - v1
    norm_grad_v = v_alpha**2 * M.G - 2 * v_alpha * v_beta * M.F + v_beta**2 * M.E
    norm_grad_v = torch.sqrt(norm_grad_v + ALMOST_ZERO)
    
    # Optional area weighting: normalize by metric determinant
    if opts.mumford_shah_area_weighted:
        norm_grad_v = norm_grad_v / M.det

    # Barycentric quadrature: sum ξ over the three vertices of each triangle
    xi_sum = xi[i] + xi[j] + xi[k]  # Sum of ξ at triangle vertices
    
    # Mask out triangles where all vertices have negligible ξ (far from transition)
    mask = xi_sum > ALMOST_ZERO
    
    # Integrate: ∫ ≈ Σ_T (ξ₁ + ξ₂ + ξ₃)/6 * |∇v|_T
    cost = torch.sum(xi_sum[mask] * norm_grad_v[mask]) / 6.0
    return cost