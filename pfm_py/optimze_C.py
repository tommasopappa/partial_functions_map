"""Optimize functional map C for partial shape matching.

This module estimates a functional map C between spectral bases of the partial mesh N
and full mesh M. The optimization uses:

1. **Optimization variable**: C in ℝ^(n_eigen × n_eigen)
2. **Data term**: L2,1 norm of descriptor alignment in spectral coordinates
3. **Regularization**: Slanted-diagonal mask and near-orthogonality of C
4. **Optimization**: Adam optimizer with early stopping on total loss
"""

from pfm_py.optimize_v import l21_norm
import torch
import numpy as np
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

ALMOST_ZERO = 1e-10

def optimize_C(M : ManifoldMesh, N : ManifoldMesh, W, func_M, func_N, C_init, v, est_rank, opts : Options):
    """Optimize functional map C via gradient descent.

    Finds C that maps spectral coefficients of descriptors on N to those on M while
    enforcing slanted-diagonal structure and near-orthogonality.

    Spectral coordinate conversion:
    A function f is projected to spectral coefficients via c = evecs.T @ (S * f),
    i.e., c_j = ⟨f, φ_j⟩_L2 (see ManifoldMesh.scalar_product).

    Args:
        M (ManifoldMesh): Full target mesh
        N (ManifoldMesh): Partial source mesh
        W (torch.Tensor): Slanted diagonal mask, shape (n_eigen, n_eigen)
        func_M (torch.Tensor): Descriptor functions on M, shape (M.n_vert, feat_dim)
        func_N (torch.Tensor): Descriptor functions on N, shape (N.n_vert, feat_dim)
        C_init (torch.Tensor | None): Optional initialization for C, shape (n_eigen, n_eigen)
        v (torch.Tensor): Soft membership on M, shape (M.n_vert,)
        est_rank (torch.Tensor): Estimated rank for diagonal target, scalar tensor
        opts (Options): Hyperparameters and options

    Returns:
        torch.Tensor: Optimized functional map C, shape (n_eigen, n_eigen)
    """
    # func_N_spectral: project func_N to spectral coordinates on N
    func_N_spectral = N.evecs.T @ (N.S.unsqueeze(1) * func_N)
    # func_M_spectral: project membership-weighted (softly restricted) func_M to spectral coordinates on M
    func_M_spectral = M.evecs.T @ ((M.S * v).unsqueeze(1) * func_M)

    # Create vector d for diagonal (rank) target in orthogonality constraint
    d = torch.zeros(opts.n_eigen, dtype=torch.float32, device=opts.device)
    d[:est_rank] = 1

    if C_init is None:
        # Initialize C to the "complement" of the mask to favor a slanted diagonal structure
        C_init = (torch.max(W) - W) / torch.max(W)

    C = torch.nn.Parameter(C_init)
    optimizer = torch.optim.Adam([C], lr=opts.C_lr)
    
    # Early stopping variables
    best_loss = None
    best_C = None
    patience_counter = 0
    
    for iter in range(opts.C_max_iter):
        optimizer.zero_grad()
        loss = C_loss(func_N_spectral, func_M_spectral, C, d, W, opts)
        loss.backward()
        optimizer.step()

        if iter == 0 or (iter + 1) % 200 == 0:
            print(f"  Iter {iter+1}/{opts.C_max_iter}, Loss: {loss.item():.6f}")
        
        # Early stopping: terminate if loss doesn't improve for patience iterations
        if opts.early_stopping:
            if best_loss is None or (best_loss - loss.item()) / max(abs(best_loss), ALMOST_ZERO) > opts.early_stopping_tol:
                best_loss = loss.item()
                best_C = C.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= opts.patience_iters:
                print(f"  Early stopping at iter {iter+1}: Loss has not decreased for {opts.patience_iters} iterations")
                break

    return best_C if best_C is not None else C.detach().clone()

def C_loss(func_N_spectral, func_M_spectral, C, d, W, opts: Options):
    r"""
    Compute the total loss for functional map optimization.

    Combines three weighted loss terms:
    1. **Data term**: L2,1 norm measuring descriptor alignment in spectral coordinates
       L_data = || C @ func_N_spectral - func_M_spectral ||_{2,1}
    2. **Slanted diagonal term**: Penalizes coefficients away from the preferred slanted diagonal
       L_mask = || C ⊙ W ||_F^2
    3. **Orthogonality terms**: Encourage `C^T C` to be close to a diagonal with target `d`
       L_orth_off = ∑_{i≠j} (C^T C)_{ij}^2
       L_orth_diag = || diag(C^T C) - d ||_2^2

    Args:
        func_N_spectral (torch.Tensor): Shape (n_eigen, feat_dim). Spectral coefficients of descriptors on N.
        func_M_spectral (torch.Tensor): Shape (n_eigen, feat_dim). Spectral coefficients of membership-weighted descriptors on M.
        C (torch.Tensor): Shape (n_eigen, n_eigen). Functional map from N's eigenspace to M's eigenspace.
        d (torch.Tensor): Shape (n_eigen,). Binary target for diagonal of `C^T C` (first `rank` entries set to 1).
        W (torch.Tensor): Shape (n_eigen, n_eigen). Slanted diagonal mask (larger values penalize off-diagonal entries).
        opts (Options): Hyperparameters and options.

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Data term: compare mapped descriptors C @ func_N_spectral with target descriptors func_M_spectral
    diff = C @ func_N_spectral - func_M_spectral
    data_term = l21_norm(diff)

    # Slanted diagonal term: penalize coefficients away from the mask
    mask_term = torch.sum((C * W)**2)

    # Orthogonality terms: push C^T C toward a diagonal with target d
    CtC = C.T @ C  # Gram matrix of columns of C
    off_diagonal_term = torch.sum(CtC**2) - torch.sum(torch.diag(CtC)**2)
    diagonal_term = torch.sum((torch.diag(CtC) - d)**2)

    return data_term + opts.mu3 * mask_term + opts.mu4 * off_diagonal_term + opts.mu5 * diagonal_term

def estimate_rank(M : ManifoldMesh, N : ManifoldMesh):
    """Estimates the rank of the functional map from partial mesh N to full mesh M.
    Counts how many eigenvalues of N fall below max(M.evals); this rank is then used
    to set d as a binary vector with d[:rank] = 1 (target diagonal in C^T C). See the PFM paper for details.
    """
    return torch.sum((N.evals - torch.max(M.evals)) < 0)

def create_slanted_diagonal_mask(est_rank, opts: Options):
    """Create a slanted diagonal mask W that favors a low-rank slanted diagonal structure in C.
    W acts as a prior for the functional map matrix C. We want C to resemble the "complement" of W,
    meaning that the component-wise product of C and W should be small.
    See the PFM paper for details on motivation / construction of W."""
    k = opts.n_eigen
    W = torch.zeros((k, k), dtype=torch.float32)
    # slope of slanted diagonal in the (i,j) index space of C, determined by the estimated rank
    slope = est_rank.item() / k if est_rank > 0 else 1.0
    direction = np.array([1, slope]) # direction vector of the slanted diagonal
    direction = direction / np.linalg.norm(direction)

    for i in range(k):
        for j in range(k):
            # Point corresponding to entry (i, j) in C
            point = np.array([i+1, j+1]) # 1-indexed as in paper
            origin = np.array([1, 1]) # coords of top-left entry of C

            # Cross product for 2D: extend to 3D then take magnitude of z component
            cross = np.abs(np.cross(np.append(direction, 0),
                                    np.append(point - origin, 0)))
            dist=np.abs(cross[2]) # distance of point to slanted diagonal

            # Weight by combination of distance to slanted diagonal
            # and radial decay (allow larger distance to slanted diagonal
            # for components far from the top-left corner)
            W[i, j] = np.exp(-opts.mask_sigma * np.sqrt(i**2 + j**2)) * dist

    return W.to(opts.device)