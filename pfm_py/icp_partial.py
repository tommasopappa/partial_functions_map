import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

def run_icp_partial(M : ManifoldMesh, N : ManifoldMesh, C_init, est_rank, opts: Options):
    """Run spectral ICP (iterative closest point) for partial-to-whole alignment.

    Aligns the truncated functional map `C` by alternating nearest-neighbor matching in the
    spectral domain and Procrustes updates (via SVD). 
    For every vertex of N, we find
    a corresponding vertex of M via nearest-neighbor matching in the spectral coordinate space
    of M. Points on N receive spectral coordinates of M via pushing forward with C.
    Using these matches, we refine C via a Procrustes update that finds the best orthogonal map 
    aligning push-forward coordinates of N to those of the matched vertices on M.

    Parameters:
        M (ManifoldMesh): Full target mesh with `evecs`, `S`, `n_vert`
        N (ManifoldMesh): Partial source mesh with `evecs`, `S`, `n_vert`
        C_init (torch.Tensor): Initial functional map, shape (opts.n_eigen, opts.n_eigen)
        est_rank (int): Truncation rank r for ICP update (columns used from N and rows in C)
        opts (Options): Options and hyperparameters.

    Returns:
        (torch.Tensor, torch.Tensor):
            - C_full_shape: shape (opts.n_eigen, opts.n_eigen) with updated first `est_rank` columns
            - matches: shape (N.n_vert,), indices of nearest neighbors on M for each N vertex
    """
    # Truncate to the first `est_rank` eigenfunctions of N, because Procrustes finds a full rank matrix 
    # and we want to restrict it `est_rank`.
    C = C_init[:, :est_rank]
    # X: spectral coordinates of N's truncated basis (est_rank × N.n_vert)
    # The i-th column of X is the spectral representation of the indicator function at vertex i of N
    X = N.evecs[:, :est_rank].T * N.S.unsqueeze(0)
    # Y: spectral coordinates of M's basis (opts.n_eigen × M.n_vert)
    # The i-th column of Y is the spectral representation of the indicator function at vertex i of M
    Y = M.evecs.T * M.S.unsqueeze(0)

    for i in range(opts.max_icp_iters):
        CX = C @ X # push forward spectral coordinates of N to M shape (opts.n_eigen, N.n_vert)

        # For every vertex P of N, find the corresponding vertex P' of M via
        # selecting the M-vertex with spectral coordinates closest to the 
        # pushed-forward spectral coordinates of P (CX[:, i])
        matches = torch.zeros(N.n_vert, dtype=torch.long, device=opts.device)
        # We partition the vertices of N into batches because computing the full distance matrix can be very large.
        for start_idx in range(0, N.n_vert, opts.icp_batch_size):
            end_idx = min(start_idx + opts.icp_batch_size, N.n_vert)
            batch_CX = CX[:, start_idx:end_idx]

            # Compute squared distances between `CX` batch and all Y columns (M vertices)
            batch_dists = distance_matrix(batch_CX.T, Y.T)
            batch_matches = torch.argmin(batch_dists, dim=1)
            matches[start_idx:end_idx] = batch_matches

        YM = Y[:, matches] # for every N-vertex P, spectral coords of matching vertex P' on M. Shape: (opts.n_eigen, N.n_vert)
        # Refine C by aligning CX (push-forward coordinates of N) to YM (coordinates of matched vertices on M)
        # Procrustes update: solve argmin_C || C X - Y_M ||_F via SVD of X Y_M^T
        U, _, V_T = torch.linalg.svd(X @ YM.T, full_matrices=False)
        C = (U @ V_T[:est_rank, :]).T

        if i == 0 or (i + 1) % 10 == 0:
            with torch.no_grad():
                err = torch.mean(torch.norm(C @ X - YM, dim=0)).item()
                print(f"  ICP iter {i+1}, MSE: {err:.4e}")

    # Return full-sized C via padding with 0 columns
    C_full_shape = torch.zeros((opts.n_eigen, opts.n_eigen), dtype=torch.float32, device=opts.device)
    C_full_shape[:, :est_rank] = C
    return C_full_shape, matches

def distance_matrix(X, Y):
    """Compute squared Euclidean distance matrix between two point sets.

    Parameters:
        X (torch.Tensor): shape (n_x, dim), row-wise points
        Y (torch.Tensor): shape (n_y, dim), row-wise points

    Returns:
        torch.Tensor: shape (n_x, n_y), with entry (i,j) = ||X[i] - Y[j]||^2
    """
    X_norm = (X**2).sum(dim=1).unsqueeze(1)
    Y_norm = (Y**2).sum(dim=1).unsqueeze(0)
    dist = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return dist