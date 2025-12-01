import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

def run_icp_partial_torch_batched(M : ManifoldMesh, N : ManifoldMesh, C_init, est_rank, opts: Options):
    C = C_init[:, :est_rank]
    X = N.evecs[:, :est_rank].T * N.S.unsqueeze(0)
    Y = M.evecs.T * M.S.unsqueeze(0)

    for i in range(opts.max_icp_iters):
        CX = C @ X

        matches = torch.zeros(N.n_vert, dtype=torch.long, device=opts.device)
        for start_idx in range(0, N.n_vert, opts.icp_batch_size):
            end_idx = min(start_idx + opts.icp_batch_size, N.n_vert)
            batch_CX = CX[:, start_idx:end_idx]

            # batch_dists = torch.sum((batch_CX.unsqueeze(2) - Y.unsqueeze(1))**2, dim=0)
            batch_dists = distance_matrix(batch_CX.T, Y.T)
            batch_matches = torch.argmin(batch_dists, dim=1)
            matches[start_idx:end_idx] = batch_matches

        # Update C
        YM = Y[:, matches]
        U, _, Vt = torch.linalg.svd(X @ YM.T, full_matrices=False)
        C = (U @ Vt[:est_rank, :]).T

        if i == 0 or (i + 1) % 10 == 0:
            with torch.no_grad():
                err = torch.mean(torch.norm(C @ X - YM, dim=0)).item()
                print(f"  ICP iter {i+1}, MSE: {err:.4e}")

    C_full_shape = torch.zeros((opts.n_eigen, opts.n_eigen), dtype=torch.float32, device=opts.device)
    C_full_shape[:, :est_rank] = C
    return C_full_shape, matches

def distance_matrix(X, Y):
    """Compute squared Euclidean distance matrix between two sets of points."""
    X_norm = (X**2).sum(dim=1).unsqueeze(1)
    Y_norm = (Y**2).sum(dim=1).unsqueeze(0)
    dist = X_norm + Y_norm - 2.0 * (X @ Y.T)
    return dist