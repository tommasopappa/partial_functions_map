import torch
import numpy as np
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

ALMOST_ZERO = 1e-10

def optimize_C(M : ManifoldMesh, N : ManifoldMesh, W, func_M, func_N, C_init, v, est_rank, opts : Options):
    A = N.evecs.T @ (N.S.unsqueeze(1) * func_N)
    B = M.evecs.T @ ((N.S * v).unsqueeze(1) * func_M)

    # Create vector d for orthogonality constraint
    d = torch.zeros(opts.n_eigen)
    d[:est_rank] = 1

    if C_init is None:
        C_init = (torch.max(W) - W) / torch.max(W)

    C = torch.nn.Parameter(C_init)
    optimizer = torch.optim.Adam([C], lr=opts.C_lr)
    for iter in range(opts.C_max_iter):
        optimizer.zero_grad()
        loss = C_loss(A, B, C, d, W, opts)
        loss.backward()
        optimizer.step()

        if iter == 0 or (iter + 1) % 200 == 0:
            print(f"  Iter {iter+1}/{opts.C_max_iter}, Loss: {loss.item():.6f}")

    return C.detach().clone()

def C_loss(A, B, C, d, W, opts: Options):
    # Data term: ||CA - B||_{2,1}
    diff = C @ A - B
    data_term = l21_norm(diff)

    # Slanted diagonal term
    mask_term = torch.sum((C * W)**2)

    # Orthogonality terms
    CtC = C.T @ C
    off_diagonal_term = torch.sum(CtC**2) - torch.sum(torch.diag(CtC)**2)
    diagonal_term = torch.sum((torch.diag(CtC) - d)**2)

    return data_term + opts.mu3 * mask_term + opts.mu4 * off_diagonal_term + opts.mu5 * diagonal_term

def l21_norm(matrix):
    return torch.sum(torch.sqrt(torch.sum(matrix**2, dim=0) + ALMOST_ZERO))

def estimate_rank(M : ManifoldMesh, N : ManifoldMesh):
    return torch.sum(N.evals - torch.max(M.evals) <= 0)

def create_slanted_diagonal_mask(est_rank, opts: Options):
    k = opts.n_eigen
    W = torch.zeros((k, k), dtype=torch.float32)
    slope = est_rank / k if est_rank > 0 else 1.0
    direction = np.array([1, slope])
    direction = direction / np.linalg.norm(direction)

    for i in range(k):
        for j in range(k):
            # Distance from slanted diagonal
            point = np.array([i+1, j+1])  # 1-indexed as in paper
            origin = np.array([1, 1])
            # Cross product for 2D: extend to 3D then take magnitude of z component
            cross = np.abs(np.cross(np.append(direction, 0),
                                  np.append(point - origin, 0)))
            dist=np.abs(cross[2])
            W[i, j] = np.exp(-opts.mask_sigma * np.sqrt(i**2 + j**2)) * dist

    return W.to(opts.device)