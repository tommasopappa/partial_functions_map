import numpy as np
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

ALMOST_ZERO = 1e-12

def optimize_v(M : ManifoldMesh, N : ManifoldMesh, func_M, func_N, C, opts : Options):
    A = C @ N.evecs.T @ (N.S.unsqueeze(1) * func_N)
    B = M.evecs.T * M.S.unsqueeze(0)
    v0 = M.evecs @ C @ np.transpose(N.evecs) @ (N.S.unsqueeze(1) * np.ones((N.n_vert, 1)))
    perturb = torch.ones_like(v0)

    v = torch.nn.Parameter(v0)
    optimizer = torch.optim.Adam([v], lr=opts.v_lr)
    for i in range(opts.v_max_iter):
        optimizer.zero_grad()
        loss = v_loss(M, N, A, B, func_M, v, perturb, opts)
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % 100 == 0:
            print(f"  Iter {i+1}/{opts.v_max_iter}, Loss: {loss.item():.6f}")

    v_opt = v.detach().clone()
    return eta(v_opt)

def v_loss(M : ManifoldMesh, N : ManifoldMesh, CA, PtS, func_M, v, perturb, opts : Options):
    tv = eta(v) # maps v into [0,1] 
    VG = tv.unsqueeze(1) * func_M
    diff = CA - PtS @ VG
    data_term = l21_norm(diff)

    area_term = (N.area - M.partial_area(tv))**2

    tv_mean = N.area / M.area
    tv_sigma = M.avg_edge_length
    reg_term = mumford_shah_cost(M, v, perturb, opts, tv_mean, tv_sigma)

    return data_term + opts.mu1 * area_term + opts.mu2 * reg_term

def l21_norm(matrix):
    return torch.sum(torch.sqrt(torch.sum(matrix**2, dim=0) + ALMOST_ZERO))

def eta(t):
    return 0.5 * torch.tanh(6*(t - 0.5)) + 0.5

def mumford_shah_cost(M : ManifoldMesh, v, perturb, opts: Options, tv_mean, tv_sigma):
    var = 2 * tv_sigma**2
    # Called "h" in the original code
    # The functional xi(v) := kronecker_delta(eta(v) - 0.5) is smoothly approximated
    xi = torch.exp(-(v - tv_mean)**2 / var)
    # Optional weighting of vertices (currently perturb is set to ones in optimize_v())
    xi = xi * perturb

    i, j, k = M.triv[:, 0], M.triv[:, 1], M.triv[:, 2]
    v1, v2, v3 = v[i], v[j], v[k]
    v_alpha, v_beta = v2 - v1, v3 - v1
    norm_grad_v = v_alpha**2 * M.G - 2 * v_alpha * v_beta * M.F + v_beta**2 * M.E
    norm_grad_v = torch.sqrt(norm_grad_v + ALMOST_ZERO)
    if opts.mumford_shah_area_weighted:
        norm_grad_v /= M.det

    xi_sum = xi[i] + xi[j] + xi[k]
    mask = xi_sum > ALMOST_ZERO
    cost = torch.sum(xi_sum[mask] * norm_grad_v[mask]) / 6.0
    return cost