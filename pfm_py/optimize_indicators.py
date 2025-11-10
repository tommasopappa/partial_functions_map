import numpy as np
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from numpy import ndarray
from dataclasses import dataclass

from pfm_py.mumford_shah import mumford_shah_cost

@dataclass
class VStepOptions:
    tv_mean: float = 0.0
    tv_sigma: float = 1.0
    mumford_shah_area_weighted: bool = False
    lr : float = 1e-2
    max_iter : int = 200
    mu1 : float = 0.1
    mu2 : float = 0.1

def optimize_v(M : ManifoldMesh, N : ManifoldMesh, G : ndarray, F : ndarray, C : ndarray, opts: VStepOptions) -> ndarray:
    A = C @ np.transpose(N.evecs) @ (N.S[:, None] * F)
    B = np.transpose(M.evecs) * M.S
    v0 = M.evecs @ (C @ (np.transpose(N.evecs) @ (N.S[:, None] * np.ones((N.n_vert, 1)))))

    areas = M.S
    target_area = np.sum(N.S)

    # Maybe store the data permanently as PyTorch tensors on the GPU?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.tensor(A, dtype=torch.float32, device=device, requires_grad=False)
    B = torch.tensor(B, dtype=torch.float32, device=device, requires_grad=False)
    G_t = torch.tensor(G, dtype=torch.float32, device=device, requires_grad=False)
    v = torch.tensor(v0, dtype=torch.float32, device=device, requires_grad=True)
    M_vert = torch.tensor(M.vert, dtype=torch.float32, device=device, requires_grad=False)
    M_triv = torch.tensor(M.triv, dtype=torch.long, device=device, requires_grad=False)
    perturb = torch.ones_like(v)

    optimizer = torch.optim.Adam([v], lr=opts.lr)
    for i in range(opts.max_iter):
        optimizer.zero_grad()
        loss = v_loss(A, B, G_t, v, perturb, target_area, areas, M_vert, M_triv, opts)
        loss.backward()
        optimizer.step()

    return v.detach().cpu().numpy()

def v_loss(CA, PtS, G, v, perturb, target_area, areas, M_vert, M_triv, opts : VStepOptions):
    tv = eta(v) # maps v into [0,1] 
    VG = tv[:, None] * G
    diff = CA - PtS @ VG
    data_term = torch.norm(diff, dim=0).sum()

    current_area = torch.sum(tv * areas)
    area_term = (target_area - current_area)**2
    reg_term = mumford_shah_cost(v, M_vert, M_triv, perturb, opts)

    return data_term + opts.mu1 * area_term + opts.mu2 * reg_term

def eta(t):
    return 0.5 * torch.tanh(6*t - 3) + 0.5