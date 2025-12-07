from dataclasses import dataclass

@dataclass
class Options:
    device : str
    descriptor_type: str = "fpfh"  # "fpfh" or "shot"
    mumford_shah_area_weighted: bool = False
    v_lr : float = 1e-2
    v_max_iter : int = 2000 #1000
    mu1 : float = 1.0
    mu2 : float = 1e2
    mu3 : float = 1.0
    mu4 : float = 1e3
    mu5 : float = 1e3
    tv_sigma = 0.2 * 4e-4
    C_lr : float = 1e-2
    C_max_iter : int = 2000
    max_outer_iter : int = 7
    n_eigen : int = 100
    mask_sigma : float = 0.03
    max_icp_iters : int = 30
    icp_batch_size : int = 1000
    refine_fps : int = 50
    refine_iters = 3
    fps_variance = 0.7 * 4e-4