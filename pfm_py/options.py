from dataclasses import dataclass

@dataclass
class Options:
    """Collection of options and hyperparameters."""
    device : str # device to use, e.g., "cpu" or "cuda:0"
    descriptor_type: str = "shot"  # Possible values: "fpfh", "shot", "dino", "dinov3"
    mumford_shah_area_weighted: bool = False # enable area-weighted Mumford-Shah regularization, see optimize_v.py
    v_lr : float = 1e-2 # learning rate for v optimization
    v_max_iter : int = 2000 # maximum iterations for v optimization
    mu1 : float = 1.0 # weight for area term in v optimization
    mu2 : float = 1e2 # weight for Mumford-Shah regularization in v optimization
    mu3 : float = 1.0 # weight for slanted-diagonal prior term in C optimization
    mu4 : float = 1e3 # weight for orthogonality off-diagonal term in C optimization
    mu5 : float = 1e3 # weight for orthogonality diagonal term in C optimization
    tv_sigma : float = 0.2 * 4e-4
    C_lr : float = 1e-2 # learning rate for C optimization
    C_max_iter : int = 2000 # maximum iterations for C optimization
    max_outer_iter : int = 7 # maximum outer iterations (alternating C and v)
    n_eigen : int = 100 # number of eigenfunctions to use
    mask_sigma : float = 0.03 # sigma for slanted diagonal mask W, see optimze_C.py
    max_icp_iters : int = 30 # maximum iterations for ICP refinement
    icp_batch_size : int = 1000 # batch size used in ICP refinement
    fps_n_sample_points : int = 50 # number of farthest point samples for geodesic refinement
    refine_iters : int = 7 # number of refinement iterations
    geo_desc_variance : float = 0.7 * 4e-4 # used for geodesic distance descriptors, see geo_refinement.py
    C_patience_iters : int = 100 # patience iterations for early stopping in C optimization
    v_patience_iters : int = 2000 # patience iterations for early stopping in v optimization
    early_stopping : bool = False # enable early stopping?