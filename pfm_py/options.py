from dataclasses import dataclass
import math

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
    C_lr : float = 1e-2 # learning rate for C optimization
    C_max_iter : int = 2000 # maximum iterations for C optimization
    max_outer_iter : int = 7 # maximum outer iterations (alternating C and v)
    n_eigen : int = 100 # number of eigenfunctions to use
    mask_sigma : float = 0.03 # sigma for slanted diagonal mask W, see optimze_C.py
    max_icp_iters : int = 30 # maximum iterations for ICP refinement
    icp_batch_size : int = 1000 # batch size used in ICP refinement
    fps_n_sample_points : int = 50 # number of farthest point samples for geodesic refinement
    refine_iters : int = 7 # number of refinement iterations
    
    early_stopping : bool = False # enable early stopping in C step and v step
    patience_iters : int = 100 # number of iterations to wait for improvement before early stopping
    early_stopping_tol : float = 1e-4 # minimum relative improvement to reset patience counter for early stopping

    tv_mean : float = 0.5 # target membership density for Mumford-Shah regularization, see optimize_v.py
    # standard deviation for Mumford-Shah Gaussian localization, see optimize_v.py
    # This value is scale-dependent. The value of 0.2 in the PFM code was optimized, for meshes with surface
    # area 1.5 - 2.0 * 10^4. The value below downscales this to meshes of area 1.
    # In the code, this is further scaled by sqrt(M.area) to adapt to the actual mesh area.
    # This might not be the optimal way to scale this parameter, but seems to be good enough.
    tv_sigma : float = 0.2 / math.sqrt(1.75 * 10**4) 
    # variance for Gaussian weighting of geodesic distance in geodesic descriptor computation, see geo_refinement.py
    # scale_dependent just like tv_sigma
    geo_descriptor_variance : float = 0.7 / math.sqrt(1.75 * 10**4) 