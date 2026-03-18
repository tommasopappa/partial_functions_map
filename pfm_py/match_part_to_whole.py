import torch
import numpy as np

from pfm_py.geo_refinement import compute_geodesic_descriptors
from pfm_py.icp_partial import run_icp_partial
from pfm_py.optimize_v import optimize_v
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.optimze_C import *
from pfm_py.options import Options


def match_and_refine(M : ManifoldMesh, N : ManifoldMesh, opts: Options):
    """Run the full two-stage partial-to-whole matching pipeline.
    
    Given two meshes `M`, `N`, where `N` represents a part of `M` 
    (but it is unknown, which part), computes a functional map `C` from `N` to `M`,
    pushing forward functions on `N` to the corresponding function on `M`
    (which should be 0 on the complement of `N`).

    Stage 1 computes descriptors, estimates rank, and solves for an initial
    functional map `C`, soft membership function `v`, and vertex matches. Stage 2
    refines the solution using geodesic descriptors and re-optimizes.

    Args:
        M: Full target mesh as a `ManifoldMesh`. Has M.n_vert vertices.
        N: Partial source mesh as a `ManifoldMesh`. Has N.n_vert vertices.
        opts: Algorithm options.

    Returns:
        Tuple `(C, v, matches)` where:
            - `C`: Functional map matrix of shape (opts.n_eigen, opts.n_eigen).
                   This is the representation matrix w.r.t. the truncated eigenbasis,
                   mapping coefficients from N's basis to M's basis.
            - `v`: Soft membership function for `N` on `M`, shape (M.n_vert,).
                   Represented as values at each vertex of M, with values in [0, 1]
                   indicating which vertices of M belong to N.
            - `matches`: Vertex correspondences from `N` to `M`, shape (N.n_vert,).
                        For each vertex i in N, matches[i] is the index of the
                        corresponding vertex in M.
    """
    print(f"M: vertices: {M.n_vert}, area: {M.area:.6f}")
    print(f"N: vertices: {N.n_vert}, area: {N.area:.6f}")
    est_rank = estimate_rank(M, N)
    print(f"Estimate rank of functional map: {est_rank} / {opts.n_eigen}")
    print(f"Descriptor type: {opts.descriptor_type.upper()}")

    W = create_slanted_diagonal_mask(est_rank, opts)
    M_descriptors, N_descriptors = ManifoldMesh.compute_compatible_descriptors(M, N, opts)

    # Per-feature mass normalization
    # This means normalizing each descriptor function in the L2 norm.
    if any(tok in opts.descriptor_type.lower() for tok in ["shot", "dino", "dinov3"]):
        eps = 1e-10
        M_norm = torch.sqrt(M.S @ (M_descriptors ** 2) + eps)   # (feat_dim,)
        N_norm = torch.sqrt(N.S @ (N_descriptors ** 2) + eps)
        M_descriptors = M_descriptors / M_norm.unsqueeze(0)
        N_descriptors = N_descriptors / N_norm.unsqueeze(0)
        print(f"[NORM:{opts.descriptor_type.upper()}] Applied per-feature mass normalization (feat_dim={M_norm.numel()})")

    # Run alternating optimization with descriptor functions
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, None, W, est_rank, opts.max_outer_iter, opts)
    if opts.refine_iters == 0:
        return C, v, matches
    
    print("="*60)
    print("REFINEMENT STAGE")
    print("="*60)

    # Refinement step: compute geodesic descriptors from using previously computed matches
    # and re-run alternating optimization to obtain refined functional map
    M_descriptors, N_descriptors = compute_geodesic_descriptors(M, N, matches, opts)
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, C, W, est_rank, opts.refine_iters, opts)
    return C, v, matches

def match_part_to_whole(M : ManifoldMesh, N : ManifoldMesh, func_M, func_N, C_init, W, est_rank, outer_iters, opts: Options):
    """Compute partial-to-whole functional map with soft membership estimation.
    
    Given pairs of corresponding functions (descriptors) func_M, func_N on 
    the full mesh M and the partial mesh N, finds a functional map C
    mapping func_N to restrictions of func_M to N. Since it is unknown which
    part of M corresponds to N, a soft membership function v for N on M is also estimated.
    The restriction of a function f on M is then approximated as f * v (pointwise multiplication).
    
    This routine alternates between optimizing the functional map `C`,
    refining it via spectral ICP, and updating the soft membership function `v`.

    Args:
        M: Full target mesh as a `ManifoldMesh`. Has M.n_vert vertices.
        N: Partial source mesh as a `ManifoldMesh`. Has N.n_vert vertices.
        func_M: Descriptor matrix on `M`, shape (M.n_vert, feat_dim).
                Functions represented as values at each vertex, one row per vertex.
        func_N: Descriptor matrix on `N`, shape (N.n_vert, feat_dim).
                Functions represented as values at each vertex, one row per vertex.
                Should ideally be the restriction of func_M to N.
        C_init: Functional map prior of shape (opts.n_eigen, opts.n_eigen), or None
                if no prior is available.
        W: Mask for regularization, shape (opts.n_eigen, opts.n_eigen).
           Diagonal or slanted diagonal mask used to weight different frequency pairs.
        est_rank: Estimated functional map rank (scalar int or torch.Tensor).
                  Indicates the effective dimensionality of the map.
        outer_iters: Number of outer iterations for alternating optimization.
        opts: Algorithm options.

    Returns:
        Tuple `(C, v, matches)` for the current stage:
            - `C`: Functional map matrix of shape (opts.n_eigen, opts.n_eigen).
                   Representation matrix w.r.t. truncated eigenbasis.
            - `v`: Soft membership function for `N` on `M`, shape (M.n_vert,).
                   Represented as values at each vertex of M, with values in [0, 1]
                   indicating which vertices of M correspond to N.
            - `matches`: Pointwise correspondences from `N` to `M`, shape (N.n_vert,).
                        For each vertex i in N, matches[i] is the index of the
                        corresponding vertex in M.
    """
    # Initialize v to all ones (so every vertex is considered to be in N).
    v = torch.ones(M.n_vert, dtype=torch.float32, device=opts.device)
    C = C_init

    # Alternating optimization loop
    for i in range(outer_iters):
        print(f"------------------------- Iteration {i + 1} -------------------------")

        # Step 1: Optimize C
        print("Optimizing C ...")
        C = optimize_C(M, N, W, func_M, func_N, C, v, est_rank, opts)

        # Step 2: Run ICP in spectral domain to refine C and get correspondences
        print("Running spectral ICP refinement ...")
        C, matches = run_icp_partial(M, N, C, est_rank, opts)

        # Step 3: Optimize v using the ICP-refined C
        print("Optimizing v ...")
        v = optimize_v(M, N, func_M, func_N, C, opts)
        area_diff = M.partial_area(v) - N.area
        print(f"area(N softly embedded into M) - area(N): {area_diff:.6e}")
        print(f"Number of unique M vertices onto which N is mapped: {len(torch.unique(matches))}")
        print()

    return C, v, matches