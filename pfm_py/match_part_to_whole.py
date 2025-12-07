import torch

from pfm_py.geo_refinement import compute_geodesic_descriptors
from pfm_py.icp_partial import run_icp_partial_torch_batched
from pfm_py.optimize_v import optimize_v
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.optimze_C import *
from pfm_py.options import Options

def match_and_refine(M : ManifoldMesh, N : ManifoldMesh, opts: Options):
    print(f"M: vertices: {M.n_vert}, area: {M.area:.6f}")
    print(f"N: vertices: {N.n_vert}, area: {N.area:.6f}")
    est_rank = estimate_rank(M, N)
    print(f"Estimate rank of functional map: {est_rank} / {opts.n_eigen}")

    W = create_slanted_diagonal_mask(est_rank, opts)
    M_descriptors, N_descriptors = M.compute_descriptors(opts), N.compute_descriptors(opts)
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, None, W, est_rank, opts)
    
    print("="*60)
    print("REFINEMENT STAGE")
    print("="*60)

    M_descriptors, N_descriptors = compute_geodesic_descriptors(M, N, matches, opts)
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, C, W, est_rank, opts)
    return C, v, matches

def match_part_to_whole(M : ManifoldMesh, N : ManifoldMesh, func_M, func_N, C_init, W, est_rank, opts: Options):
    v = torch.ones(M.n_vert, dtype=torch.float32, device=opts.device)
    C = C_init

    # Mass normalization: normalize descriptors by square root of mass matrix
    func_M_normalized = func_M / torch.sqrt(M.S.unsqueeze(1) + 1e-10)
    func_N_normalized = func_N / torch.sqrt(N.S.unsqueeze(1) + 1e-10)

    for i in range(opts.max_outer_iter):
        print(f"------------------------- Iteration {i + 1} -------------------------")

        # Step 1: Optimize C
        print("Optimizing C ...")
        C = optimize_C(M, N, W, func_M_normalized, func_N_normalized, C, v, est_rank, opts)

        # Step 2: Run ICP in spectral domain to refine C and get correspondences
        print("Running spectral ICP refinement ...")
        C, matches = run_icp_partial_torch_batched(M, N, C, est_rank, opts)

        # Step 3: Optimize v using the ICP-refined C
        print("Optimizing v ...")
        v = optimize_v(M, N, func_M_normalized, func_N_normalized, C, opts)
        area_diff = M.partial_area(v) - N.area
        print(f"area(N softly embedded into M) - area(N): {area_diff:.6e}")
        print(f"Number of unique M vertices onto which N is mapped: {len(torch.unique(matches))}")
        small_v_verts = (v < N.area / M.area).sum().item()
        print(f"M vertices with v < area(N) / area(M): {small_v_verts}/{M.n_vert}")

        print()

    return C, v, matches