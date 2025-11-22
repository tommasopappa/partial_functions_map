import torch

from pfm_py.geo_refinement import compute_geodesic_descriptors
from pfm_py.icp_partial import run_icp_partial_torch_batched
from pfm_py.optimize_v import optimize_v
from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.optimze_C import *
from pfm_py.options import Options

def match_and_refine(M : ManifoldMesh, N : ManifoldMesh, opts: Options):
    print("M: vertices: {M.n_vert}, area: {M.area}")
    print("N: vertices: {N.n_vert}, area: {N.area}")
    est_rank = estimate_rank(M, N)
    print("Estimate rank of functional map: {est_rank} / {opts.n_eigen}")

    W = create_slanted_diagonal_mask(est_rank, opts)
    M_descriptors, N_descriptors = M.compute_fpfh_features(opts), N.compute_fpfh_features(opts)
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, W, est_rank, opts)
    
    print("="*60)
    print("REFINEMENT STAGE")
    print("="*60)

    M_descriptors, N_descriptors = compute_geodesic_descriptors(M, N, matches, opts)
    C, v, matches = match_part_to_whole(M, N, M_descriptors, N_descriptors, W, est_rank, opts)
    return C, v, matches

def match_part_to_whole(M : ManifoldMesh, N : ManifoldMesh, func_M, func_N, W, est_rank, opts: Options):
    v = torch.ones(size=M.n_vert, dtype=torch.float32, device=opts.device)
    C = None

    for i in range(opts.max_outer_iter):
        print(f"------------------------- Iteration %{i + 1} -------------------------")

        # Step 1: Optimize C
        print("Optimizing C ...")
        C = optimize_C(M, N, W, func_M, func_N, C, v, est_rank, opts)

        # Step 2: Run ICP in spectral domain to refine C and get correspondences
        print("Running spectral ICP refinement ...")
        C, matches = run_icp_partial_torch_batched(M, N, C, est_rank, opts)

        # Step 3: Optimize v using the ICP-refined C
        print("Optimizing v ...")
        v = optimize_v(M, N, func_M, func_N, C, opts)
        area_diff = M.partial_area(v) - N.area
        print("area(N softly embedded into M) - area(N): {area_diff:.4e}")
        print("Number of unique M vertices onto which N is mapped: {len(torch.unique(matches))}")
        small_v_verts = (v < N.area / M.area).sum().item()
        print("M vertices with v < area(N) / area(M): {small_v_verts}/{M.n_vert}")

        print()

    return C, v, matches