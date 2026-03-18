import argparse

import numpy as np
import open3d as o3d
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options

def test_spectral_coordinates(M: ManifoldMesh):
    """Analyze spectral coordinate distinctiveness via nearest-neighbour matching.

    For each vertex v, this computes the L2-normalized landmark function
    hat_delta_v = delta_v / ||delta_v||_L2 (where ||delta_v||_L2 = sqrt(S[v])),
    projects it into the truncated spectral basis, reconstructs it back in the
    vertex domain, and L2-normalizes the result.  A nearest-neighbour search
    then matches each reconstructed function to the most similar original
    landmark function (in L2 function-space distance).

    Because all hat_delta_w already have unit L2 norm, and all reconstructed
    functions are also unit-normalized, finding the nearest original landmark
    reduces to maximizing the L2 inner product:
        <r_hat_v, hat_delta_w>_L2 = r_hat_v[w] * sqrt(S[w])
    which is computed for all pairs at once via a single matrix operation.

    Returns:
        dict with the following entries:
            - `spectral_coordinates`: shape (n_eigen, n_vert).
            - `reconstructed_functions`: shape (n_vert, n_vert), columns are reconstructed
              (but not yet unit-normalized) functions.
            - `recon_normalized`: shape (n_vert, n_vert), unit L2-normalized reconstructions.
            - `nn_matches`: shape (n_vert,), index of the nearest original landmark for each
              reconstructed function.
            - `correct_mask`: shape (n_vert,) bool, True where nn_matches[v] == v.
            - `n_correct` (int): number of vertices whose reconstruction matches themselves.
            - `match_counts`: shape (n_vert,), how many reconstructed functions point to each
              original vertex.
            - `n_matched` (int): number of original vertices that receive at least one match.
            - `n_unmatched` (int): number of original vertices that receive no match.
            - `mean_matches_among_matched` (float): mean match count over matched vertices.
    """
    evecs = M.evecs
    mass_weights = M.S
    n_vert = M.n_vert

    # L2 norm of each indicator: ||delta_v||_L2 = sqrt(S[v])
    landmark_norms = torch.sqrt(mass_weights)  # (n_vert,)

    # Spectral coords of hat_delta_v: c_k(v) = sqrt(S[v]) * phi_k[v]
    spectral_coordinates = (landmark_norms[:, None] * evecs).T  # (n_eigen, n_vert)
    reconstructed_functions = evecs @ spectral_coordinates       # (n_vert, n_vert)

    # L2-normalize each reconstructed function (column)
    recon_l2_norms = torch.sqrt((reconstructed_functions ** 2 * mass_weights[:, None]).sum(dim=0))  # (n_vert,)
    recon_normalized = reconstructed_functions / recon_l2_norms[None, :]  # (n_vert, n_vert)

    # Similarity matrix: sim[v, w] = <recon_hat_v, hat_delta_w>_L2
    #   = recon_hat_v[w] * sqrt(S[w])   (hat_delta_w is zero everywhere except at w)
    # Vectorized: scale rows of recon_normalized by sqrt(S), then transpose.
    similarity = (recon_normalized * landmark_norms[:, None]).T  # (n_vert, n_vert)

    # For each reconstructed function (row = query vertex v),
    # find the original landmark with highest similarity.
    nn_matches = similarity.argmax(dim=1)  # (n_vert,)

    correct_mask = nn_matches == torch.arange(n_vert, device=evecs.device)
    n_correct = int(correct_mask.sum().item())

    match_counts = torch.bincount(nn_matches, minlength=n_vert)  # (n_vert,)
    matched_mask = match_counts > 0
    n_matched = int(matched_mask.sum().item())
    n_unmatched = n_vert - n_matched
    mean_matches_among_matched = float(match_counts[matched_mask].float().mean().item())

    return {
        "spectral_coordinates": spectral_coordinates,
        "reconstructed_functions": reconstructed_functions,
        "recon_normalized": recon_normalized,
        "nn_matches": nn_matches,
        "correct_mask": correct_mask,
        "n_correct": n_correct,
        "match_counts": match_counts,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "mean_matches_among_matched": mean_matches_among_matched,
    }


def _stats(t: torch.Tensor) -> str:
    """Return a compact min / mean ± std / max summary for a 1-D tensor."""
    return (f"min={t.min():.4f}  mean={t.mean():.4f}  "
            f"std={t.std():.4f}  max={t.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze spectral coordinate distinctiveness for a mesh."
    )
    parser.add_argument("--mesh", required=True, help="Path to a .off mesh file.")
    parser.add_argument(
        "--n-eigen", type=int, default=100,
        help="Number of eigenfunctions to compute (default: 100)."
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda:0 if available, else cpu)."
    )
    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh}")
    o3d_mesh = o3d.io.read_triangle_mesh(args.mesh)
    verts = np.asarray(o3d_mesh.vertices)
    trivs = np.asarray(o3d_mesh.triangles)
    print(f"  Vertices : {verts.shape[0]}")
    print(f"  Faces    : {trivs.shape[0]}")

    opts = Options(device=args.device)
    opts.n_eigen = args.n_eigen

    print(f"\nBuilding ManifoldMesh (n_eigen={opts.n_eigen}, device={opts.device}) ...")
    M = ManifoldMesh(verts, trivs, opts)
    print(f"  Surface area : {M.area:.4f}")

    print("\nRunning test_spectral_coordinates ...")
    results = test_spectral_coordinates(M)

    n_vert  = M.n_vert
    n_eigen = M.evecs.shape[1]
    print(f"\n{'='*60}")
    print(f"Spectral coordinates NN-matching analysis")
    print(f"  n_vert  = {n_vert}")
    print(f"  n_eigen = {n_eigen}")
    print(f"{'='*60}")

    n_correct   = results["n_correct"]
    n_matched   = results["n_matched"]
    n_unmatched = results["n_unmatched"]
    mean_mult   = results["mean_matches_among_matched"]
    match_counts = results["match_counts"]

    print(f"\nCorrect self-matches (recon of v matched back to v):")
    print(f"  {n_correct} / {n_vert}  ({100 * n_correct / n_vert:.1f}%)")

    print(f"\nOriginal vertices receiving NO match:")
    print(f"  {n_unmatched} / {n_vert}  ({100 * n_unmatched / n_vert:.1f}%)")

    print(f"\nOriginal vertices receiving at least one match: {n_matched}")
    print(f"  Mean matches among them : {mean_mult:.2f}")

    # Distribution of match counts (count=0 are unmatched vertices)
    max_count = int(match_counts.max().item())
    print(f"\nMatch-count distribution (over all {n_vert} vertices):")
    print(f"  {'matches':>8}  {'# vertices':>10}  {'% of all verts':>15}")
    for c in range(0, max_count + 1):
        n = int((match_counts == c).sum().item())
        if n > 0:
            print(f"  {c:>8d}  {n:>10d}  {100 * n / n_vert:>14.1f}%")

    # Top-5 most-matched original vertices (hot-spots)
    top_k = min(5, n_matched)
    top_idx = torch.topk(match_counts, top_k).indices
    print(f"\nTop-{top_k} most-matched original vertices:")
    print(f"  {'vertex':>8}  {'matches':>8}")
    for idx in top_idx:
        print(f"  {idx.item():>8d}  {match_counts[idx].item():>8d}")
