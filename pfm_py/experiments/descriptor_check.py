import argparse
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import open3d as o3d
import torch

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.options import Options


def _set_axes_clean(ax):
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.set_axis_off()


def _compute_face_normals(verts_vis: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = []
    for f in triangles:
        a, b, c = verts_vis[f]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n = n / norm
        else:
            n = np.array([0.0, 0.0, 1.0])
        normals.append(n)
    return np.array(normals)


def _apply_blinn_phong(facecols: np.ndarray, normals: np.ndarray) -> np.ndarray:
    light_dir = np.array([0.577, 0.577, 0.577])
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0])
    view_dir = view_dir / np.linalg.norm(view_dir)
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec)

    shininess = 128.0
    ambient = 0.25
    kd = 0.85
    ks = 0.80

    diffuse = np.maximum(normals @ light_dir, 0.0)
    spec = np.maximum(normals @ half_vec, 0.0) ** shininess
    shading = ambient + kd * diffuse[:, None]
    return np.clip(facecols * shading + ks * spec[:, None], 0.0, 1.0)


def _l1_norm(f: np.ndarray, mass: np.ndarray) -> float:
    # L1 norm on manifold: ||f||_L1 = sum_i |f_i| S_i
    return float(np.sum(np.abs(f) * mass))


def _load_ground_truth(path: str) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float).astype(int) - 1
    return arr.reshape(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Per-dimension descriptor visualization on full/partial meshes with optional GT pullback."
    )
    parser.add_argument("--full-mesh", required=True, help="Path to full mesh (.off)")
    parser.add_argument("--partial-mesh", required=True, help="Path to partial mesh (.off)")
    parser.add_argument("--desc", required=True, help="Descriptor type (e.g., shot, fpfh, dino, dinov3, shot+dino)")
    parser.add_argument("--target-path", required=True, help="Output folder")
    parser.add_argument("--gt-path", dest="gt_path", default=None, help="Optional path to ground truth correspondences (.vts)")
    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    opts = Options(device)
    opts.descriptor_type = args.desc

    print(f"Device: {device}")
    print(f"Loading full mesh: {args.full_mesh}")
    mesh_full = o3d.io.read_triangle_mesh(args.full_mesh)
    print(f"Loading partial mesh: {args.partial_mesh}")
    mesh_partial = o3d.io.read_triangle_mesh(args.partial_mesh)

    vert_M = np.asarray(mesh_full.vertices)
    triv_M = np.asarray(mesh_full.triangles)
    vert_N = np.asarray(mesh_partial.vertices)
    triv_N = np.asarray(mesh_partial.triangles)

    print("Building manifold meshes...")
    M = ManifoldMesh(vert_M, triv_M, opts, compute_geo=False)
    N = ManifoldMesh(vert_N, triv_N, opts, compute_geo=False)

    print(f"Computing compatible descriptors: {opts.descriptor_type}")
    desc_M_t, desc_N_t = ManifoldMesh.compute_compatible_descriptors(M, N, opts)
    desc_M = desc_M_t.detach().cpu().numpy()
    desc_N = desc_N_t.detach().cpu().numpy()

    if desc_M.ndim != 2 or desc_N.ndim != 2:
        raise RuntimeError("Descriptors must be 2D arrays (n_vertices, feat_dim).")
    if desc_M.shape[1] != desc_N.shape[1]:
        raise RuntimeError("Feature dimensions differ between full and partial descriptors.")
    mass_N = N.S.detach().cpu().numpy().reshape(-1)

    feat_dim = int(desc_M.shape[1])
    print(f"Descriptor dimensions: {feat_dim}")

    # Optional ground-truth mapping N -> M (one full-mesh vertex index per partial vertex)
    gt_matches = None
    has_gt = False
    if args.gt_path:
        print(f"Loading ground truth: {args.gt_path}")
        gt_matches = _load_ground_truth(args.gt_path)
        if gt_matches.shape[0] != vert_N.shape[0]:
            raise ValueError(
                f"Ground truth length ({gt_matches.shape[0]}) does not match partial vertex count ({vert_N.shape[0]})."
            )
        if np.any(gt_matches < 0) or np.any(gt_matches >= vert_M.shape[0]):
            raise ValueError("Ground truth correspondences contain out-of-range indices.")
        has_gt = True

    # Report averages BEFORE any PNG creation starts.
    # Always report average L1 norm of native functions on N.
    l1_native_vals = np.sum(np.abs(desc_N) * mass_N[:, None], axis=0)
    avg_l1_native = float(np.mean(l1_native_vals))
    print(f"Average L1 norm of features on N over all dimensions: {avg_l1_native:.8f}")

    # If GT is available, also report average deviation and relative average error (%).
    if has_gt and gt_matches is not None:
        desc_N_gt_pullback = desc_M[gt_matches, :]  # (n_partial_vertices, feat_dim)
        l1_dev_vals = np.sum(np.abs(desc_N_gt_pullback - desc_N) * mass_N[:, None], axis=0)
        avg_l1_dev = float(np.mean(l1_dev_vals))
        avg_rel_percent = 100.0 * avg_l1_dev / max(avg_l1_native, 1e-14)
        print(
            "Average L1 norm of deviation over all feature dimensions "
            f"(GT pullback vs native partial feature): {avg_l1_dev:.8f}"
        )
        print(
            "Relative average error (avg deviation L1 / avg native L1) [%]: "
            f"{avg_rel_percent:.4f}%"
        )

    # Shared centering and scaling, same style as color pullback visualization
    all_verts = np.vstack([vert_M, vert_N])
    shared_center = all_verts.mean(0)
    ranges = all_verts.max(0) - all_verts.min(0)
    max_range = float(ranges.max())
    scale = (1.0 / max_range) if max_range > 1e-12 else 1.0
    v_M_vis = (vert_M - shared_center) * scale
    v_N_vis = (vert_N - shared_center) * scale

    poly_M = [v_M_vis[f] for f in triv_M]
    poly_N = [v_N_vis[f] for f in triv_N]
    normals_M = _compute_face_normals(v_M_vis, triv_M)
    normals_N = _compute_face_normals(v_N_vis, triv_N)

    cmap = plt.get_cmap("seismic")
    light_source = mpl.colors.LightSource(azdeg=315, altdeg=45)

    created_count = 0
    try:
        for i in range(feat_dim):
            f_M = desc_M[:, i]
            f_N = desc_N[:, i]

            f_N_gt_pullback = None
            l1_native = _l1_norm(f_N, mass_N)
            l1_dev = None
            rel_err_percent = None
            if has_gt and gt_matches is not None:
                f_N_gt_pullback = f_M[gt_matches]
                l1_dev = _l1_norm(f_N_gt_pullback - f_N, mass_N)
                rel_err_percent = 100.0 * l1_dev / max(l1_native, 1e-14)

            # Shared symmetric color range so blue = negative, red = positive, comparable across panels
            abs_max = float(max(np.max(np.abs(f_M)), np.max(np.abs(f_N))))
            if f_N_gt_pullback is not None:
                abs_max = float(max(abs_max, np.max(np.abs(f_N_gt_pullback))))
            abs_max = max(abs_max, 1e-12)
            norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

            cols_M = cmap(norm(f_M))[:, :3]
            cols_N = cmap(norm(f_N))[:, :3]
            facecols_M = cols_M[triv_M].mean(axis=1)
            facecols_N = cols_N[triv_N].mean(axis=1)
            facecols_M_shaded = _apply_blinn_phong(facecols_M, normals_M)
            facecols_N_shaded = _apply_blinn_phong(facecols_N, normals_N)

            n_plots = 3 if f_N_gt_pullback is not None else 2
            fig = plt.figure(figsize=(6 * n_plots, 5))

            ax1 = fig.add_subplot(1, n_plots, 1, projection="3d")
            pc1 = Poly3DCollection(
                poly_M,
                facecolors=facecols_M_shaded,
                linewidths=0,
                edgecolor=None,
                alpha=1.0,
                shade=True,
                lightsource=light_source,
            )
            ax1.add_collection3d(pc1)
            ax1.set_title(f"Full mesh M\nfeature[{i}]")
            _set_axes_clean(ax1)
            ax1.view_init(elev=20, azim=45)
            ax1.grid(False)

            if f_N_gt_pullback is not None:
                cols_N_gt = cmap(norm(f_N_gt_pullback))[:, :3]
                facecols_N_gt = cols_N_gt[triv_N].mean(axis=1)
                facecols_N_gt_shaded = _apply_blinn_phong(facecols_N_gt, normals_N)

                ax2 = fig.add_subplot(1, n_plots, 2, projection="3d")
                pc2 = Poly3DCollection(
                    poly_N,
                    facecolors=facecols_N_gt_shaded,
                    linewidths=0,
                    edgecolor=None,
                    alpha=1.0,
                    shade=True,
                    lightsource=light_source,
                )
                ax2.add_collection3d(pc2)
                ax2.set_title(f"Partial mesh N\nGT pullback feature[{i}]")
                _set_axes_clean(ax2)
                ax2.view_init(elev=20, azim=45)
                ax2.grid(False)
                method_col = 3
            else:
                method_col = 2

            ax3 = fig.add_subplot(1, n_plots, method_col, projection="3d")
            pc3 = Poly3DCollection(
                poly_N,
                facecolors=facecols_N_shaded,
                linewidths=0,
                edgecolor=None,
                alpha=1.0,
                shade=True,
                lightsource=light_source,
            )
            ax3.add_collection3d(pc3)
            ax3.set_title(f"Partial mesh N\ncomputed feature[{i}]")
            _set_axes_clean(ax3)
            ax3.view_init(elev=20, azim=45)
            ax3.grid(False)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # Dedicated colorbar axis prevents overlap with 3D subplots.
            cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
            fig.colorbar(sm, cax=cax)

            if l1_dev is not None and rel_err_percent is not None:
                fig.text(
                    0.5,
                    0.035,
                    (
                        f"L1 native on N: {l1_native:.6f}   |   "
                        f"L1 deviation (GT pullback - native): {l1_dev:.6f}   |   "
                        f"Relative error: {rel_err_percent:.3f}%"
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
                print(
                    f"[dim {i:04d}] L1_native={l1_native:.8f}, "
                    f"L1_deviation={l1_dev:.8f}, rel_error={rel_err_percent:.4f}%"
                )
            else:
                fig.text(
                    0.5,
                    0.035,
                    f"L1 native on N: {l1_native:.6f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
                print(f"[dim {i:04d}] L1_native={l1_native:.8f} (no GT, deviation/relative error unavailable)")

            # Avoid tight_layout warning with 3D axes; leave room for colorbar and bottom text.
            fig.subplots_adjust(left=0.02, right=0.90, bottom=0.14, top=0.94, wspace=0.03)
            out_name = f"descriptor_dim_{i:04d}.png"
            out_path = os.path.join(args.target_path, out_name)
            plt.savefig(out_path, dpi=300)
            plt.close(fig)

            created_count += 1
            print(f"[{i+1:4d}/{feat_dim:4d}] saved {out_path}")
    except KeyboardInterrupt:
        plt.close("all")
        print("\nInterrupted by user (Ctrl+C).")
        print(f"Kept {created_count} already-created file(s) in: {args.target_path}")
        print("Exiting gracefully.")
        return

    print("Done.")


if __name__ == "__main__":
    main()
