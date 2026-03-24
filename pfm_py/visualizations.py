import os
import numpy as np


def create_functional_map_visualization(vert_M, vert_N, triv_M, triv_N, M, N, C, v, matches, gt_matches, dist_method_geo, opts, output_folder):
    """Create and save the functional map visualization showing:
    - Source function on N
    - Ground truth color transfer to M (if available)
    - Method push-forward RGB to M
    - Method error heatmap (if available)
    - Soft membership function v (on N or M)
    - Ground truth membership (binary on M) if ground truth is available
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib as mpl

    def create_full_colormap(n):
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 1, n))[:, :3]
        return colors

    def find_boundary_edges(triangles):
        """Find boundary edges (edges that appear only once in the mesh)."""
        from collections import defaultdict
        edge_count = defaultdict(int)
        for tri in triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                edge_count[edge] += 1
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges

    # Center and downscale meshes using a shared centroid and common scale
    all_verts = np.vstack([vert_M, vert_N])
    shared_center = all_verts.mean(0)
    ranges = all_verts.max(0) - all_verts.min(0)
    max_range = float(ranges.max())
    scale = (1.0 / max_range) if max_range > 1e-12 else 1.0
    v_N_vis = (vert_N - shared_center) * scale
    v_M_vis = (vert_M - shared_center) * scale

    # Find boundary edges
    boundary_edges_N = find_boundary_edges(triv_N)
    boundary_edges_M = find_boundary_edges(triv_M)

    def set_axes_clean(ax):
        # Fit perfectly into a fixed cube without coordinate systems
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_axis_off()

    # --- Paper colormap on N and push-forward RGB via the functional map ---
    def create_paper_colormap(verts_A: np.ndarray, verts_B: np.ndarray) -> np.ndarray:
        """RGB = normalized XYZ of A using min/max computed on B (create_colormap(B,B))."""
        mins = verts_B.min(axis=0)
        maxs = verts_B.max(axis=0)
        denom = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
        return (verts_A - mins) / denom

    # Source colors on N (create_colormap(N,N))
    colors_N = np.clip(create_paper_colormap(vert_N, vert_N), 0.0, 1.0)

    # Ground truth RGB transfer to M (vertex-wise assignment)
    has_gt = gt_matches is not None and len(gt_matches) == vert_N.shape[0]
    facecols_gt = None
    if has_gt:
        colors_M_gt = np.zeros((vert_M.shape[0], 3), dtype=float)
        colors_M_gt[gt_matches] = colors_N
        facecols_gt = colors_M_gt[triv_M].mean(axis=1)

    # Push-forward each RGB component via C: a_M = C (E_N^T S_N f_N), then f_M = E_M a_M
    evecs_N_T = N.evecs.T.numpy(force=True)
    mass_N = N.S.numpy(force=True)
    evecs_M = M.evecs.numpy(force=True)
    colors_M_method = np.zeros((vert_M.shape[0], 3), dtype=float)
    for ch in range(3):
        fN = colors_N[:, ch]
        aN = evecs_N_T @ (mass_N * fN)
        aM = C @ aN
        fM = evecs_M @ aM
        # normalize per-channel for visualization stability
        fM = (fM - fM.min()) / (fM.max() - fM.min() + 1e-10)
        colors_M_method[:, ch] = fM

    # Prepare mesh polygons and face colors
    poly_N = [v_N_vis[f] for f in triv_N]
    poly_M = [v_M_vis[f] for f in triv_M]

    # Map vertex colors to face colors (average of vertex colors)
    cmap_viridis = plt.get_cmap("viridis")
    cmap_coolwarm = plt.get_cmap("coolwarm")
    cmap_gray = plt.get_cmap("gray")

    facecols_source = colors_N[triv_N].mean(axis=1)
    facecols_method = colors_M_method[triv_M].mean(axis=1)

    facecols_method_error = None
    error_cb_cfg = None
    if dist_method_geo is not None and dist_method_geo.size > 0:
        vmax_error = np.percentile(dist_method_geo, 95)
        dist_method_geo_norm = np.clip(dist_method_geo / vmax_error, 0, 1)
        dist_method_geo_colors = cmap_coolwarm(dist_method_geo_norm)[:, :3]
        facecols_method_error = dist_method_geo_colors[triv_N].mean(axis=1)
        error_cb_cfg = (cmap_coolwarm, plt.Normalize(vmin=0, vmax=vmax_error))

    # Soft membership function v visualization (on full mesh M; grayscale 0→1)
    v_cb_cfg = None
    facecols_v = None
    v_arr = None
    if v is not None:
        try:
            v_arr = np.asarray(v, dtype=float).reshape(-1)
            if v_arr.size == vert_M.shape[0]:
                vnorm = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min() + 1e-12)
                vcols = cmap_gray(vnorm)[:, :3]
                facecols_v = vcols[triv_M].mean(axis=1)
                v_cb_cfg = (cmap_gray, plt.Normalize(vmin=0, vmax=1))
            else:
                try:
                    print(f"Warning: membership v has size {v_arr.size}, expected {vert_M.shape[0]} (full mesh M); skipping membership visualization.")
                except Exception:
                    pass
                facecols_v = None
                v_cb_cfg = None
        except Exception:
            facecols_v = None
            v_cb_cfg = None

    # --- Build dynamic panel list ---
    panels = []
    panels.append(("N: Paper Colormap (RGB)", poly_N, facecols_source, boundary_edges_N, v_N_vis, None))
    if has_gt and facecols_gt is not None:
        panels.append(("GROUND TRUTH Transfer", poly_M, facecols_gt, boundary_edges_M, v_M_vis, None))
    panels.append(("METHOD Push-forward (RGB)", poly_M, facecols_method, boundary_edges_M, v_M_vis, None))
    if facecols_method_error is not None and error_cb_cfg is not None:
        panels.append((f"Method Error (mean = {dist_method_geo.mean():.4f})", poly_N, facecols_method_error, boundary_edges_N, v_N_vis, error_cb_cfg))
    if facecols_v is not None and v_cb_cfg is not None:
        # Membership is defined on M; visualize and add GT adjacent when available
        panels.append(("Soft Membership v (M)", poly_M, facecols_v, boundary_edges_M, v_M_vis, v_cb_cfg))
        if has_gt:
            try:
                v_gt_M = np.zeros(vert_M.shape[0], dtype=float)
                v_gt_M[gt_matches] = 1.0
                v_gt_cols_M = cmap_gray(v_gt_M)[:, :3]
                facecols_v_gt_M = v_gt_cols_M[triv_M].mean(axis=1)
                v_gt_cb_cfg = (cmap_gray, plt.Normalize(vmin=0, vmax=1))
                panels.append(("Ground Truth Membership (M)", poly_M, facecols_v_gt_M, boundary_edges_M, v_M_vis, v_gt_cb_cfg))
            except Exception:
                pass

    # Layout: up to 3 columns per row
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    boundary_line_width = 0
    opacity = 1.0
    # Render 3D mesh panels
    for idx, (title, polys, facecols, boundary_edges, verts_vis, cb_cfg) in enumerate(panels, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
        pc = Poly3DCollection(polys, facecolors=facecols, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
        ax.add_collection3d(pc)
        for edge in boundary_edges:
            pts = verts_vis[list(edge)]
            ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', linewidth=boundary_line_width)
        ax.set_title(title)
        set_axes_clean(ax)
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        # Add colorbar when configuration is provided (error or membership)
        if cb_cfg is not None:
            cmap_cb, norm_cb = cb_cfg
            sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.6)

    plt.tight_layout()
    functional_map_fname = f"functional_map_visualization_{opts.descriptor_type}.png"
    functional_map_path = os.path.join(output_folder, functional_map_fname)
    plt.savefig(functional_map_path, dpi=300)
    print(f"Saved visualization to {functional_map_path}")

    return functional_map_path


def create_functional_map_heatmap(C, opts, output_folder):
    """Create and save a standalone heatmap PNG for the functional map matrix C."""
    import matplotlib.pyplot as plt

    C_arr = np.asarray(C, dtype=float)
    vmax = np.max(np.abs(C_arr)) if C_arr.size > 0 else 1.0
    if vmax < 1e-12:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    hm = ax.imshow(C_arr, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title('Functional Map C (Heatmap)')
    ax.set_xlabel('Basis index on N')
    ax.set_ylabel('Basis index on M')
    plt.colorbar(hm, ax=ax, shrink=0.85)

    plt.tight_layout()
    heatmap_fname = f"functional_map_heatmap_{opts.descriptor_type}.png"
    heatmap_path = os.path.join(output_folder, heatmap_fname)
    plt.savefig(heatmap_path, dpi=300)
    print(f"Saved heatmap to {heatmap_path}")
    return heatmap_path


def create_color_pullback_visualization(vert_M, vert_N, triv_M, triv_N, matches, gt_matches, opts, output_folder):
    """Create and save the color pullback visualization showing 3 plots:
    - Full mesh with paper colormap (like functional map pushforward)
    - Partial mesh with method pullback (via pointwise matches)
    - Partial mesh with ground truth pullback (if available)

    Uses same centering, lighting, and rendering as functional map pushforward.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib as mpl

    def create_paper_colormap(verts_A: np.ndarray, verts_B: np.ndarray) -> np.ndarray:
        """RGB = normalized XYZ of A using min/max computed on B (create_colormap(B,B))."""
        mins = verts_B.min(axis=0)
        maxs = verts_B.max(axis=0)
        denom = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
        return (verts_A - mins) / denom

    def build_vertex_adjacency(num_verts, faces):
        """Build vertex adjacency list from triangle faces."""
        neighbors = [set() for _ in range(int(num_verts))]
        f_int = np.asarray(faces, dtype=int)
        for f in f_int:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            neighbors[i].update((j, k))
            neighbors[j].update((i, k))
            neighbors[k].update((i, j))
        return [list(s) for s in neighbors]

    def smooth_vertex_colors(faces, colors, iters=2, alpha=0.5):
        """Laplacian-like neighbor averaging to soften vertex color gradients."""
        n = int(colors.shape[0])
        adj = build_vertex_adjacency(n, faces)
        cur = colors.astype(float).copy()
        for _ in range(max(0, int(iters))):
            nxt = cur.copy()
            for i, nbrs in enumerate(adj):
                if not nbrs:
                    continue
                avg = cur[nbrs].mean(axis=0)
                nxt[i] = (1.0 - alpha) * cur[i] + alpha * avg
            cur = np.clip(nxt, 0.0, 1.0)
        return cur

    def compute_face_normals(verts_vis, triangles):
        """Compute per-face normals using cross product."""
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

    def set_axes_clean(ax):
        """Set axes to clean view with fixed bounds."""
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_axis_off()

    # Shared centering and uniform scaling (same as functional map)
    all_verts = np.vstack([vert_M, vert_N])
    shared_center = all_verts.mean(0)
    ranges = all_verts.max(0) - all_verts.min(0)
    max_range = float(ranges.max())
    scale = (1.0 / max_range) if max_range > 1e-12 else 1.0
    v_M_vis = (vert_M - shared_center) * scale
    v_N_vis = (vert_N - shared_center) * scale

    # Compute paper colormap on full mesh M (normalized XYZ over M bounds)
    colors_M = np.clip(create_paper_colormap(vert_M, vert_M), 0.0, 1.0)

    # Pullback method: colors_N_method = colors_M[matches] via pointwise matches
    colors_N_method = colors_M[matches]
    colors_N_method = smooth_vertex_colors(triv_N, colors_N_method, iters=2, alpha=0.5)

    # Ground truth pullback if available
    has_gt = gt_matches is not None and len(gt_matches) == vert_N.shape[0]
    colors_N_gt = None
    if has_gt:
        colors_N_gt = colors_M[gt_matches]
        colors_N_gt = smooth_vertex_colors(triv_N, colors_N_gt, iters=2, alpha=0.5)

    # Prepare polygons and compute per-face colors (average of vertex colors)
    poly_M = [v_M_vis[f] for f in triv_M]
    facecols_M = colors_M[triv_M].mean(axis=1)

    poly_N_method = [v_N_vis[f] for f in triv_N]
    facecols_N_method = colors_N_method[triv_N].mean(axis=1)

    poly_N_gt = None
    facecols_N_gt_avg = None
    if has_gt and colors_N_gt is not None:
        poly_N_gt = [v_N_vis[f] for f in triv_N]
        facecols_N_gt_avg = colors_N_gt[triv_N].mean(axis=1)

    # Compute face normals for all meshes
    normals_M = compute_face_normals(v_M_vis, triv_M)
    normals_N_method = compute_face_normals(v_N_vis, triv_N)
    normals_N_gt = None
    if has_gt and colors_N_gt is not None:
        normals_N_gt = compute_face_normals(v_N_vis, triv_N)

    # Lighting parameters (from functional map)
    light_dir = np.array([0.577, 0.577, 0.577])  # normalized diagonal light
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0])
    view_dir = view_dir / np.linalg.norm(view_dir)
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec)
    shininess = 128.0
    ambient = 0.25
    kd = 0.85
    ks = 0.80

    # Apply Blinn-Phong shading to full mesh
    diffuse_M = np.maximum(normals_M @ light_dir, 0.0)
    spec_M = np.maximum(normals_M @ half_vec, 0.0) ** shininess
    shading_M = ambient + kd * diffuse_M[:, None]
    facecols_M_shaded = np.clip(facecols_M * shading_M + ks * spec_M[:, None], 0.0, 1.0)

    # Apply Blinn-Phong shading to method pullback
    diffuse_N_method = np.maximum(normals_N_method @ light_dir, 0.0)
    spec_N_method = np.maximum(normals_N_method @ half_vec, 0.0) ** shininess
    shading_N_method = ambient + kd * diffuse_N_method[:, None]
    facecols_N_method_shaded = np.clip(facecols_N_method * shading_N_method + ks * spec_N_method[:, None], 0.0, 1.0)

    # Apply Blinn-Phong shading to ground truth pullback if available
    facecols_N_gt_shaded = None
    if has_gt and colors_N_gt is not None and normals_N_gt is not None:
        diffuse_N_gt = np.maximum(normals_N_gt @ light_dir, 0.0)
        spec_N_gt = np.maximum(normals_N_gt @ half_vec, 0.0) ** shininess
        shading_N_gt = ambient + kd * diffuse_N_gt[:, None]
        facecols_N_gt_shaded = np.clip(facecols_N_gt_avg * shading_N_gt + ks * spec_N_gt[:, None], 0.0, 1.0)

    # Create figure with 2 or 3 subplots depending on ground truth availability
    num_plots = 3 if (has_gt and colors_N_gt is not None) else 2
    fig = plt.figure(figsize=(6 * num_plots, 5))
    boundary_line_width = 0
    opacity = 1.0

    # Plot 1: Full mesh with paper colormap
    ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
    pc1 = Poly3DCollection(poly_M, facecolors=facecols_M_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax1.add_collection3d(pc1)
    ax1.set_title("Full Mesh (M)\nPaper Colormap")
    set_axes_clean(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(False)

    # Plot 2: Partial mesh with ground truth pullback (if available)
    if has_gt and colors_N_gt is not None and facecols_N_gt_shaded is not None:
        ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
        pc2 = Poly3DCollection(poly_N_gt, facecolors=facecols_N_gt_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
        ax2.add_collection3d(pc2)
        ax2.set_title("Partial Mesh (N)\nGround Truth Pullback")
        set_axes_clean(ax2)
        ax2.view_init(elev=20, azim=45)
        ax2.grid(False)
        plot_idx = 3
    else:
        plot_idx = 2

    # Plot 3: Partial mesh with method pullback
    ax3 = fig.add_subplot(1, num_plots, plot_idx, projection='3d')
    pc3 = Poly3DCollection(poly_N_method, facecolors=facecols_N_method_shaded, linewidths=0, edgecolor=None, alpha=opacity, shade=True, lightsource=mpl.colors.LightSource(azdeg=315, altdeg=45))
    ax3.add_collection3d(pc3)
    ax3.set_title("Partial Mesh (N)\nMethod Pullback")
    set_axes_clean(ax3)
    ax3.view_init(elev=20, azim=45)
    ax3.grid(False)

    plt.tight_layout()
    color_pullback_fname = f"color_pullback_{opts.descriptor_type}.png"
    color_pullback_path = os.path.join(output_folder, color_pullback_fname)
    plt.savefig(color_pullback_path, dpi=300)
    print(f"Saved: {color_pullback_path}")

    return color_pullback_path
