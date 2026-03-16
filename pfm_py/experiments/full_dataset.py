import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import open3d as o3d
from scipy.sparse.csgraph import dijkstra
from dataclasses import dataclass
from contextlib import redirect_stdout
import io

from pfm_py.manifold_mesh import ManifoldMesh
from pfm_py.match_part_to_whole import match_and_refine
from pfm_py.options import Options

DIFF3F_AVAILABLE = False
_diff3f_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Diffusion-3D-Features'),
    os.path.join(os.getcwd(), 'Diffusion-3D-Features'),
]
for _p in _diff3f_paths:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from functional_map import compute_surface_map
    DIFF3F_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MeshPair:
    name: str
    full_mesh: str
    partial_mesh: str
    ground_truth: str
    folder: str


def get_all_pairs(data_path):
    pairs = []
    for folder in ["cuts", "holes"]:
        for partial_path in sorted(glob.glob(f"{data_path}/SHREC16/{folder}/off/*.off")):
            name = os.path.splitext(os.path.basename(partial_path))[0]
            animal = name.split('_')[1]
            pair = MeshPair(
                name=name,
                full_mesh=f"{data_path}/SHREC16/null/off/{animal}.off",
                partial_mesh=partial_path,
                ground_truth=f"{data_path}/SHREC16/{folder}/corres/{name}.vts",
                folder=folder,
            )
            if os.path.exists(pair.full_mesh) and os.path.exists(pair.ground_truth):
                pairs.append(pair)
    print(f"Found {len(pairs)} pairs ({sum(p.folder=='cuts' for p in pairs)} cuts, {sum(p.folder=='holes' for p in pairs)} holes)")
    return pairs


def geodesic_matrix(vertices, faces):
    n = len(vertices)
    edges = set()
    for f in faces:
        edges |= {tuple(sorted([f[0], f[1]])), tuple(sorted([f[1], f[2]])), tuple(sorted([f[2], f[0]]))}
    graph = np.full((n, n), np.inf)
    for i, j in edges:
        d = np.linalg.norm(vertices[i] - vertices[j])
        graph[i, j] = graph[j, i] = d
    return dijkstra(graph, directed=False)


def argmax_correspondences(desc_M, desc_N):
    dn = desc_N / (torch.norm(desc_N, dim=1, keepdim=True) + 1e-8)
    dm = desc_M / (torch.norm(desc_M, dim=1, keepdim=True) + 1e-8)
    return torch.argmax(dn @ dm.T, dim=1).cpu().numpy()


def icp_baseline(v_M, v_N, gt_corr, geo_M, area_M):
    import scipy.spatial
    pcd_M, pcd_N = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_M.points = o3d.utility.Vector3dVector(v_M)
    pcd_N.points = o3d.utility.Vector3dVector(v_N)
    pcd_M.estimate_normals()
    pcd_N.estimate_normals()
    r = 0.05 * np.linalg.norm(v_M.max(0) - v_M.min(0))
    fpfh_M = o3d.pipelines.registration.compute_fpfh_feature(pcd_M, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    fpfh_N = o3d.pipelines.registration.compute_fpfh_feature(pcd_N, o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=100))
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_N, pcd_M, fpfh_N, fpfh_M, True, 0.05,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    icp = o3d.pipelines.registration.registration_icp(
        pcd_N, pcd_M, 0.02, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    v_N_aligned = (icp.transformation[:3, :3] @ v_N.T + icp.transformation[:3, 3:4]).T
    _, corr = scipy.spatial.cKDTree(v_M).query(v_N_aligned)
    err = np.array([geo_M[gt_corr[i], corr[i]] / area_M for i in range(len(v_N))])
    return corr, err


def mge(corr, gt_corr, geo_M, area_M):
    return np.array([geo_M[gt_corr[i], corr[i]] / area_M for i in range(len(gt_corr))]).mean()


def cache_path(output_dir, name):
    return os.path.join(output_dir, f"{name}.json")


def load_cached(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cache(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def run_pfm(M, N, opts):
    with redirect_stdout(io.StringIO()):
        C, v, corr = match_and_refine(M, N, opts)
    return corr.numpy(force=True)


def run_pair(pair: MeshPair, opts: Options, output_dir: str):
    cp = cache_path(output_dir, pair.name)
    cached = load_cached(cp)
    if cached:
        print(f"  [cached] {pair.name}")
        return cached

    mesh_M = o3d.io.read_triangle_mesh(pair.full_mesh)
    mesh_N = o3d.io.read_triangle_mesh(pair.partial_mesh)
    v_M, f_M = np.asarray(mesh_M.vertices), np.asarray(mesh_M.triangles)
    v_N, f_N = np.asarray(mesh_N.vertices), np.asarray(mesh_N.triangles)

    print(f"  {pair.name}: M={len(v_M)}v, N={len(v_N)}v")

    gt_corr = np.loadtxt(pair.ground_truth, dtype=float).astype(int) - 1

    print("    geodesics...")
    geo_M = geodesic_matrix(v_M, f_M)
    area_M = np.sqrt(0.5 * np.linalg.norm(
        np.cross(v_M[f_M[:,1]] - v_M[f_M[:,0]], v_M[f_M[:,2]] - v_M[f_M[:,0]]), axis=1).sum())

    print("    ICP...")
    icp_corr, icp_err = icp_baseline(v_M, v_N, gt_corr, geo_M, area_M)

    result = {
        'name': pair.name,
        'folder': pair.folder,
        'icp': float(icp_err.mean()),
    }

    for desc in ['shot', 'dino', 'fpfh', 'shot+dino']:
        print(f"    {desc}...")
        opts.descriptor_type = desc
        result[f'{desc}_argmax'] = None
        result[f'{desc}_fm'] = None
        result[f'{desc}_pfm'] = None

        try:
            M_mesh = ManifoldMesh(v_M, f_M, opts, compute_geo=True)
            N_mesh = ManifoldMesh(v_N, f_N, opts, compute_geo=False)
            desc_M, desc_N = ManifoldMesh.compute_compatible_descriptors(M_mesh, N_mesh, opts)

            corr_argmax = argmax_correspondences(desc_M, desc_N)
            result[f'{desc}_argmax'] = float(mge(corr_argmax, gt_corr, geo_M, area_M))

            if DIFF3F_AVAILABLE:
                try:
                    corr_fm = compute_surface_map(pair.full_mesh, pair.partial_mesh,
                                                   desc_M.cpu().numpy(), desc_N.cpu().numpy()).cpu().numpy()
                    result[f'{desc}_fm'] = float(mge(corr_fm, gt_corr, geo_M, area_M))
                except Exception as e:
                    print(f"      FM failed: {e}")

            M_pfm = ManifoldMesh(v_M, f_M, opts, compute_geo=True)
            N_pfm = ManifoldMesh(v_N, f_N, opts, compute_geo=False)
            corr_pfm = run_pfm(M_pfm, N_pfm, opts)
            result[f'{desc}_pfm'] = float(mge(corr_pfm, gt_corr, geo_M, area_M))

            print(f"      argmax={result[f'{desc}_argmax']:.4f}  "
                  f"fm={result.get(f'{desc}_fm') or 'N/A'}  "
                  f"pfm={result[f'{desc}_pfm']:.4f}")
        except Exception as e:
            print(f"      [ERROR] {desc}: {e}")

    save_cache(cp, result)
    return result


def write_summary(results, output_dir):
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")

    descs = ['shot', 'dino', 'fpfh', 'shot+dino']
    methods = ['argmax', 'fm', 'pfm']

    header = ['name', 'folder', 'icp'] + [f'{d}_{m}' for d in descs for m in methods]
    rows = []
    for r in results:
        row = [r['name'], r['folder'], f"{r['icp']:.4f}"]
        for d in descs:
            for m in methods:
                v = r.get(f'{d}_{m}')
                row.append(f"{v:.4f}" if v is not None else '-')
        rows.append(row)

    avg_row = ['AVERAGE', '-']
    avg_row.append(f"{np.mean([r['icp'] for r in results]):.4f}")
    for d in descs:
        for m in methods:
            vals = [r[f'{d}_{m}'] for r in results if r.get(f'{d}_{m}') is not None]
            avg_row.append(f"{np.mean(vals):.4f}" if vals else '-')

    col_widths = [max(len(str(row[i])) for row in [header] + rows + [avg_row]) for i in range(len(header))]
    fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)

    print('\n' + fmt.format(*header))
    print('-' * sum(col_widths + [2*(len(header)-1)]))
    for row in rows:
        print(fmt.format(*row))
    print('-' * sum(col_widths + [2*(len(header)-1)]))
    print(fmt.format(*avg_row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/usr/prakt/w0010/SAVHA/shape_data')
    parser.add_argument('--output', default='benchmark_v2_results')
    parser.add_argument('--diff3f-path', default=None)
    args = parser.parse_args()

    if args.diff3f_path and os.path.isdir(args.diff3f_path):
        sys.path.insert(0, args.diff3f_path)
        try:
            from functional_map import compute_surface_map
            DIFF3F_AVAILABLE = True
        except ImportError:
            pass

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Diff3F: {DIFF3F_AVAILABLE}")

    opts = Options(device)
    pairs = get_all_pairs(args.data_path)

    os.makedirs(args.output, exist_ok=True)

    # load any results already cached from previous runs
    all_results = []
    for pair in pairs:
        cp = cache_path(os.path.join(args.output, pair.folder), pair.name)
        cached = load_cached(cp)
        if cached:
            all_results.append(cached)

    seen = {r['name'] for r in all_results}
    for pair in pairs:
        pair_dir = os.path.join(args.output, pair.folder)
        try:
            result = run_pair(pair, opts, pair_dir)
        except Exception as e:
            print(f"  [ERROR] {pair.name}: {e}")
            continue
        if pair.name not in seen:
            all_results.append(result)
            seen.add(pair.name)
        write_summary(all_results, args.output)
    print("\nDone.")
