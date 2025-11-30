import numpy as np
from dataclasses import dataclass
from numpy.linalg import norm
from scipy.spatial import cKDTree
import math

# 角度常量（和 C++ 代码一致）
DEG_45_TO_RAD  = 0.7853981633974483
DEG_90_TO_RAD  = 1.5707963267948966
DEG_135_TO_RAD = 2.356194490192345
DEG_168_TO_RAD = 2.748893571891069


@dataclass
class SHOTParams:
    radius: float = 15.0          # 邻域半径
    localRFradius: float = None   # LRF 半径，不设的话 = radius
    bins: int = 10                # cosine 量化 bins
    doubleVolumes: bool = True
    useInterpolation: bool = True
    useNormalization: bool = True
    minNeighbors: int = 10

    def __post_init__(self):
        if self.localRFradius is None:
            self.localRFradius = self.radius


class SHOTDescriptor:
    def __init__(self, params: SHOTParams):
        self.params = params
        if params.doubleVolumes:
            self.m_k = 32
        else:
            self.m_k = 16
        self.desc_length = self.m_k * (params.bins + 1)

        self.vertices = None
        self.normals = None
        self.tree = None
        self.adj = None       # ★ mesh adjacency

    def get_descriptor_length(self):
        return self.desc_length

    # ---------- 绑定点云 + 法向 & 建 KDTree ----------
    def set_data(self, vertices, normals, faces=None):
        """
        vertices: (N,3)
        normals:  (N,3)
        faces:    (F,3) int, 如果给了就用 mesh 邻接；否则退回 KDTree
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.normals = np.asarray(normals, dtype=float)
        assert self.vertices.shape == self.normals.shape

        if faces is not None:
            faces = np.asarray(faces, dtype=int)
            self.adj = self._build_adjacency(faces, self.vertices.shape[0])
            self.tree = None
        else:
            self.adj = None
            self.tree = cKDTree(self.vertices)

    def _build_adjacency(self, faces, n_vertices):
        """根据三角形列表构建顶点邻接表"""
        adj = [set() for _ in range(n_vertices)]
        for tri in faces:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            adj[i].update((j, k))
            adj[j].update((i, k))
            adj[k].update((i, j))
        # 转成 numpy 数组，后面遍历更快一点
        return [np.fromiter(neigh, dtype=int) for neigh in adj]

    # ---------- 邻域搜索 ----------
    def nearest_neighbors_with_dist(self, center_idx, radius):
        """如果有 mesh 邻接，就用 BFS；否则退回 KDTree 半径搜索"""
        if self.adj is not None:
            return self._nearest_neighbors_mesh(center_idx, radius)
        else:
            return self._nearest_neighbors_kdtree(center_idx, radius)

    # ---------- KDTree 版本（原来的实现，留作 fallback） ----------
    def _nearest_neighbors_kdtree(self, center_idx, radius):
        pts = self.vertices
        center = pts[center_idx]
        idx = self.tree.query_ball_point(center, r=radius)
        idx = [i for i in idx if i != center_idx]
        if not idx:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.asarray(idx, dtype=int)
        dists = norm(pts[idx] - center, axis=1)
        return idx, dists

    # ---------- mesh-based BFS 版本 ----------
    def _nearest_neighbors_mesh(self, center_idx, radius):
        """
        基于 mesh 邻接的 BFS：
        - 从 center_idx 出发，在邻接图上做 BFS
        - 按欧氏距离 r 进行截断
        """
        verts = self.vertices
        adj = self.adj
        N = verts.shape[0]

        visited = np.zeros(N, dtype=bool)
        q = [center_idx]
        visited[center_idx] = True

        neighbors = []
        dists = []
        r2 = radius * radius
        c = verts[center_idx]

        while q:
            i = q.pop(0)
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                d2 = float(np.dot(verts[j] - c, verts[j] - c))
                # 只要在半径内，就记为邻居并继续往外扩
                if d2 <= r2:
                    neighbors.append(j)
                    dists.append(math.sqrt(d2))
                    q.append(j)

        if not neighbors:
            return np.array([], dtype=int), np.array([], dtype=float)

        return np.asarray(neighbors, dtype=int), np.asarray(dists, dtype=float)

    # ---------- LRF 计算（对应 getSHOTLocalRF） ----------
    def get_local_rf(self, center_idx, neigh_idx, neigh_dists, radius):
        pts = self.vertices
        pt = pts[center_idx]
        neigh_idx = np.asarray(neigh_idx, dtype=int)
        neigh_dists = np.asarray(neigh_dists, dtype=float)

        if len(neigh_idx) < 5:
            raise ValueError("Not enough points for computing SHOT local RF")

        q = pts[neigh_idx] - pt           # (k,3)
        w = radius - neigh_dists          # 线性权重
        w = np.clip(w, 1e-8, None)

        M = (q.T * w) @ q / np.sum(w)     # 3x3
        evals, evecs = np.linalg.eigh(M)  # evals 升序, evecs 列为特征向量

        x, y, z = 2, 1, 0   # 最大特征值 → X, 最小 → Z
        if not (evals[x] >= evals[y] >= evals[z]):
            raise ValueError("Eigenvalues are not in decreasing order")

        X = evecs[:, x].copy()
        Z = evecs[:, z].copy()

        proj_X = (pts[neigh_idx] - pt) @ X
        proj_Z = (pts[neigh_idx] - pt) @ Z
        posx = np.sum(proj_X >= 0)
        posz = np.sum(proj_Z >= 0)
        if posx < len(neigh_idx) - posx:
            X = -X
        if posz < len(neigh_idx) - posz:
            Z = -Z

        Y = np.cross(Z, X)

        X /= norm(X) + 1e-12
        Y /= norm(Y) + 1e-12
        Z /= norm(Z) + 1e-12
        return X, Y, Z

    # ---------- 单点 SHOT 描述 ----------
    def describe_point(self, center_idx: int):
        # 既没有 vertices/normals，或者既没有 KDTree 也没有 mesh 邻接，才报错
        if (
          self.vertices is None
          or self.normals is None
          or (self.tree is None and self.adj is None)
        ):
          raise RuntimeError("Call set_data(vertices, normals, faces) before describe_point")
        params = self.params
        radius = params.radius
        desc = np.zeros(self.desc_length, dtype=float)

        # ---------- 选邻域 & LRF ----------
        if abs(params.localRFradius - params.radius) < 1e-6:
            neigh_idx, dists = self.nearest_neighbors_with_dist(center_idx, radius)
            if len(neigh_idx) < params.minNeighbors:
                return desc
            try:
                X, Y, Z = self.get_local_rf(center_idx, neigh_idx, dists, params.localRFradius)
            except Exception:
                return desc
        else:
            neigh_idx_L, dists_L = self.nearest_neighbors_with_dist(center_idx, params.localRFradius)
            try:
                X, Y, Z = self.get_local_rf(center_idx, neigh_idx_L, dists_L, params.localRFradius)
            except Exception:
                return desc
            neigh_idx, dists = self.nearest_neighbors_with_dist(center_idx, radius)
            if len(neigh_idx) < params.minNeighbors:
                return desc

        vertices = self.vertices
        normals = self.normals
        centralPoint = vertices[center_idx]

        sq_radius = radius * radius
        sqradius4 = sq_radius / 4.0
        radius3_4 = (radius * 3.0) / 4.0
        radius1_4 = radius / 4.0
        radius1_2 = radius / 2.0

        maxAngularSectors = 28 if params.doubleVolumes else 12

        # ---------- 遍历邻居 ----------
        for ni in neigh_idx:
            q = vertices[ni] - centralPoint
            distance = np.dot(q, q)
            sqrtSqDistance = math.sqrt(distance)
            if abs(distance) < 1e-14:
                continue

            normal = normals[ni]

            cosineDesc = np.dot(Z, normal)
            cosineDesc = max(-1.0, min(1.0, float(cosineDesc)))

            xInFeatRef = np.dot(q, X)
            yInFeatRef = np.dot(q, Y)
            zInFeatRef = np.dot(q, Z)

            if abs(xInFeatRef) < 1e-30: xInFeatRef = 0.0
            if abs(yInFeatRef) < 1e-30: yInFeatRef = 0.0
            if abs(zInFeatRef) < 1e-30: zInFeatRef = 0.0

            # ---- desc_index: 空间 volume 编号 ----
            bit4 = 1 if ((yInFeatRef > 0) or ((yInFeatRef == 0.0) and (xInFeatRef < 0.0))) else 0
            if (xInFeatRef > 0) or ((xInFeatRef == 0.0) and (yInFeatRef > 0.0)):
                bit3 = 0 if bit4 == 1 else 1  # !bit4
            else:
                bit3 = bit4

            desc_index = (bit4 << 3) + (bit3 << 2)

            if params.doubleVolumes:
                desc_index = desc_index << 1
                if (xInFeatRef * yInFeatRef > 0) or (xInFeatRef == 0.0):
                    desc_index += 0 if (abs(xInFeatRef) >= abs(yInFeatRef)) else 4
                else:
                    desc_index += 4 if (abs(xInFeatRef) > abs(yInFeatRef)) else 0

            if zInFeatRef > 0:
                desc_index += 1

            if sqrtSqDistance > radius1_2:
                desc_index += 2

            # ---- normal → cosine bins ----
            binDistance = ((1.0 + cosineDesc) * params.bins) / 2.0
            if binDistance < 0.0:
                step_index = math.ceil(binDistance - 0.5)
            else:
                step_index = math.floor(binDistance + 0.5)

            volume_index = desc_index * (params.bins + 1)
            weight = 1.0

            # ---- 插值 ----
            if params.useInterpolation:
                intWeight = 0.0

                # 1) normal 插值
                binDistance -= step_index
                intWeight += (1 - abs(binDistance))

                if binDistance > 0:
                    idx = volume_index + ((step_index + 1) % params.bins)
                    if 0 <= idx < self.desc_length:
                        desc[idx] += binDistance * weight
                else:
                    idx = volume_index + ((step_index - 1 + params.bins) % params.bins)
                    if 0 <= idx < self.desc_length:
                        desc[idx] += -binDistance * weight

                # 2) radius 插值
                if sqrtSqDistance > radius1_2:
                    radiusDistance = (sqrtSqDistance - radius3_4) / radius1_2
                    if sqrtSqDistance > radius3_4:
                        intWeight += 1 - radiusDistance
                    else:
                        intWeight += 1 + radiusDistance
                        idx = (desc_index - 2) * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * (-radiusDistance)
                else:
                    radiusDistance = (sqrtSqDistance - radius1_4) / radius1_2
                    if sqrtSqDistance < radius1_4:
                        intWeight += 1 + radiusDistance
                    else:
                        intWeight += 1 - radiusDistance
                        idx = (desc_index + 2) * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * radiusDistance

                # 3) inclination 仰角插值
                inclinationCos = zInFeatRef / sqrtSqDistance
                inclinationCos = max(-1.0, min(1.0, float(inclinationCos)))
                inclination = math.acos(inclinationCos)

                if inclination > DEG_90_TO_RAD or (abs(inclination - DEG_90_TO_RAD) < 1e-6 and zInFeatRef <= 0):
                    inclinationDistance = (inclination - DEG_135_TO_RAD) / DEG_90_TO_RAD
                    if inclination > DEG_135_TO_RAD:
                        intWeight += 1 - inclinationDistance
                    else:
                        intWeight += 1 + inclinationDistance
                        idx = (desc_index + 1) * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * (-inclinationDistance)
                else:
                    inclinationDistance = (inclination - DEG_45_TO_RAD) / DEG_90_TO_RAD
                    if inclination < DEG_45_TO_RAD:
                        intWeight += 1 + inclinationDistance
                    else:
                        intWeight += 1 - inclinationDistance
                        idx = (desc_index - 1) * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * inclinationDistance

                # 4) azimuth 方位角插值
                if not (yInFeatRef == 0.0 and xInFeatRef == 0.0):
                    azimuth = math.atan2(yInFeatRef, xInFeatRef)

                    sel = desc_index >> 2
                    angularSectorSpan = DEG_45_TO_RAD if params.doubleVolumes else DEG_90_TO_RAD
                    angularSectorStart = -DEG_168_TO_RAD if params.doubleVolumes else -DEG_135_TO_RAD

                    azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan * sel)) / angularSectorSpan
                    azimuthDistance = max(-0.5, min(azimuthDistance, 0.5))

                    if azimuthDistance > 0:
                        intWeight += 1 - azimuthDistance
                        interp_index = (desc_index + 4) % maxAngularSectors
                        idx = interp_index * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * azimuthDistance
                    else:
                        interp_index = (desc_index - 4 + maxAngularSectors) % maxAngularSectors
                        intWeight += 1 + azimuthDistance
                        idx = interp_index * (params.bins + 1) + step_index
                        if 0 <= idx < self.desc_length:
                            desc[idx] += weight * (-azimuthDistance)

                # 当前 volume 主 bin
                idx = volume_index + step_index
                if 0 <= idx < self.desc_length:
                    desc[idx] += weight * intWeight

            else:
                idx = volume_index + step_index
                if 0 <= idx < self.desc_length:
                    desc[idx] += weight

        # ---------- L2 归一化 ----------
        if params.useNormalization:
            accNorm = math.sqrt(float(np.dot(desc, desc)))
            if accNorm > 1e-12:
                desc /= accNorm

        return desc

    def describe_all(self, query_idx=None):
        if query_idx is None:
            query_idx = np.arange(self.vertices.shape[0], dtype=int)
        else:
            query_idx = np.asarray(query_idx, dtype=int)

        all_desc = np.zeros((len(query_idx), self.desc_length), dtype=float)
        for k, idx in enumerate(query_idx):
            all_desc[k] = self.describe_point(idx)
        return all_desc