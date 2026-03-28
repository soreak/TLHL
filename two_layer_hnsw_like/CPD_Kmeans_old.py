from dataclasses import dataclass
import numpy as np
import faiss


class CPD_KMeans:
    """
    保留原始 CPD-KMeans 逻辑：
      KMeans++ init
      assign -> PSD -> weight -> weighted center update -> converge

    关键修复：
      - PSD 与 mean_center_dist 都改成“无大矩阵、无向量临时分配”的实现
      - build-time 才跑，偏慢但能跑，不会卡死
    """

    def __init__(
        self,
        data: np.ndarray,
        k: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
        train_sample_size: int = None,
        # === 加速超参：保持 CPD 逻辑，但避免 K 很大时 O(K^2) 爆炸 ===
        psd_sample: int = 64,          # 每个中心采样多少个其它中心估计 PSD
        psd_exact_k: int = 512,        # K<=该值时走全量 PSD
        mean_sample: int = 64,         # 每个中心采样多少个其它中心估计 mean_center_dist
        mean_exact_k: int = 512,       # K<=该值时走全量 mean_center_dist
        use_gpu: bool = False,         # 预留接口（IndexFlatL2 默认 CPU）
    ):
        self.data_full = np.asarray(data, dtype=np.float32, order="C")
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state) if random_state is not None else 42
        self.train_sample_size = train_sample_size

        # sampling knobs (保持 CPD 逻辑，控制 build-time)
        self.psd_sample = int(psd_sample) if psd_sample is not None else None
        self.psd_exact_k = int(psd_exact_k)
        self.mean_sample = int(mean_sample) if mean_sample is not None else None
        self.mean_exact_k = int(mean_exact_k)
        self.use_gpu = bool(use_gpu)


        self.n_full, self.m = self.data_full.shape
        self.centers = None
        self.labels_ = None

        self._rng = np.random.RandomState(self.random_state)

        # 训练子样本
        if train_sample_size is not None and int(train_sample_size) < self.n_full:
            samp = self._rng.choice(self.n_full, int(train_sample_size), replace=False)
            self.data = self.data_full[samp]
        else:
            self.data = self.data_full

        self.n = self.data.shape[0]

        # faiss 复用
        self._assign_index = faiss.IndexFlatL2(self.m)

    # --------------------------------------------------
    # KMeans++ 初始化
    # --------------------------------------------------
    def init_cluster_center(self):
        X = self.data
        n, d = X.shape
        centers = np.empty((self.k, d), dtype=np.float32)

        centers[0] = X[self._rng.randint(n)]
        diff = X - centers[0][None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)

        for i in range(1, self.k):
            s = float(dist2.sum())
            if s <= 0.0:
                centers[i] = X[self._rng.randint(n)]
                continue
            probs = dist2 / np.float32(s)
            r = self._rng.rand()
            idx = np.searchsorted(np.cumsum(probs), r)
            if idx >= n:
                idx = n - 1
            centers[i] = X[idx]

            diff = X - centers[i][None, :]
            new_d2 = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
            dist2 = np.minimum(dist2, new_d2)

        return centers

    # --------------------------------------------------
    # 最近中心分配（FAISS）
    # --------------------------------------------------
    def assign_clusters(self, centers: np.ndarray, X: np.ndarray):
        centers = np.asarray(centers, dtype=np.float32, order="C")
        X = np.asarray(X, dtype=np.float32, order="C")

        self._assign_index.reset()
        self._assign_index.add(centers)
        _, labels = self._assign_index.search(X, 1)
        return labels.reshape(-1).astype(np.int32, copy=False)

    # --------------------------------------------------
    # Potential Difference（无分配版）
    # --------------------------------------------------

    def calculate_potential_difference(self, labels: np.ndarray, centers: np.ndarray):
        """
        Potential Difference（簇规模势差）——保留原公式含义，但做工程化加速。

        原版是对所有 (i,j) 计算 unit_vec(i,j) 并加权求和，复杂度 O(K^2 * D)；
        当 K 很大（例如 4096）会导致 build-time 过长。

        这里做“采样近似”：
          - 当 K <= self.psd_exact_k 时，走全量计算（和原版一致）。
          - 当 K >  self.psd_exact_k 时，对每个 i 只采样 psd_sample 个 j 来估计 PSD[i]。
        这仍然遵循：assign -> PSD -> weight -> weighted update 的 CPD 逻辑。
        """
        K = self.k
        centers = np.asarray(centers, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)

        cluster_sizes = np.bincount(labels, minlength=K).astype(np.int32)
        # size_diff[i,j] = size_j - size_i （和原实现一致）
        # 原代码: size_diff = cluster_sizes[None,:] - cluster_sizes[:,None]
        # 我们不显式构造 KxK（太大），而是按需取 size_j - size_i。
        PSD = np.zeros_like(centers, dtype=np.float32)

        # 超参数
        exact_k = int(self.psd_exact_k)
        psd_samp = int(self.psd_sample) if self.psd_sample is not None else K

        if K <= exact_k:
            # 全量：与原公式等价，但不构造 (K,K,D) 张量
            for i in range(K):
                ci = centers[i]
                si = cluster_sizes[i]
                for j in range(K):
                    if i == j:
                        continue
                    sj = cluster_sizes[j]
                    sdiff = sj - si
                    if sdiff == 0:
                        continue
                    # sign：若 sdiff<0 翻转
                    sign = -1.0 if sdiff < 0 else 1.0
                    v = (centers[j] - ci) * np.float32(sign)
                    norm = np.sqrt(np.dot(v, v))
                    if norm <= 1e-10:
                        continue
                    unit = v / np.float32(norm)
                    w = np.float32(abs(sdiff) / float(self.n))
                    PSD[i] += unit * w
            return PSD

        # 采样近似（K 很大时）
        for i in range(K):
            ci = centers[i]
            si = cluster_sizes[i]

            # 采样 psd_samp 个其它中心
            m = psd_samp
            if m >= K - 1:
                # 极端：采样数>=K，退化为全量（但此分支 K>exact_k 很少用）
                for j in range(K):
                    if i == j:
                        continue
                    sj = cluster_sizes[j]
                    sdiff = sj - si
                    if sdiff == 0:
                        continue
                    sign = -1.0 if sdiff < 0 else 1.0
                    v = (centers[j] - ci) * np.float32(sign)
                    norm = np.sqrt(np.dot(v, v))
                    if norm <= 1e-10:
                        continue
                    unit = v / np.float32(norm)
                    w = np.float32(abs(sdiff) / float(self.n))
                    PSD[i] += unit * w
                continue

            # 抽样：允许重复会有轻微噪声，但更快（build-time 友好）
            for _ in range(m):
                j = int(self._rng.randint(0, K - 1))
                if j >= i:
                    j += 1  # 跳过 i
                sj = cluster_sizes[j]
                sdiff = sj - si
                if sdiff == 0:
                    continue

                sign = -1.0 if sdiff < 0 else 1.0
                v = (centers[j] - ci) * np.float32(sign)
                norm = np.sqrt(np.dot(v, v))
                if norm <= 1e-10:
                    continue
                unit = v / np.float32(norm)
                w = np.float32(abs(sdiff) / float(self.n))
                PSD[i] += unit * w

            # 归一化一下，避免采样数改变 PSD 尺度（保持和全量同量级）
            PSD[i] *= np.float32(float(K - 1) / float(m))

        return PSD

    def _mean_center_dist(self, centers: np.ndarray):
        K, D = centers.shape
        mean_dist = np.zeros((K,), dtype=np.float32)

        for i in range(K):
            ci = centers[i]
            s = np.float32(0.0)
            for j in range(K):
                if i == j:
                    continue
                cj = centers[j]
                norm2 = np.float32(0.0)
                for d in range(D):
                    diff = cj[d] - ci[d]
                    norm2 += diff * diff
                s += np.float32(np.sqrt(norm2))
            denom = np.float32(max(K - 1, 1))
            mean_dist[i] = s / denom

        return mean_dist

    # --------------------------------------------------
    # 样本权重计算（保留原逻辑，但避免 K×K 矩阵）
    # --------------------------------------------------

    def calculate_weight(self, labels: np.ndarray, centers: np.ndarray, PSD: np.ndarray):
        """
        样本权重计算（PSD 驱动）——保留原逻辑，但把 O(K^2) 的中心距离均值改为采样估计。

        原版：mean_center_dist(i) = avg_j ||c_j - c_i|| （j!=i）
        当 K 很大时构造 KxK 距离矩阵会非常慢且占内存。

        这里：
          - K <= self.mean_exact_k：走全量（但不显式存 KxK 矩阵）
          - K >  self.mean_exact_k：对每个 i 采样 mean_sample 个中心估计 mean_center_dist(i)
        """
        X = self.data
        K = self.k
        n = X.shape[0]

        centers = np.asarray(centers, dtype=np.float32, order="C")
        PSD = np.asarray(PSD, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)

        weight = np.ones(n, dtype=np.float32)

        # ---- 预估 mean_center_dist(i) ----
        mean_center_dist = np.zeros((K,), dtype=np.float32)
        exact_k = int(self.mean_exact_k)
        mean_samp = int(self.mean_sample) if self.mean_sample is not None else K

        if K <= exact_k:
            # 全量：对每个 i 扫一遍所有 j（不构造矩阵）
            for i in range(K):
                ci = centers[i]
                acc = np.float32(0.0)
                cnt = 0
                for j in range(K):
                    if i == j:
                        continue
                    diff = centers[j] - ci
                    acc += np.float32(np.sqrt(np.dot(diff, diff)))
                    cnt += 1
                if cnt > 0:
                    mean_center_dist[i] = acc / np.float32(cnt)
        else:
            # 采样估计
            m = mean_samp
            if m >= K - 1:
                m = K - 1
            for i in range(K):
                ci = centers[i]
                acc = np.float32(0.0)
                for _ in range(m):
                    j = int(self._rng.randint(0, K - 1))
                    if j >= i:
                        j += 1
                    diff = centers[j] - ci
                    acc += np.float32(np.sqrt(np.dot(diff, diff)))
                mean_center_dist[i] = acc / np.float32(max(m, 1))

        # ---- 原逻辑：cos>0 -> 1 + 0.5*sqrt(mean_center_dist)*psd_norm ----
        for i in range(K):
            idx = np.where(labels == i)[0]
            if idx.size == 0:
                continue

            psd = PSD[i]
            psd_norm = float(np.sqrt(np.dot(psd, psd)))
            if psd_norm <= 0.0:
                continue

            Xi = X[idx] - centers[i]  # (Mi,D)
            x_norm = np.sqrt(np.einsum("ij,ij->i", Xi, Xi)).astype(np.float32)
            x_norm = np.maximum(x_norm, 1e-10)

            cos = (Xi @ psd) / (x_norm * np.float32(psd_norm))

            gain = np.float32(1.0) + np.float32(0.5) * np.float32(np.sqrt(mean_center_dist[i])) * np.float32(psd_norm)
            weight[idx] = np.where(cos > 0, gain, np.float32(1.0))

        return weight

    def compute_cluster_centers(self, labels: np.ndarray, weight: np.ndarray):
        X = self.data
        K = self.k
        d = self.m

        centers = np.zeros((K, d), dtype=np.float32)

        for j in range(K):
            mask = (labels == j)
            if not np.any(mask):
                centers[j] = X[self._rng.randint(self.n)]
                continue

            w = weight[mask][:, None]
            sw = float(w.sum())
            if sw <= 1e-12:
                centers[j] = X[self._rng.randint(self.n)]
                continue
            centers[j] = (X[mask] * w).sum(axis=0) / np.float32(sw)

        return centers

    def has_converged(self, old: np.ndarray, new: np.ndarray):
        return float(np.linalg.norm(new - old)) < self.tol

    # --------------------------------------------------
    # train：训练集上跑 CPD-KMeans；最终对全量 assign labels
    # --------------------------------------------------
    def train(self):
        centers = self.init_cluster_center().astype(np.float32, copy=False)

        for _ in range(self.max_iter):
            old = centers.copy()

            labels = self.assign_clusters(centers, self.data)
            PSD = self.calculate_potential_difference(labels, centers)
            weight = self.calculate_weight(labels, centers, PSD)
            centers = self.compute_cluster_centers(labels, weight)

            if self.has_converged(old, centers):
                break

        self.centers = centers.astype(np.float32, copy=False)
        self.labels_ = self.assign_clusters(self.centers, self.data_full)  # ✅ 全量标签
        return self.centers, self.labels_


