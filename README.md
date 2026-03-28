# two_layer_hnsw_like_virtual_seed

修改后的版本采用“虚拟中心点先播种 base graph”的构图逻辑：

1. 先用 KMeans 得到 `K` 个中心；
2. 先在中心层上构建中心图；
3. 再把这 `K` 个中心作为 **base graph 前 K 个虚拟节点**；
4. 后续真实数据点按 HNSW 第 0 层方式继续插入这个 graph；
5. 查询时从最近的中心虚拟点进入 base graph，并在结果里剔除虚拟节点。

## 关键变化

- `index.py`
  - 不再依赖单独的 bridge 构造。
  - `base_vectors = [centers; X]`，前 `K` 个节点是虚拟中心。
- `base_layer.py`
  - 新增 `seed_adj` / `seed_count` / `insertion_entry_points`。
  - 允许先用中心图作为初始子图，再插入真实点。
- `search()`
  - 候选集中会过滤掉前 `K` 个虚拟节点。
  - 返回的节点 id 是原始数据集下标，不包含虚拟中心。

## 使用方式

```python
from two_layer_hnsw_like import TwoLayerHNSWLikeIndex

index = TwoLayerHNSWLikeIndex(
    n_centers=32,
    m=16,
    ef_construction=200,
).fit(X)

ans = index.search(query, k=10, ef=100, n_probe_centers=3)
```
