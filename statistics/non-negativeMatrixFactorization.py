import ast
import math
import heapq
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


# =========================================================
# 0) 读入矩阵 & 解析节点名（你的索引形如 "('提出','知识协作类')"）
# =========================================================
def parse_label(x):
    """把 "('提出','知识协作类')" 解析成 ('提出','知识协作类')；失败则退化为 (str(x), '')"""
    if isinstance(x, tuple) and len(x) == 2:
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            t = ast.literal_eval(s)
            if isinstance(t, tuple) and len(t) == 2:
                return t
        except Exception:
            pass
    return (str(x), "")


def label_to_str(t):
    a, b = t
    return f"{a}|{b}" if b else str(a)


def load_transition_matrix(file_path, sheet_name=0):
    """
    读取转移频次矩阵 A (n x n)，并返回节点名列表 node_names。
    支持 Excel 和 CSV 格式，自动根据文件扩展名判断。
    默认假设首列是 index，首行是 columns（和你文件一致）。
    """
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.csv'):
        df = pd.read_csv(file_path, header=0, index_col=0, encoding="utf-8-sig")
    elif file_path_lower.endswith(('.xlsx', '.xls', '.xlsm')):
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0, index_col=0)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}。请使用 .csv, .xlsx, .xls 或 .xlsm 格式。")
    df = df.fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)

    # 如果行列名集合一致，则强制对齐成同一顺序的方阵
    if set(df.index) == set(df.columns):
        df = df.loc[df.index, df.index]

    # 非负
    df[df < 0] = 0

    node_names = [label_to_str(parse_label(x)) for x in df.index.tolist()]
    A = df.values.astype(float)
    return A, node_names


# =========================================================
# 1) NMF 去噪：A* = W H （建议 KL 损失适配计数矩阵）
# =========================================================
def nmf_denoise(A, rank=8, random_state=0, max_iter=2000):
    X = csr_matrix(A)
    nmf = NMF(
        n_components=rank,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=max_iter,
        random_state=random_state,
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    A_star = W @ H
    return A_star, W, H, nmf


def row_normalize(A_star):
    """把 A* 行归一化为转移概率 P*（每行和为 1；空行全 0）"""
    rs = A_star.sum(axis=1, keepdims=True)
    P = np.divide(A_star, rs, out=np.zeros_like(A_star), where=rs > 0)
    return P


# =========================================================
# 2) 构图：为避免 A* 稠密，按每个节点保留 top_out 条出边
#    在 P* 上找“最大乘积概率路径” ⇔ 在 -log(P*) 上找最短路
# =========================================================
def build_sparse_adj(P, top_out=20, p_min=1e-6, remove_self=True):
    """
    返回邻接表 adj[u] = [(v, cost, p), ...]
    cost = -log(p) >= 0（因 P 行归一化，p∈(0,1]）
    """
    n = P.shape[0]
    adj = [[] for _ in range(n)]

    for i in range(n):
        row = P[i].copy()
        if remove_self:
            row[i] = 0.0

        # 选 top_out 个候选后继（比全量快很多）
        if top_out is not None and top_out < n:
            idx = np.argpartition(row, -top_out)[-top_out:]
            idx = idx[np.argsort(row[idx])[::-1]]
        else:
            idx = np.argsort(row)[::-1]

        for j in idx:
            p = float(row[j])
            if p <= p_min:
                continue
            cost = -math.log(p)
            adj[i].append((int(j), cost, p))

    return adj


def dijkstra_best_path(adj, s, t):
    """
    在 cost=-log(p) 上做最短路，得到最大概率路径（最大乘积概率）。
    返回：path(list[int]), log_prob, prob
    """
    n = len(adj)
    INF = 1e100
    dist = [INF] * n
    prev = [None] * n
    dist[s] = 0.0
    pq = [(0.0, s)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == t:
            break
        for v, cost, _p in adj[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[t] >= INF / 2:
        return None, None, None

    # 复原路径
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path = path[::-1]

    log_prob = -dist[t]
    prob = math.exp(log_prob) if log_prob > -745 else 0.0  # 防 underflow
    return path, log_prob, prob


# =========================================================
# 3) 全局候选源/汇选择：用 A* 的 out_strength / in_strength
# =========================================================
def pick_sources_sinks(A_star, n_sources=12, n_sinks=12, min_strength=1e-9):
    out_strength = A_star.sum(axis=1)
    in_strength = A_star.sum(axis=0)

    src_idx = np.where(out_strength > min_strength)[0]
    snk_idx = np.where(in_strength > min_strength)[0]

    sources = src_idx[np.argsort(out_strength[src_idx])[::-1][:n_sources]]
    sinks = snk_idx[np.argsort(in_strength[snk_idx])[::-1][:n_sinks]]
    return list(map(int, sources)), list(map(int, sinks)), out_strength, in_strength


def names_to_indices(node_names, names):
    """把节点字符串名映射到索引（找不到会忽略）"""
    mp = {n: i for i, n in enumerate(node_names)}
    return [mp[x] for x in names if x in mp]


# =========================================================
# 4) 抽取全局主干路径（Dijkstra：不限制长度）
# =========================================================
def top_global_paths(adj, node_names, sources, sinks, top_k=30, min_steps=1, max_steps=None):
    rows = []
    for s in sources:
        for t in sinks:
            if s == t:
                continue
            path, logp, p = dijkstra_best_path(adj, s, t)
            if path is None:
                continue
            steps = len(path) - 1
            if steps < min_steps:
                continue
            if max_steps is not None and steps > max_steps:
                continue

            rows.append(
                {
                    "source": node_names[s],
                    "sink": node_names[t],
                    "steps": steps,
                    "log_prob": float(logp),
                    "prob": float(p),
                    "path": " → ".join(node_names[i] for i in path),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("log_prob", ascending=False)
    return df.head(top_k).reset_index(drop=True)


# =========================================================
# 5) 主干骨架边：统计 Top 路径中边出现频次 + 累积流量 A*[u,v]
# =========================================================
def backbone_edges_from_paths(paths_df, A_star, node_names):
    if paths_df.empty:
        return pd.DataFrame()

    name_to_idx = {name: i for i, name in enumerate(node_names)}
    usage = {}
    flow_sum = {}

    for p in paths_df["path"].tolist():
        nodes = p.split(" → ")
        idx = [name_to_idx[x] for x in nodes if x in name_to_idx]
        for u, v in zip(idx[:-1], idx[1:]):
            key = (u, v)
            usage[key] = usage.get(key, 0) + 1
            flow_sum[key] = flow_sum.get(key, 0.0) + float(A_star[u, v])

    rows = []
    for (u, v), cnt in usage.items():
        rows.append(
            {
                "from": node_names[u],
                "to": node_names[v],
                "used_in_top_paths": cnt,
                "denoised_flow(A*)": flow_sum[(u, v)],
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(["used_in_top_paths", "denoised_flow(A*)"], ascending=False)
        .reset_index(drop=True)
    )
    return out


# =========================================================
# 6) 固定长度主干路径（可强制更“链式”）：Beam Search
# =========================================================
def fixed_length_beam_paths(P, node_names, starts, L=4, beam_width=80, per_node_expand=40, no_repeat=True, top_k=30):
    """
    返回从 starts 出发，长度固定为 L 步（L+1 个节点）的高概率路径。
    目标：最大化 sum(log(P[u,v]))。
    """
    n = P.shape[0]
    eps = 1e-15

    # 每个节点预取 top 后继，减少搜索
    top_nb = []
    for i in range(n):
        row = P[i]
        m = min(per_node_expand, n)
        idx = np.argpartition(row, -m)[-m:]
        idx = idx[np.argsort(row[idx])[::-1]]
        top_nb.append(idx)

    cand = []
    for s in starts:
        beams = [([int(s)], 0.0)]
        for _ in range(L):
            new = []
            for path, sc in beams:
                u = path[-1]
                for v in top_nb[u]:
                    p = float(P[u, v])
                    if p <= 0:
                        continue
                    if no_repeat and v in path:
                        continue
                    new.append((path + [int(v)], sc + math.log(p + eps)))
            if not new:
                break
            new.sort(key=lambda x: x[1], reverse=True)
            beams = new[:beam_width]
        cand.extend(beams)

    cand.sort(key=lambda x: x[1], reverse=True)

    out = []
    seen = set()
    for path, sc in cand:
        if len(path) != L + 1:
            continue
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "steps": L,
                "log_prob": float(sc),
                "prob": float(math.exp(sc)),
                "path": " → ".join(node_names[i] for i in path),
            }
        )
        if len(out) >= top_k:
            break

    return pd.DataFrame(out)


# =========================================================
# 7) 全局 Top 边（按 A* 的流量，最“主干”的边）
# =========================================================
def top_global_edges(A_star, node_names, top_m=200, remove_self=True):
    n = A_star.shape[0]
    S = A_star.copy()
    if remove_self:
        np.fill_diagonal(S, 0.0)
    flat = S.ravel()
    idx = np.argsort(flat)[::-1][:top_m]

    rows = []
    for rank, k in enumerate(idx, start=1):
        i, j = divmod(int(k), n)
        val = float(flat[k])
        if val <= 0:
            continue
        rows.append({"rank": rank, "from": node_names[i], "to": node_names[j], "denoised_flow(A*)": val})
    return pd.DataFrame(rows)


# =========================================================
# 8) 主流程：输出 Excel
# =========================================================
def extract_global_trunk(
    excel_path,
    sheet_name=0,
    rank=8,
    top_out=20,
    p_min=1e-6,
    n_sources=12,
    n_sinks=12,
    top_k_paths=30,
    fixed_L=4,
    fixed_top_k=30,
    out_excel="global_trunk_paths.xlsx",
    manual_sources=None,   # 可传节点名列表，例如 ["提出|知识协作类", ...]
    manual_sinks=None,     # 可传节点名列表
):
    A, node_names = load_transition_matrix(excel_path, sheet_name=sheet_name)

    # 1) 去噪
    A_star, W, H, model = nmf_denoise(A, rank=rank)
    P = row_normalize(A_star)

    # 2) 稀疏化建图（在 P 上保留 top_out 出边）
    adj = build_sparse_adj(P, top_out=top_out, p_min=p_min)

    # 3) 源/汇选择（或手动覆盖）
    sources, sinks, out_strength, in_strength = pick_sources_sinks(A_star, n_sources=n_sources, n_sinks=n_sinks)
    if manual_sources:
        sources = names_to_indices(node_names, manual_sources)
    if manual_sinks:
        sinks = names_to_indices(node_names, manual_sinks)

    # 4) 全局主干路径（最大概率路径）
    paths_df = top_global_paths(
        adj, node_names, sources, sinks,
        top_k=top_k_paths,
        min_steps=1,          # 可改 2 或 3，避免只取一步/两步
        max_steps=None
    )

    # 5) 主干骨架边（路径聚合）
    backbone_df = backbone_edges_from_paths(paths_df, A_star, node_names)

    # 6) 固定长度主干路径（更像“链式主干”）
    #    起点用 sources（也可换成 out_strength 最大的若干节点）
    fixed_df = fixed_length_beam_paths(
        P, node_names, starts=sources[:max(3, min(10, len(sources)))],
        L=fixed_L, top_k=fixed_top_k
    )

    # 7) 全局 Top 边（按 A* 流量）
    edges_df = top_global_edges(A_star, node_names, top_m=200)

    # 8) 节点强度
    strength_df = pd.DataFrame({
        "node": node_names,
        "out_strength(A*)": out_strength,
        "in_strength(A*)": in_strength,
    }).sort_values("out_strength(A*)", ascending=False)

    # 写出
    with pd.ExcelWriter(out_excel) as writer:
        pd.DataFrame([{
            "rank(k)": rank,
            "nmf_iter": int(model.n_iter_),
            "top_out_per_node": top_out,
            "p_min": p_min,
            "n_sources": len(sources),
            "n_sinks": len(sinks),
        }]).to_excel(writer, index=False, sheet_name="meta")

        strength_df.to_excel(writer, index=False, sheet_name="node_strength")
        edges_df.to_excel(writer, index=False, sheet_name="top_edges_Astar")
        paths_df.to_excel(writer, index=False, sheet_name="top_paths_global")
        backbone_df.to_excel(writer, index=False, sheet_name="backbone_edges")
        fixed_df.to_excel(writer, index=False, sheet_name=f"fixed_len_L{fixed_L}")

    print(f"[OK] Saved: {out_excel}")
    return paths_df, backbone_df, fixed_df, edges_df


if __name__ == "__main__":
    excel_path = r"data\excel\主路径_加权_边spc转移矩阵_加权.csv"

    extract_global_trunk(
        excel_path=excel_path,
        sheet_name=0,

        # ---- 核心可调参数 ----
        rank=8,          # NMF 低秩：越大越细，越小越“主干”
        top_out=20,      # 每个节点保留多少条出边（控制主干稀疏性）
        p_min=1e-6,      # 丢弃太小的转移概率

        n_sources=12,    # 自动挑选源候选（按 out_strength）
        n_sinks=12,      # 自动挑选汇候选（按 in_strength）

        top_k_paths=5,  # 输出多少条全局主干路径（Dijkstra）
        fixed_L=20,       # 固定长度主干路径（更链式）
        fixed_top_k=5,

        out_excel="data\excel\非负矩阵分解.xlsx",

        # 如需手动指定源/汇，取消注释并填入 node_names 里的字符串：
        # manual_sources=["提出|知识协作类"],
        # manual_sinks=["发表|成果发布类"],
    )
