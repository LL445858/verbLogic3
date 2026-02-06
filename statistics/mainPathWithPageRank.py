# -*- coding: utf-8 -*-
"""
改进的主路径识别算法 - 融合 PageRank 值

核心改进：
- 原 SPC 值仅基于路径计数
- 新 SPC 值 = 原 SPC × 起点 PageRank × 转移概率
- 这样能更好地反映节点在网络中的重要性

输出文件包含以下列：
- from/to: 边的起点和终点
- prob: 转移概率
- pagerank_from: 起点的 PageRank 值
- spc_original: 原始 SPC 值
- spc_weighted: 加权后的 SPC 值（新 SPC）
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# -----------------------------
# 1) 读入矩阵并构图
# -----------------------------
def load_transition_matrix(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, index_col=0).fillna(0)

    # 尽量对齐：如果列名集合与行名集合一致，就按行顺序重排列
    if set(df.columns) == set(df.index):
        df = df.loc[df.index, df.index]
    else:
        # 若不一致，取交集（避免出现"行/列不匹配"导致的奇怪边）
        common = [c for c in df.columns if c in df.index]
        if len(common) == 0:
            raise ValueError("行名与列名没有交集：无法把它当作方阵转移矩阵来构图。")
        df = df.loc[common, common]

    return df


def load_pagerank(pagerank_path: str) -> Dict[str, float]:
    """
    读取 PageRank 值，返回节点到 PageRank 值的映射字典
    """
    df = pd.read_excel(pagerank_path)
    # 创建动词阶段到 PageRank 值的映射
    pagerank_dict = dict(zip(df['动词阶段'], df['PageRank']))
    return pagerank_dict


def build_digraph_from_matrix(df: pd.DataFrame, threshold: float = 0.0) -> nx.DiGraph:
    nodes = list(df.index)
    mat = df.to_numpy()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    rows, cols = np.where(mat > threshold)
    for r, c in zip(rows, cols):
        if r == c:
            continue
        u, v = nodes[r], nodes[c]
        G.add_edge(u, v, prob=float(mat[r, c]))

    return G


# -----------------------------
# 2) 防御性"去环"：把有环图变成 DAG（避免死循环）
#    用 Eades 启发式产生一个节点全序，然后只保留"顺序向前"的边
# -----------------------------
def eades_order(G: nx.DiGraph, weight_attr: str = "prob") -> List[str]:
    """
    Eades 启发式：近似最大权重无环子图的节点排序（反馈弧集近似）。
    输出从"左到右"的节点顺序 order；保留所有 pos[u] < pos[v] 的边即可保证无环。
    """
    H = G.copy()
    S1: List[str] = []
    S2: List[str] = []

    def in_w(n):
        return sum(H.edges[p, n].get(weight_attr, 1.0) for p in H.predecessors(n))

    def out_w(n):
        return sum(H.edges[n, q].get(weight_attr, 1.0) for q in H.successors(n))

    while H.number_of_nodes() > 0:
        changed = True
        while changed:
            changed = False

            # 先剥离 sinks（出度0）到 S2
            sinks = [n for n in list(H.nodes) if H.out_degree(n) == 0]
            if sinks:
                changed = True
                for n in sorted(sinks, key=str):
                    S2.append(n)
                    H.remove_node(n)

            # 再剥离 sources（入度0）到 S1
            sources = [n for n in list(H.nodes) if H.in_degree(n) == 0]
            if sources:
                changed = True
                for n in sorted(sources, key=str):
                    S1.append(n)
                    H.remove_node(n)

        if H.number_of_nodes() == 0:
            break

        # 若还剩"核心有环部分"，选 out_w - in_w 最大的点放入 S1
        best = None
        best_delta = None
        for n in H.nodes:
            delta = out_w(n) - in_w(n)
            if best is None or delta > best_delta or (delta == best_delta and str(n) < str(best)):
                best, best_delta = n, delta
        S1.append(best)
        H.remove_node(best)

    return S1 + list(reversed(S2))


def dagify_by_order(G: nx.DiGraph, weight_attr: str = "prob") -> Tuple[nx.DiGraph, List[Tuple[str, str, dict]]]:
    """
    若 G 有环：用 eades_order 得到顺序，只保留 pos[u] < pos[v] 的边，得到 DAG。
    返回：DAG，以及被移除的边列表（用于审计/输出）。
    """
    if nx.is_directed_acyclic_graph(G):
        return G.copy(), []

    order = eades_order(G, weight_attr=weight_attr)
    pos = {n: i for i, n in enumerate(order)}

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    removed = []
    for u, v, data in G.edges(data=True):
        if pos[u] < pos[v]:
            H.add_edge(u, v, **data)
        else:
            removed.append((u, v, dict(data)))

    if not nx.is_directed_acyclic_graph(H):
        raise RuntimeError("去环失败：得到的图仍然存在环。")

    return H, removed


# -----------------------------
# 3) 计算 SPC（DAG 上精确计算）
# -----------------------------
def compute_spc_on_dag(G: nx.DiGraph) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, int], Dict[Tuple[str, str], int]]:
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("compute_spc_on_dag 仅适用于 DAG。请先 dagify。")

    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]

    topo = list(nx.topological_sort(G))

    # forward[u] = 从任意 source 到 u 的路径数
    forward = {n: 0 for n in G.nodes}
    for s in sources:
        forward[s] = 1
    for u in topo:
        for v in G.successors(u):
            forward[v] += forward[u]

    # backward[u] = 从 u 到任意 sink 的路径数
    backward = {n: 0 for n in G.nodes}
    for t in sinks:
        backward[t] = 1
    for u in reversed(topo):
        if u in sinks:
            continue
        backward[u] = sum(backward[v] for v in G.successors(u))

    # SPC(u,v) = forward[u] * backward[v]
    spc = {}
    for u, v in G.edges:
        spc[(u, v)] = forward[u] * backward[v]

    return sources, sinks, forward, backward, spc


def compute_weighted_spc(
    G: nx.DiGraph,
    spc: Dict[Tuple[str, str], int],
    pagerank_dict: Dict[str, float]
) -> Dict[Tuple[str, str], float]:
    """
    计算加权 SPC 值：
    新 SPC = 原 SPC × 起点 PageRank × 转移概率

    参数:
        G: 有向图
        spc: 原始 SPC 字典 {(u,v): spc_value}
        pagerank_dict: 节点到 PageRank 值的映射

    返回:
        加权后的 SPC 字典 {(u,v): weighted_spc_value}
    """
    weighted_spc = {}

    for (u, v), original_spc in spc.items():
        # 获取转移概率
        prob = G.edges[u, v].get("prob", 0.0)

        # 获取起点的 PageRank 值（默认为0）
        pagerank_u = pagerank_dict.get(u, 0.0)

        # 计算加权 SPC
        weighted_value = original_spc * pagerank_u * prob

        weighted_spc[(u, v)] = weighted_value

    return weighted_spc


# -----------------------------
# 4) 提取主路径：max-sum 主路径 + key-route 主路径
# -----------------------------
def max_sum_main_path(G: nx.DiGraph, sources: List[str], sinks: List[str], weight: str = "spc_weighted") -> List[str]:
    """
    在 DAG 上求"边权累加和最大"的 source->sink 路径（动态规划）。
    """
    sinks_set = set(sinks)
    topo = list(nx.topological_sort(G))

    dp = {n: -math.inf for n in G.nodes}  # 从 n 到某个 sink 的最大路径和
    nxt = {}  # 最优后继

    for t in sinks:
        dp[t] = 0.0

    for u in reversed(topo):
        if u in sinks_set:
            continue
        best_val = -math.inf
        best_v = None
        for v in G.successors(u):
            w = G.edges[u, v].get(weight, 0)
            cand = w + dp.get(v, -math.inf)
            if cand > best_val:
                best_val = cand
                best_v = v
        if best_v is not None:
            dp[u] = best_val
            nxt[u] = best_v

    # 选择 dp 最大的 source 作为起点
    start = max(sources, key=lambda s: dp.get(s, -math.inf))
    path = [start]
    cur = start
    guard = 0
    while cur not in sinks_set and cur in nxt and guard < len(G.nodes) + 5:
        cur = nxt[cur]
        path.append(cur)
        guard += 1
    return path


def key_route_paths(G: nx.DiGraph, key_edges: List[Tuple[str, str]], sources: List[str], sinks: List[str], weight: str = "spc_weighted") -> List[List[str]]:
    """
    Key-route：选 top-k SPC 的关键边，对每条关键边向前/向后各自贪心延伸（每步选最大 SPC 边）。
    """
    sources_set, sinks_set = set(sources), set(sinks)
    paths = []
    max_steps = len(G.nodes) + 5

    for (u, v) in key_edges:
        if not G.has_edge(u, v):
            continue

        # backward extend from u
        back = [u]
        cur = u
        step = 0
        while cur not in sources_set and step < max_steps:
            in_edges = list(G.in_edges(cur, data=True))
            if not in_edges:
                break
            maxw = max(d.get(weight, 0) for _, _, d in in_edges)
            cand = [(p, d.get(weight, 0)) for p, _, d in in_edges if d.get(weight, 0) == maxw]
            p = sorted(cand, key=lambda x: str(x[0]))[0][0]
            if p in back:
                break
            back.append(p)
            cur = p
            step += 1
        back = list(reversed(back))  # source ... u

        # forward extend from v
        fwd = [v]
        cur = v
        step = 0
        while cur not in sinks_set and step < max_steps:
            out_edges = list(G.out_edges(cur, data=True))
            if not out_edges:
                break
            maxw = max(d.get(weight, 0) for _, _, d in out_edges)
            cand = [(n, d.get(weight, 0)) for _, n, d in out_edges if d.get(weight, 0) == maxw]
            n = sorted(cand, key=lambda x: str(x[0]))[0][0]
            if n in fwd:
                break
            fwd.append(n)
            cur = n
            step += 1

        path = back + [v] + fwd[1:]
        paths.append(path)

    # 去重
    uniq = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            uniq.append(p)
    return uniq


# -----------------------------
# 5) 主流程
# -----------------------------
def main(
    xlsx_path: str,
    pagerank_path: str,
    threshold: float,
    top_k: int,
    out_prefix: str
):
    # 读取转移矩阵和 PageRank 值
    df = load_transition_matrix(xlsx_path)
    pagerank_dict = load_pagerank(pagerank_path)

    G = build_digraph_from_matrix(df, threshold=threshold)

    print(f"[原始图] 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}, 是否DAG={nx.is_directed_acyclic_graph(G)}")

    dag, removed = dagify_by_order(G, weight_attr="prob")
    print(f"[去环后] 节点数={dag.number_of_nodes()}, 边数={dag.number_of_edges()}, 移除边数={len(removed)}, 是否DAG={nx.is_directed_acyclic_graph(dag)}")

    sources, sinks, forward, backward, spc = compute_spc_on_dag(dag)
    print(f"[sources/sinks] sources={len(sources)}, sinks={len(sinks)}")

    # 计算加权 SPC
    weighted_spc = compute_weighted_spc(dag, spc, pagerank_dict)
    print(f"[加权 SPC] 已融合 PageRank 和转移概率")

    # 写回边属性
    for (u, v), val in spc.items():
        dag.edges[u, v]["spc_original"] = int(val)
        dag.edges[u, v]["spc_weighted"] = float(weighted_spc[(u, v)])
        dag.edges[u, v]["pagerank_from"] = float(pagerank_dict.get(u, 0.0))

    # 导出边表（按加权 SPC 降序）
    edge_rows = []
    for u, v, data in dag.edges(data=True):
        edge_rows.append(
            {
                "from": u,
                "to": v,
                "prob": data.get("prob", np.nan),
                "pagerank_from": data.get("pagerank_from", 0.0),
                "spc_original": data.get("spc_original", 0),
                "spc_weighted": data.get("spc_weighted", 0.0),
                "forward_from_sources": forward[u],
                "backward_to_sinks": backward[v],
            }
        )
    edge_df = pd.DataFrame(edge_rows).sort_values("spc_weighted", ascending=False).reset_index(drop=True)
    edge_csv = f"{out_prefix}_边spc值_加权.csv"
    edge_df.to_csv(edge_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 加权边SPC表已保存：{edge_csv}")

    # 导出加权边SPC转移矩阵
    # 获取所有顶点并排序（保持一致的顺序）
    all_nodes = sorted(list(dag.nodes()))
    
    # 创建转移矩阵（行到列的SPC值）
    matrix_data = []
    for u in all_nodes:
        row = [u]  # 第一列是行顶点名称
        for v in all_nodes:
            if dag.has_edge(u, v):
                row.append(dag.edges[u, v].get("spc_weighted", 0.0))
            else:
                row.append(0.0)
        matrix_data.append(row)
    
    # 创建DataFrame，第一行为列顶点名称
    matrix_columns = [""] + all_nodes  # 第一列空，后面是顶点名称
    matrix_df = pd.DataFrame(matrix_data, columns=matrix_columns)
    
    matrix_csv = f"{out_prefix}_边spc转移矩阵_加权.csv"
    matrix_df.to_csv(matrix_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 加权边SPC转移矩阵已保存：{matrix_csv}")

    best_path = max_sum_main_path(dag, sources, sinks, weight="spc_weighted")
    key_edges = list(edge_df.head(top_k)[["from", "to"]].itertuples(index=False, name=None))
    kr_paths = key_route_paths(dag, key_edges, sources, sinks, weight="spc_weighted")

    print("\n[Max-sum 主路径（基于加权SPC）]")
    print(" -> ".join([f"{eval(i)[0]}_{eval(i)[1]}" for i in best_path]))

    print(f"\n[Key-route 主路径：top_k={top_k}，去重后{len(kr_paths)}条，展示前5条]")
    for i, p in enumerate(kr_paths[:5], 1):
        print(f"[{i}] " + " -> ".join([f"{eval(i)[0]}_{eval(i)[1]}" for i in p]))


if __name__ == "__main__":
    input_matrix = r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移概率_非马尔可夫.xlsx"
    pagerank_file = r"Y:\Project\Python\VerbLogic\data\excel\PageRank.xlsx"
    threshold = 0
    top_k = 15
    out_prefix = r"Y:\Project\Python\VerbLogic\data\excel\主路径_加权"
    main(input_matrix, pagerank_file, threshold, top_k, out_prefix)
