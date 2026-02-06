# -*- coding: utf-8 -*-
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
        # 若不一致，取交集（避免出现“行/列不匹配”导致的奇怪边）
        common = [c for c in df.columns if c in df.index]
        if len(common) == 0:
            raise ValueError("行名与列名没有交集：无法把它当作方阵转移矩阵来构图。")
        df = df.loc[common, common]

    return df


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
# 2) 防御性“去环”：把有环图变成 DAG（避免死循环）
#    用 Eades 启发式产生一个节点全序，然后只保留“顺序向前”的边
# -----------------------------
def eades_order(G: nx.DiGraph, weight_attr: str = "prob") -> List[str]:
    """
    Eades 启发式：近似最大权重无环子图的节点排序（反馈弧集近似）。
    输出从“左到右”的节点顺序 order；保留所有 pos[u] < pos[v] 的边即可保证无环。
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

        # 若还剩“核心有环部分”，选 out_w - in_w 最大的点放入 S1
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


# -----------------------------
# 4) 提取主路径：max-sum 主路径 + key-route 主路径
# -----------------------------
def max_sum_main_path(G: nx.DiGraph, sources: List[str], sinks: List[str], weight: str = "spc") -> List[str]:
    """
    在 DAG 上求“边权累加和最大”的 source->sink 路径（动态规划）。
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


def key_route_paths(G: nx.DiGraph, key_edges: List[Tuple[str, str]], sources: List[str], sinks: List[str], weight: str = "spc") -> List[List[str]]:
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
def main(xlsx_path: str, threshold: float, top_k: int, out_prefix: str):
    df = load_transition_matrix(xlsx_path)
    G = build_digraph_from_matrix(df, threshold=threshold)

    print(f"[原始图] 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}, 是否DAG={nx.is_directed_acyclic_graph(G)}")

    dag, removed = dagify_by_order(G, weight_attr="prob")
    print(f"[去环后] 节点数={dag.number_of_nodes()}, 边数={dag.number_of_edges()}, 移除边数={len(removed)}, 是否DAG={nx.is_directed_acyclic_graph(dag)}")

    sources, sinks, forward, backward, spc = compute_spc_on_dag(dag)
    print(f"[sources/sinks] sources={len(sources)}, sinks={len(sinks)}")

    # 写回 spc 属性
    for (u, v), val in spc.items():
        dag.edges[u, v]["spc"] = int(val)

    # 导出边表（按 SPC 降序）
    edge_rows = []
    for u, v, data in dag.edges(data=True):
        edge_rows.append(
            {
                "from": u,
                "to": v,
                "prob": data.get("prob", np.nan),
                "spc": data.get("spc", 0),
                "forward_from_sources": forward[u],
                "backward_to_sinks": backward[v],
            }
        )
    edge_df = pd.DataFrame(edge_rows).sort_values("spc", ascending=False).reset_index(drop=True)
    edge_csv = f"{out_prefix}_边spc值.csv"
    edge_df.to_csv(edge_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] 边SPC表已保存：{edge_csv}")

    best_path = max_sum_main_path(dag, sources, sinks, weight="spc")
    key_edges = list(edge_df.head(top_k)[["from", "to"]].itertuples(index=False, name=None))
    kr_paths = key_route_paths(dag, key_edges, sources, sinks, weight="spc")

    # txt_path = f"{out_prefix}_main_paths.txt"
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     f.write("=== Max-sum main path (1条) ===\n")
    #     f.write(" -> ".join(best_path) + "\n\n")

    #     f.write(f"=== Key-route main paths (top_k={top_k}, 去重后{len(kr_paths)}条) ===\n")
    #     for i, p in enumerate(kr_paths, 1):
    #         f.write(f"[{i}] " + " -> ".join(p) + "\n")

    # print(f"[输出] 主路径已保存：{txt_path}")

    print("\n[Max-sum 主路径]")
    print(" -> ".join([f"{eval(i)[0]}\_{eval(i)[1]}" for i in best_path]))

    print(f"\n[Key-route 主路径：top_k={top_k}，去重后{len(kr_paths)}条，展示前5条]")
    for i, p in enumerate(kr_paths[:5], 1):
        print(f"[{i}] " + " -> ".join([f"{eval(i)[0]}_{eval(i)[1]}" for i in p]))


if __name__ == "__main__":
    input = "data\\excel\\动词类别转移概率_非马尔可夫.xlsx"
    threshold = 0
    top_k = 15
    out_prefix = "data\\excel\主路径"
    main(input, threshold, top_k, out_prefix)
