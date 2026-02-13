import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def parse_c_data(content: str) -> dict:

    data_blocks = re.findall(
        r'"(data\d+)"\s*:\s*\{(.*?)\}\s*(?=,\s*"data\d+"\s*:|\s*\}\s*$)',
        content,
        re.DOTALL,
    )

    parsed_data = {}
    for key, block in data_blocks:
        items = re.findall(r'"(.*?)"\s*:\s*"(.*?)"', block)
        counter = defaultdict(int)
        word_labels = {}
        for word, label in items:
            counter[word] += 1
            suffix = f"_{counter[word]}" if counter[word] > 1 else ""
            word_labels[f"{word}{suffix}"] = label
        parsed_data[key] = word_labels

    return parsed_data


def find_matching_brace(s: str, start: int) -> int:

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i

    raise ValueError("No matching brace found")


def iter_data_blocks(content: str):

    i = 0
    pattern = re.compile(r'"(data\d+)"\s*:\s*\{', re.DOTALL)
    while True:
        m = pattern.search(content, i)
        if not m:
            break

        key = m.group(1)
        brace_start = m.end() - 1  # 指向 '{'
        brace_end = find_matching_brace(content, brace_start)
        block = content[brace_start + 1 : brace_end]

        yield key, block
        i = brace_end + 1


def parse_a_data(content: str) -> dict:

    parsed = {}
    for data_key, block in iter_data_blocks(content):
        j = 0
        counter = defaultdict(int)
        occ_attrs = {}

        while j < len(block):
            while j < len(block) and block[j] in " \t\r\n,":
                j += 1
            if j >= len(block):
                break
            if block[j] != '"':
                j += 1
                continue

            j += 1
            verb_chars = []
            escape = False
            while j < len(block):
                ch = block[j]
                if escape:
                    verb_chars.append(ch)
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    break
                else:
                    verb_chars.append(ch)
                j += 1
            verb = "".join(verb_chars)
            j += 1 

            while j < len(block) and block[j].isspace():
                j += 1
            if j < len(block) and block[j] == ":":
                j += 1
            while j < len(block) and block[j].isspace():
                j += 1

            if j >= len(block) or block[j] != "{":
                continue

            inner_start = j
            inner_end = find_matching_brace(block, inner_start)
            inner = block[inner_start + 1 : inner_end]

            attrs = dict(re.findall(r'"(.*?)"\s*:\s*"(.*?)"', inner))

            counter[verb] += 1
            suffix = f"_{counter[verb]}" if counter[verb] > 1 else ""
            occ_key = f"{verb}{suffix}"
            occ_attrs[occ_key] = attrs

            j = inner_end + 1

        parsed[data_key] = occ_attrs

    return parsed


def scale_linear(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:

    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def build_graph(
    v_dict: dict,
    c_map: dict,
    a_map: dict,
    data_indices: list[int],
    verb_stage_node_min_freq: int,
    attr_node_min_freq: int,
    stage_edge_min_freq: int,
    va_edge_min_freq: int,
):

    stage_freq = Counter()
    attr_freq = Counter()
    stage_edge_w = Counter()
    va_edge_w = Counter()

    for idx in data_indices:
        data_key = f"data{idx}"
        if data_key not in v_dict:
            continue

        verbs = v_dict[data_key]
        occ_counter = defaultdict(int)
        seq_stage_nodes = []

        for verb in verbs:
            occ_counter[verb] += 1
            occ_key = verb if occ_counter[verb] == 1 else f"{verb}_{occ_counter[verb]}"

            stage = (
                c_map.get(data_key, {}).get(occ_key)
                or c_map.get(data_key, {}).get(verb)
                or "未知阶段"
            )

            stage_node = f"{verb}_{stage}"
            seq_stage_nodes.append(stage_node)
            stage_freq[stage_node] += 1

            attrs = (
                a_map.get(data_key, {}).get(occ_key)
                or a_map.get(data_key, {}).get(verb, {})
                or {}
            )
            for attr_name in attrs.keys():
                attr_freq[attr_name] += 1
                va_edge_w[(stage_node, attr_name)] += 1

        for u, v in zip(seq_stage_nodes[:-1], seq_stage_nodes[1:]):
            stage_edge_w[(u, v)] += 1

    keep_stage = {n for n, f in stage_freq.items() if f >= verb_stage_node_min_freq}
    keep_attr = {n for n, f in attr_freq.items() if f >= attr_node_min_freq}

    stage_edge_w = Counter(
        {
            (u, v): w
            for (u, v), w in stage_edge_w.items()
            if w >= stage_edge_min_freq and u in keep_stage and v in keep_stage
        }
    )

    va_edge_w = Counter(
        {
            (u, a): w
            for (u, a), w in va_edge_w.items()
            if w >= va_edge_min_freq and u in keep_stage and a in keep_attr
        }
    )

    used_stage = set()
    for u, v in stage_edge_w.keys():
        used_stage.add(u)
        used_stage.add(v)
    for u, _ in va_edge_w.keys():
        used_stage.add(u)
    keep_stage &= used_stage

    used_attr = {a for _, a in va_edge_w.keys()}
    keep_attr &= used_attr

    G = nx.DiGraph()

    for n in keep_stage:
        G.add_node(n, kind="stage", freq=stage_freq[n])

    for n in keep_attr:
        G.add_node(n, kind="attr", freq=attr_freq[n])

    for (u, v), w in stage_edge_w.items():
        if u in keep_stage and v in keep_stage:
            G.add_edge(u, v, kind="trans", weight=w)

    for (u, a), w in va_edge_w.items():
        if u in keep_stage and a in keep_attr:
            G.add_edge(u, a, kind="attr", weight=w)

    return G


def repel_positions(
    pos: dict,
    nodes: list[str],
    min_dist: float,
    iterations: int = 200,
    step: float = 0.02,
    seed: int = 42,
) -> dict:

    if len(nodes) <= 1:
        return pos

    random.seed(seed)

    for n in nodes:
        if n not in pos:
            pos[n] = (random.uniform(-1, 1), random.uniform(-1, 1))

    nodes_list = list(nodes)
    for _ in range(iterations):
        moved = 0
        for i in range(len(nodes_list) - 1):
            ni = nodes_list[i]
            xi, yi = pos[ni]
            for j in range(i + 1, len(nodes_list)):
                nj = nodes_list[j]
                xj, yj = pos[nj]
                dx = xi - xj
                dy = yi - yj
                dist = math.hypot(dx, dy) + 1e-9
                if dist < min_dist:
                    push = (min_dist - dist) / dist * step
                    xi += dx * push
                    yi += dy * push
                    xj -= dx * push
                    yj -= dy * push
                    pos[nj] = (xj, yj)
                    moved += 1
            pos[ni] = (xi, yi)
        if moved == 0:
            break

    return pos


def compute_positions(
    G: nx.DiGraph,
    stage_area_radius: float = 8.0,
    attr_radius_factor: float = 2.4,
    stage_min_dist: float = 0.55,
    seed: int = 42,
) -> dict:

    stage_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "stage"]
    attr_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "attr"]

    G_stage = nx.DiGraph()
    for n in stage_nodes:
        G_stage.add_node(n)

    for u, v, d in G.edges(data=True):
        if d.get("kind") == "trans" and u in G_stage and v in G_stage:
            G_stage.add_edge(u, v, weight=d.get("weight", 1))

    if len(G_stage) > 1:
        k = max(0.9, 2.6 / math.sqrt(len(G_stage)))
        pos_stage = nx.spring_layout(G_stage, seed=seed, k=k, iterations=900, weight="weight")
    else:
        pos_stage = {n: (0.0, 0.0) for n in stage_nodes}

    pos_stage = repel_positions(
        pos_stage,
        nodes=stage_nodes,
        min_dist=stage_min_dist,
        iterations=280,
        step=0.06,
        seed=seed,
    )

    if pos_stage:
        r = max(math.hypot(x, y) for x, y in pos_stage.values()) or 1.0
        scale = stage_area_radius / r
        for n in pos_stage:
            x, y = pos_stage[n]
            pos_stage[n] = (x * scale, y * scale)

    pos = dict(pos_stage)

    if attr_nodes:
        attr_nodes_sorted = sorted(attr_nodes, key=lambda n: G.nodes[n].get("freq", 1), reverse=True)
        radius = stage_area_radius * attr_radius_factor

        for i, n in enumerate(attr_nodes_sorted):
            theta = 2 * math.pi * i / len(attr_nodes_sorted)
            jitter = 0.12 * math.sin(i)
            pos[n] = (
                (radius + jitter) * math.cos(theta),
                (radius + jitter) * math.sin(theta),
            )

    for n in G.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)

    return pos


def set_chinese_font():

    matplotlib.rcParams["font.sans-serif"] = [
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
    ]

    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_network(
    G: nx.DiGraph,
    pos: dict,
    stage_node_size_min: float,
    stage_node_size_max: float,
    attr_node_size_min: float,
    attr_node_size_max: float,
    stage_edge_width_min: float,
    stage_edge_width_max: float,
    va_edge_width_min: float,
    va_edge_width_max: float,
    stage_label_fontsize: int,
    attr_label_fontsize: int,
    label_min_freq_stage: int = 1,
    label_min_freq_attr: int = 1,
    output_path: str | None = None,
):

    set_chinese_font()

    stage_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "stage"]
    attr_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "attr"]

    stage_freqs = [G.nodes[n].get("freq", 1) for n in stage_nodes] or [1]
    attr_freqs = [G.nodes[n].get("freq", 1) for n in attr_nodes] or [1]

    stage_fmin, stage_fmax = min(stage_freqs), max(stage_freqs)
    attr_fmin, attr_fmax = min(attr_freqs), max(attr_freqs)

    stage_sizes = [
        scale_linear(
            G.nodes[n]["freq"], stage_fmin, stage_fmax, stage_node_size_min, stage_node_size_max
        )
        for n in stage_nodes
    ]

    attr_sizes = [
        scale_linear(
            G.nodes[n]["freq"], attr_fmin, attr_fmax, attr_node_size_min, attr_node_size_max
        )
        for n in attr_nodes
    ]

    trans_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("kind") == "trans"]
    attr_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("kind") == "attr"]

    trans_ws = [d.get("weight", 1) for _, _, d in trans_edges] or [1]
    attr_ws = [d.get("weight", 1) for _, _, d in attr_edges] or [1]

    tw_min, tw_max = min(trans_ws), max(trans_ws)
    aw_min, aw_max = min(attr_ws), max(attr_ws)

    n_total = len(stage_nodes) + len(attr_nodes)
    fig_size = min(8, 6.0 + n_total / 60)
    plt.figure(figsize=(fig_size, fig_size), dpi=100)

    for (u, v, d) in trans_edges:
        w = d.get("weight", 1)
        width = scale_linear(w, tw_min, tw_max, stage_edge_width_min, stage_edge_width_max)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color="black",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            connectionstyle="arc3,rad=0.10",
            min_source_margin=8,
            min_target_margin=8,
        )

    for i, (u, v, d) in enumerate(attr_edges):
        w = d.get("weight", 1)
        width = scale_linear(w, aw_min, aw_max, va_edge_width_min, va_edge_width_max)
        rad = 0.25 * math.sin(i) 
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color="gray",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=8,
            min_target_margin=8,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=stage_nodes,
        node_color="#4A90E2",
        node_shape="s",
        node_size=stage_sizes,
        linewidths=0.8,
        edgecolors="white",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=attr_nodes,
        node_color="#F5A623",
        node_shape="o",
        node_size=attr_sizes,
        linewidths=0.8,
        edgecolors="white",
    )

    for n, (x, y) in pos.items():
        kind = G.nodes[n].get("kind")
        freq = G.nodes[n].get("freq", 1)

        if kind == "stage" and freq < label_min_freq_stage:
            continue
        if kind == "attr" and freq < label_min_freq_attr:
            continue

        fontsize = stage_label_fontsize if kind == "stage" else attr_label_fontsize
        bbox = (
            dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.75)
            if kind == "stage"
            else dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75)
        )
        plt.text(x, y, n, fontsize=fontsize, ha="center", va="center", bbox=bbox)

    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    v_dict = json.loads(Path(v_path).read_text(encoding="utf-8"))
    c_map = parse_c_data(Path(c_path).read_text(encoding="utf-8"))
    a_map = parse_a_data(Path(a_path).read_text(encoding="utf-8"))

    G = build_graph(
        v_dict=v_dict,
        c_map=c_map,
        a_map=a_map,
        data_indices=data_indices,
        verb_stage_node_min_freq=verb_stage_node_min_freq,
        attr_node_min_freq=attr_node_min_freq,
        stage_edge_min_freq=stage_edge_min_freq,
        va_edge_min_freq=va_edge_min_freq,
    )

    pos = compute_positions(
        G,
        stage_area_radius=stage_area_radius,
        attr_radius_factor=attr_radius_factor,
        stage_min_dist=stage_min_dist,
        seed=42,
    )

    draw_network(
        G,
        pos,
        stage_node_size_min=stage_node_size_min,
        stage_node_size_max=stage_node_size_max,
        attr_node_size_min=attr_node_size_min,
        attr_node_size_max=attr_node_size_max,
        stage_edge_width_min=stage_edge_width_min,
        stage_edge_width_max=stage_edge_width_max,
        va_edge_width_min=va_edge_width_min,
        va_edge_width_max=va_edge_width_max,
        stage_label_fontsize=stage_label_fontsize,
        attr_label_fontsize=attr_label_fontsize,
        label_min_freq_stage=label_min_freq_stage,
        label_min_freq_attr=label_min_freq_attr,
        output_path=output_path,
    )


if __name__ == "__main__":

    v_path = r"data\result\gold\v.txt"  # 动词列表
    c_path = r"data\result\gold\c.txt"  # 动词阶段列表
    a_path = r"data\result\gold\a.txt"  # 动词属性列表
    # v_dict = r""

    # 动词_阶段 节点大小范围
    stage_node_size_max = 800
    stage_node_size_min = 200

    # 属性 节点大小范围
    attr_node_size_max = 500
    attr_node_size_min = 100

    # 动词阶段节点之间连线宽度范围
    stage_edge_width_max = 3.0
    stage_edge_width_min = 1

    # 动词阶段节点到属性节点连线宽度范围
    va_edge_width_max = 1.0
    va_edge_width_min = 0.5

    # 节点频次阈值（小于阈值的节点将被移除）
    verb_stage_node_min_freq = 3  # 动词_阶段 节点频次阈值
    attr_node_min_freq = 2  # 属性 节点频次阈值

    # 边频次阈值（小于阈值的边将被移除）
    stage_edge_min_freq = 1  # 动词_阶段 -> 动词_阶段 连线阈值
    va_edge_min_freq = 1  # 动词_阶段 -> 属性 连线阈值

    # 节点标签显示阈值
    label_min_freq_stage = 1  # 仅显示频次 >= 此阈值的动词_阶段标签
    label_min_freq_attr = 1  # 仅显示频次 >= 此阈值的属性标签

    # 节点标签字体大小
    stage_label_fontsize = 6  # “动词_阶段”节点标签字体大小
    attr_label_fontsize = 6  # “属性”节点标签字体大小

    # 布局参数
    stage_area_radius = 300  # 阶段节点占据中间“大区域”的半径（越大越分散）
    attr_radius_factor = 1.3  # 属性节点放到阶段区域之外的倍数（>1 即可，越大离中心越远） 
    stage_min_dist = 1.2  # 阶段节点最小间距（用于排斥后处理，越大越不容易重叠，但图会更松）

    # 参与统计的文本序号列表
    data_indices = list(range(16, 39))

    # 输出图片路径（None 表示不保存）
    output_path = None 

    main()
