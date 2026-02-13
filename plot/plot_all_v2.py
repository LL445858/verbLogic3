from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def parse_c_data(content):

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


def find_matching_brace(s, start):

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


def iter_data_blocks(content):

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


def parse_a_data(content):

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


def parse_verb_lulu(path):

    text = Path(path).read_text(encoding="utf-8")
    mapping: dict[str, str] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        part = line.split(":", 1)[1] if ":" in line else line
        words = re.findall(r"([^\(\)、，\s]+)\(\d+次\)", part)
        if not words:
            continue

        canon = words[0]
        for w in words:
            mapping[w] = canon

    return mapping


def scale_linear(value, vmin, vmax, out_min, out_max):

    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def shorten_value(text, max_len):

    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    return f"{s[:3]}...{s[-3:]}"


def split_stage_label(node_name):

    m = re.match(r"^(.*)\(([^()]*)\)$", node_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def build_graph(v_dict, c_map, a_map, verb_syn_map, data_indices, verb_stage_node_min_freq, attr_node_min_freq, stage_edge_min_freq, va_edge_min_freq):

    stage_freq = Counter()
    attr_freq = Counter()
    stage_edge_w = Counter()

    pair_count = Counter()
    attr_edges_occ = []

    for idx in data_indices:
        data_key = f"data{idx}"
        if data_key not in v_dict:
            continue

        verbs = v_dict[data_key]
        occ_counter = defaultdict(int)
        seq_stage_nodes: list[str] = []

        for verb_raw in verbs:
            occ_counter[verb_raw] += 1
            occ_key = verb_raw if occ_counter[verb_raw] == 1 else f"{verb_raw}_{occ_counter[verb_raw]}"

            stage_full = (
                c_map.get(data_key, {}).get(occ_key)
                or c_map.get(data_key, {}).get(verb_raw)
                or "未知阶段"
            ).strip()
            abbr = STAGE_ABBR.get(stage_full, "UNK")

            canon_verb = verb_syn_map.get(verb_raw, verb_raw)
            stage_node = f"{canon_verb}({abbr})"

            seq_stage_nodes.append(stage_node)
            stage_freq[stage_node] += 1

            attrs = (
                a_map.get(data_key, {}).get(occ_key)
                or a_map.get(data_key, {}).get(verb_raw, {})
                or {}
            )

            for attr_name, attr_val in attrs.items():
                attr_node = f"R_{attr_name}"
                attr_freq[attr_node] += 1

                pair_count[(stage_node, attr_node)] += 1

                # 修复点 1：保证每条 (stage->attr) 实例边都有可显示标签
                # a.txt 中如果 attr_val 为空/空白，旧逻辑会导致 label==''，从而被绘图时跳过
                val_str = "" if attr_val is None else str(attr_val).strip()
                if val_str == "":
                    val_str = "空"

                attr_edges_occ.append((stage_node, attr_node, val_str))

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

    keep_pairs = {
        (u, a)
        for (u, a), cnt in pair_count.items()
        if cnt >= va_edge_min_freq and u in keep_stage and a in keep_attr
    }
    attr_edges_occ = [(u, a, val) for (u, a, val) in attr_edges_occ if (u, a) in keep_pairs]

    used_stage = set()
    used_attr = set()

    for (u, v) in stage_edge_w.keys():
        used_stage.add(u)
        used_stage.add(v)

    for (u, a, _) in attr_edges_occ:
        used_stage.add(u)
        used_attr.add(a)

    keep_stage &= used_stage
    keep_attr &= used_attr

    G = nx.MultiDiGraph()

    for n in keep_stage:
        G.add_node(n, kind="stage", freq=stage_freq[n])

    for n in keep_attr:
        G.add_node(n, kind="attr", freq=attr_freq[n])

    for (u, v), w in stage_edge_w.items():
        if u in keep_stage and v in keep_stage:
            G.add_edge(u, v, kind="trans", weight=w)

    for u, a, val in attr_edges_occ:
        if u in keep_stage and a in keep_attr:
            G.add_edge(
                u,
                a,
                kind="attr",
                value=val,
                label=shorten_value(val, max_len=6) or "空",
            )

    return G


def repel_positions(pos, nodes, min_dist, iterations, step, seed):

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


def compute_positions(G, stage_area_radius, attr_radius_factor, stage_min_dist, attr_min_dist, seed):

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
        iterations=300,
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
        n_attr = len(attr_nodes_sorted)
        base_radius = stage_area_radius * attr_radius_factor

        if n_attr > 1:
            min_circumference = n_attr * attr_min_dist
            min_radius_for_spacing = min_circumference / (2 * math.pi)
            radius = max(base_radius, min_radius_for_spacing)
        else:
            radius = base_radius

        for i, n in enumerate(attr_nodes_sorted):
            theta = 2 * math.pi * i / n_attr
            jitter = 0.12 * math.sin(i)
            pos[n] = ((radius + jitter) * math.cos(theta), (radius + jitter) * math.sin(theta))

    for n in G.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)

    return pos


def compute_positions_irregular(G, stage_area_radius, attr_radius_factor, stage_min_dist, attr_min_dist, seed):
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
        iterations=300,
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
        random.seed(seed)
        attr_nodes_shuffled = list(attr_nodes)
        random.shuffle(attr_nodes_shuffled)

        n_attr = len(attr_nodes_shuffled)
        
        # 计算每个节点的视觉半径（与绘图时的大小对应）
        attr_freqs = [G.nodes[n].get("freq", 1) for n in attr_nodes_shuffled]
        if attr_freqs:
            attr_fmin, attr_fmax = min(attr_freqs), max(attr_freqs)
            attr_radii = []
            
            for freq in attr_freqs:
                # 根据绘图时的大小映射到视觉半径（基于绘图函数中的逻辑）
                visual_size = scale_linear(freq, attr_fmin, attr_fmax, 100, 600)  # 使用相对大小
                # 将视觉大小转换为近似的半径（这里做一个合理的估算）
                radius = math.sqrt(visual_size / math.pi) * 0.5
                attr_radii.append(radius)
        else:
            attr_radii = [10.0] * n_attr  # 默认半径
        
        # 考虑节点大小的最小间距（节点中心之间的距离应该至少是两个半径之和加上缓冲）
        min_area_needed = sum(math.pi * (r + attr_min_dist/2)**2 for r in attr_radii) * 1.5
        base_inner_radius = stage_area_radius * 1.15
        base_outer_radius = stage_area_radius * attr_radius_factor * 1.25

        base_ring_area = math.pi * (base_outer_radius ** 2 - base_inner_radius ** 2)
        if base_ring_area < min_area_needed:
            required_outer = math.sqrt(base_inner_radius ** 2 + min_area_needed / math.pi)
            outer_radius = required_outer * 1.2
        else:
            outer_radius = base_outer_radius

        inner_radius = base_inner_radius

        attr_positions = []
        for i, n in enumerate(attr_nodes_shuffled):
            angle_offset = (i / n_attr) * 2 * math.pi if n_attr > 0 else 0
            r = inner_radius + (outer_radius - inner_radius) * (0.2 + 0.6 * random.random())
            theta = angle_offset + random.random() * (2 * math.pi / n_attr) * 1.5
            attr_positions.append((n, attr_radii[i], r * math.cos(theta), r * math.sin(theta)))  # 包含半径信息

        max_iterations = 1000
        initial_step = max(attr_min_dist * 1.0, 150.0)
        step_size = initial_step
        cooling_rate = 0.995
        min_step = 2.0

        for iteration in range(max_iterations):
            max_overlap = 0.0
            any_movement = False

            indices = list(range(len(attr_positions)))
            random.shuffle(indices)

            for i in indices:
                ni, ri, xi, yi = attr_positions[i]
                fx, fy = 0.0, 0.0

                for j in range(len(attr_positions)):
                    if i == j:
                        continue
                    nj, rj, xj, yj = attr_positions[j]
                    dx = xi - xj
                    dy = yi - yj
                    dist = math.hypot(dx, dy) + 1e-9
                    
                    # 节点圆心间距必须大于半径之和加上最小间距
                    # 节点不重叠的条件: dist > (ri + rj)，即 dist - (ri + rj) > 0
                    # 加上最小间距要求: dist > (ri + rj + attr_min_dist)
                    required_dist = ri + rj + attr_min_dist
                    if dist < required_dist:
                        overlap = required_dist - dist  # 这是实际缺少的距离
                        max_overlap = max(max_overlap, overlap)
                        force = overlap / dist * 2.5  # 增加排斥力系数
                        fx += dx * force
                        fy += dy * force

                dist_center = math.hypot(xi, yi)
                if dist_center < inner_radius * 0.95:
                    force = (inner_radius * 0.95 - dist_center) / (dist_center + 1e-9)
                    fx -= xi * force * 0.3
                    fy -= yi * force * 0.3
                elif dist_center > outer_radius * 1.05:
                    force = (dist_center - outer_radius * 1.05) / (dist_center + 1e-9)
                    fx -= xi * force * 0.2
                    fy -= yi * force * 0.2

                new_x = xi + fx * step_size
                new_y = yi + fy * step_size

                new_dist = math.hypot(new_x, new_y)
                if new_dist > 0:
                    attr_positions[i] = (ni, ri, new_x, new_y)
                    if abs(fx) > 0.5 or abs(fy) > 0.5:
                        any_movement = True

            step_size = max(step_size * cooling_rate, min_step)

            if max_overlap < 0.1 and iteration > 100:
                break

            if not any_movement and iteration > 800:
                step_size = initial_step * 0.5  # 增加扰动幅度

        # 最终检查并修正任何可能的重叠
        for i in range(len(attr_positions)):
            for j in range(i + 1, len(attr_positions)):
                ni, ri, xi, yi = attr_positions[i]
                nj, rj, xj, yj = attr_positions[j]
                
                dx = xi - xj
                dy = yi - yj
                dist = math.hypot(dx, dy)
                
                required_dist = ri + rj + attr_min_dist
                if dist < required_dist:
                    # 强制分离重叠的节点
                    overlap = required_dist - dist
                    separation_force = overlap * 0.5  # 分离力
                    unit_dx = dx / dist if dist > 0 else 1.0
                    unit_dy = dy / dist if dist > 0 else 0.0
                    
                    # 移动两个节点远离彼此
                    move_x = unit_dx * separation_force / 2
                    move_y = unit_dy * separation_force / 2
                    
                    attr_positions[i] = (ni, ri, xi + move_x, yi + move_y)
                    attr_positions[j] = (nj, rj, xj - move_x, yj - move_y)

        for n, r, x, y in attr_positions:
            pos[n] = (x, y)

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


def _attr_edge_rads(m, base, step, max_abs=None):

    if m <= 0:
        return []
    if m == 1:
        return [base]

    if max_abs is None:
        max_abs = min(1.8, base + step * ((m - 1) // 2))
        max_abs = max(max_abs, base)

    rads = []
    k = 1
    while len(rads) < m:
        rad = min(max_abs, base + (k - 1) * step)
        rads.append(rad)
        if len(rads) >= m:
            break
        rads.append(-rad)
        k += 1

    return rads[:m]


def _edge_label_pos(pos_u, pos_v, rad, idx, total):

    x1, y1 = pos_u
    x2, y2 = pos_v

    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy) + 1e-9

    if total <= 1:
        t = 0.5
    else:
        center = (total - 1) / 2
        t = 0.5 + (idx - center) * 0.06
        t = max(0.25, min(0.75, t))

    mx, my = x1 + dx * t, y1 + dy * t

    px, py = -dy / dist, dx / dist
    shift = rad * dist * 0.40

    return mx + px * shift, my + py * shift


def _set_zorder(artists, z):

    if artists is None:
        return
    if isinstance(artists, (list, tuple)):
        for a in artists:
            try:
                a.set_zorder(z)
            except Exception:
                pass
    else:
        try:
            artists.set_zorder(z)
        except Exception:
            pass


def draw_network(G, pos, unify_node_size, node_size_min, node_size_max, stage_node_size_min, stage_node_size_max, attr_node_size_min, attr_node_size_max, stage_edge_width_min, stage_edge_width_max, va_edge_width, stage_edge_color, va_edge_color, stage_label_font_color, attr_label_font_color, va_edge_label_font_color, va_edge_label_fontsize, stage_label_fontsize, attr_label_fontsize, label_min_freq_stage, label_min_freq_attr, output_path):

    set_chinese_font()

    stage_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "stage"]
    attr_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "attr"]

    stage_freqs = [G.nodes[n].get("freq", 1) for n in stage_nodes] or [1]
    attr_freqs = [G.nodes[n].get("freq", 1) for n in attr_nodes] or [1]

    if unify_node_size:
        all_freqs = stage_freqs + attr_freqs
        fmin, fmax = min(all_freqs), max(all_freqs)

        stage_sizes = [
            scale_linear(G.nodes[n]["freq"], fmin, fmax, node_size_min, node_size_max)
            for n in stage_nodes
        ]
        attr_sizes = [
            scale_linear(G.nodes[n]["freq"], fmin, fmax, node_size_min, node_size_max)
            for n in attr_nodes
        ]
    else:
        stage_fmin, stage_fmax = min(stage_freqs), max(stage_freqs)
        attr_fmin, attr_fmax = min(attr_freqs), max(attr_freqs)

        stage_sizes = [
            scale_linear(G.nodes[n]["freq"], stage_fmin, stage_fmax,
                        stage_node_size_min, stage_node_size_max)
            for n in stage_nodes
        ]
        attr_sizes = [
            scale_linear(G.nodes[n]["freq"], attr_fmin, attr_fmax,
                        attr_node_size_min, attr_node_size_max)
            for n in attr_nodes
        ]

    trans_edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == "trans"]
    attr_edges = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == "attr"]

    trans_ws = [d.get("weight", 1) for _, _, _, d in trans_edges] or [1]
    tw_min, tw_max = min(trans_ws), max(trans_ws)

    n_total = len(stage_nodes) + len(attr_nodes)
    fig_size = min(8, int(6.0 + n_total / 60))
    plt.figure(figsize=(fig_size, fig_size), dpi=120)
    ax = plt.gca()

    # --------------------- 属性连线：较低图层 ---------------------
    edges_by_pair: dict[tuple[str, str], list[tuple[str, str, int, dict]]] = defaultdict(list)
    for u, v, k, d in attr_edges:
        edges_by_pair[(u, v)].append((u, v, k, d))

    for (u, v), edges_list in edges_by_pair.items():
        rads = _attr_edge_rads(len(edges_list), base=0.22, step=0.12)
        for idx, (edge, rad) in enumerate(zip(edges_list, rads)):
            uu, vv, kk, dd = edge
            arts = nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(uu, vv, kk)],
                width=va_edge_width,
                edge_color=va_edge_color,
                arrows=True,
                arrowstyle="-",
                arrowsize=0,
                connectionstyle=f"arc3,rad={rad}",
                min_source_margin=8,
                min_target_margin=8,
                ax=ax,
            )
            _set_zorder(arts, 1)

            # --------------------- 边标签：中间图层（始终绘制） ---------------------
            label = dd.get("label")
            if label is None or str(label).strip() == "":
                label = shorten_value(dd.get("value", "空"), max_len=6) or "空"

            lx, ly = _edge_label_pos(pos[uu], pos[vv], rad, idx=idx, total=len(edges_list))
            ax.text(
                lx,
                ly,
                label,
                fontsize=va_edge_label_fontsize,
                color=va_edge_label_font_color,
                ha="center",
                va="center",
                zorder=2,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )

    # --------------------- 阶段间连线：较高图层 ---------------------
    for (u, v, k, d) in trans_edges:
        w = d.get("weight", 1)
        width = scale_linear(w, tw_min, tw_max, stage_edge_width_min, stage_edge_width_max)
        arts = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v, k)],
            width=width,
            edge_color=stage_edge_color,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=6,
            connectionstyle="arc3,rad=0.10",
            min_source_margin=6,
            min_target_margin=6,
            ax=ax,
        )
        _set_zorder(arts, 3)

    arts_stage = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=stage_nodes,
        node_color="#4A90E2",
        node_shape="s",
        node_size=stage_sizes,
        linewidths=0.0,
        edgecolors="none",
        ax=ax,
    )
    _set_zorder(arts_stage, 4)

    arts_attr = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=attr_nodes,
        node_color="#F5A623",
        node_shape="o",
        node_size=attr_sizes,
        linewidths=0.0,
        edgecolors="none",
        ax=ax,
    )
    _set_zorder(arts_attr, 4)

    for n, (x, y) in pos.items():
        kind = G.nodes[n].get("kind")
        freq = G.nodes[n].get("freq", 1)

        if kind == "stage" and freq < label_min_freq_stage:
            continue
        if kind == "attr" and freq < label_min_freq_attr:
            continue

        if kind == "stage":
            parts = split_stage_label(n)
            label = f"{parts[0]}\n({parts[1]})" if parts else n
            fontsize = stage_label_fontsize
            color = stage_label_font_color
        else:
            label = n
            fontsize = attr_label_fontsize
            color = attr_label_font_color

        ax.text(
            x,
            y,
            label,
            fontsize=fontsize,
            color=color,
            ha="center",
            va="center",
            zorder=5,
            clip_on=False,
        )

    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=1200, bbox_inches="tight")

    plt.show()


def main():

    v_dict = json.loads(Path(v_path).read_text(encoding="utf-8"))
    c_map = parse_c_data(Path(c_path).read_text(encoding="utf-8"))
    a_map = parse_a_data(Path(a_path).read_text(encoding="utf-8"))
    verb_syn_map = parse_verb_lulu(verb_lulu_path)

    G = build_graph(
        v_dict=v_dict,
        c_map=c_map,
        a_map=a_map,
        verb_syn_map=verb_syn_map,
        data_indices=data_indices,
        verb_stage_node_min_freq=verb_stage_node_min_freq,
        attr_node_min_freq=attr_node_min_freq,
        stage_edge_min_freq=stage_edge_min_freq,
        va_edge_min_freq=va_edge_min_freq,
    )

    if use_circular_layout:
        pos = compute_positions(
            G,
            stage_area_radius=stage_area_radius,
            attr_radius_factor=attr_radius_factor,
            stage_min_dist=stage_min_dist,
            attr_min_dist=attr_min_dist,
            seed=seed,
        )
    else:
        pos = compute_positions_irregular(
            G,
            stage_area_radius=stage_area_radius,
            attr_radius_factor=attr_radius_factor,
            stage_min_dist=stage_min_dist,
            attr_min_dist=attr_min_dist,
            seed=seed,
        )

    draw_network(
        G,
        pos,
        unify_node_size=unify_node_size,
        node_size_min=node_size_min,
        node_size_max=node_size_max,    
        stage_node_size_min=stage_node_size_min,
        stage_node_size_max=stage_node_size_max,
        attr_node_size_min=attr_node_size_min,
        attr_node_size_max=attr_node_size_max,
        stage_edge_width_min=stage_edge_width_min,
        stage_edge_width_max=stage_edge_width_max,
        va_edge_width=va_edge_width,
        stage_edge_color=stage_edge_color,
        va_edge_color=va_edge_color,
        stage_label_font_color=stage_label_font_color,
        attr_label_font_color=attr_label_font_color,
        va_edge_label_font_color=va_edge_label_font_color,
        va_edge_label_fontsize=va_edge_label_fontsize,
        stage_label_fontsize=stage_label_fontsize,
        attr_label_fontsize=attr_label_fontsize,
        label_min_freq_stage=label_min_freq_stage,
        label_min_freq_attr=label_min_freq_attr,
        output_path=output_path,
    )


if __name__ == "__main__":

    STAGE_ABBR = {
        "知识规划类": "KP",
        "知识整合类": "KI",
        "资源获取类": "RA",
        "知识协作类": "KC",
        "知识重构类": "KR",
        "成果发布类": "RR",
        "成果影响类": "RI",
    }

    data_indices = list(range(16, 39)) # 参与统计的文本序号列表
    v_path = r"Y:\Project\Python\VerbLogic\data\result_old\gold\v.txt" # 动词列表
    c_path = r"Y:\Project\Python\VerbLogic\data\result_old\gold\c.txt"  # 动词阶段列表
    a_path = r"Y:\Project\Python\VerbLogic\data\result_old\gold\a.txt"  # 动词属性列表
    verb_lulu_path = r"Y:\Project\Python\VerbLogic\data\analysis\verb_lulu.txt"# 同义词典文件路径
    output_path = r"Y:\Project\Python\VerbLogic\data\figure\pic_all.svg"  # 输出图片路径,不输出可设为None
    
    # 节点大小统一归一化开关
    unify_node_size = True   # True=动词+属性一起归一化；False=分开归一化
    node_size_max = 5000     # 统一模式下的最大节点尺寸
    node_size_min = 2     # 统一模式下的最小节点尺寸

    # 动词节点配置 
    stage_node_size_max = 600
    stage_node_size_min = 100
    stage_label_fontsize = 3
    stage_label_font_color = "black" 
    
    # 属性节点配置 
    attr_node_size_max = 600 
    attr_node_size_min = 100 
    attr_label_fontsize = 3
    attr_label_font_color = "black" 
    
    # 阶段->阶段连线配置 
    stage_edge_width_max = 0.8
    stage_edge_width_min = 0.5 
    stage_edge_color = "#333333" 
    
    # 阶段->属性连线配置 
    va_edge_width = 0.4
    va_edge_color = "#AAAAAA" 
    va_edge_label_font_color = "#353535"
    va_edge_label_fontsize = 2.5

    # 标签显示阈值 
    label_min_freq_stage = 1 
    label_min_freq_attr = 1 
    
    # 节点/边显示阈值 
    verb_stage_node_min_freq = 2 # 阶段节点频次阈值
    attr_node_min_freq = 2 # 属性节点频次阈值
    stage_edge_min_freq = 1 # 阶段->阶段边阈值 
    va_edge_min_freq = 1 # 阶段->属性聚合次数阈值 
    
    # 布局参数
    seed = 21  # 随机数种子
    use_circular_layout = False  # True=圆形布局；False=非圆形不规则布局
    stage_area_radius = 500 # 中间阶段区域半径：越大越分散 
    attr_radius_factor = 1.35 # 属性外圈半径倍数：越大离中心越远
    stage_min_dist = 1.50 # 阶段节点最小间距：越大越不重叠 
    attr_min_dist = 20 # 属性节点最小间距：越大越不重叠
    
    main()