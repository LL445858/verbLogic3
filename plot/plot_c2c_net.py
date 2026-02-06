#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14
# @Author  : YinLuLu
# @File    : plot_c2c_net.py
# @Software: PyCharm

import matplotlib
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
matplotlib.use('TkAgg')


def create_graph():
    df = pd.read_excel(r"Y:\Project\Python\VerbLogic\data\excel\阶段转移概率.xlsx", sheet_name='Sheet1',
                       index_col=0)
    G = nx.DiGraph()
    words = {'知识规划类': 72, '知识整合类': 37, '资源获取类': 48, '知识协作类': 83, "知识重构类": 318,
             '成果发布类': 56, '成果影响类': 73}
    for word, weight in words.items():
        G.add_node(word, weight=weight)

    edge_set = set()
    for i, source in enumerate(words):
        for j, target in enumerate(words):
            p = df.iloc[i, j]
            if p > 0:
                if (source, target) not in edge_set:
                    edge_set.add((target, source))
                    G.add_edge(source, target, probability=float(p), two=1)
                else:
                    G.add_edge(source, target, probability=float(p), two=-1)
    print(G)
    return G


def visualize_paper_ready_network():
    G = create_graph()
    output_filename = r"Y:\Project\Python\VerbLogic\data\figure\类别转移.svg"
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.figure(figsize=(12, 10))

    node_weights = {node: G.nodes[node]['weight'] for node in G.nodes()}
    pos = nx.circular_layout(G)
    node_sizes = [1000 + (G.nodes[node]['weight'] / max(node_weights.values())) * 3000
                  for node in G.nodes()]
    norm_weights = [(node_weights[node] / max(node_weights.values())) for node in G.nodes()]

    reds = [(1.0, 0.9, 0.9), (1.0, 0.7, 0.7), (1.0, 0.5, 0.5), (0.9, 0.3, 0.3), (0.7, 0.0, 0.0)]
    red_cmap = LinearSegmentedColormap.from_list("custom_reds", reds)
    greens = [(0.4, 0.8, 0.4), (0.2, 0.6, 0.2), (0.0, 0.4, 0.0)]
    green_cmap = LinearSegmentedColormap.from_list("custom_greens", greens)

    pos = nx.spring_layout(G, seed=42)
    pos["知识重构类"] = (0.0, 0.0)
    pos["知识规划类"] = (0.7, -0.7)

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=norm_weights,
        cmap=red_cmap,
        alpha=0.9,
        # edgecolors='darkred',
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=14,
        font_weight='bold',
        font_color='black',
        font_family=prop.get_name()
    )

    edges = nx.draw_networkx_edges(
        G, pos,
        width=[3 + (G[u][v]['probability'] * 8) for u, v in G.edges()],
        edge_color=[G[u][v]['probability'] for u, v in G.edges()],
        edge_cmap=green_cmap,
        alpha=0.9,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle=[f'arc3,rad={0.3 * G[u][v]['two']}' for u, v in G.edges()],
        min_source_margin=25,
        min_target_margin=25
    )

    edge_labels = {(u, v): f"{G[u][v]['probability']:.1%}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=12,
        font_weight='bold',
        font_color='darkgreen',
        bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.9),
        font_family=prop.get_name(),
        label_pos=0.6,
        rotate=False
    )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=2000, bbox_inches='tight')
    print(f"已保存为: {output_filename}")
    return plt


if __name__ == '__main__':
    visualize_paper_ready_network()
