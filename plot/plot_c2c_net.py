import matplotlib
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

matplotlib.use('TkAgg')


# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def create_graph():
    df = pd.read_excel(r"Y:\Project\PythonProject\VerbLogic\data\excel\阶段转移概率.xlsx", sheet_name='Sheet1',
                       index_col=0)
    G = nx.DiGraph()
    words = {'知识规划类': 72, '知识整合类': 37, '资源获取类': 48, '知识协作类': 83, "知识重构类": 318,
             '成果发布类': 56, '成果影响类': 73}
    for word, weight in words.items():
        G.add_node(word, weight=weight)  # 添加节点和'weight'属性

    # 添加带权重的边
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
    output_filename = r"Y:\Project\PythonProject\VerbLogic\data\figure\类别转移.svg"
    title = "传记中的动词转移网络"

    # 设置微软雅黑字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑字体路径
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

    # 为了确保所有中文字符正确显示，使用fontproperties参数
    chinese_font = fm.FontProperties(fname=font_path)

    # 创建图形
    plt.figure(figsize=(12, 10))  # 论文适合的大小

    # 只保留最重要的节点和边
    node_weights = {node: G.nodes[node]['weight'] for node in G.nodes()}
    # top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:15]  # 只取15个最重要节点
    #
    # top_node_names = [node for node, _ in top_nodes]
    # subgraph = G.subgraph(top_node_names)

    # 获取高概率的边
    # H = nx.DiGraph()
    # for node in subgraph.nodes():
    #     H.add_node(node, **G.nodes[node])
    #
    # for u, v, data in subgraph.edges(data=True):
    #     if data['probability'] >= 0.010:  # 只保留概率>10%的边
    #         H.add_edge(u, v, **data)

    # 使用更有效的布局
    pos = nx.circular_layout(G)  # 保持环形布局

    # 简化配色方案
    node_sizes = [1000 + (G.nodes[node]['weight'] / max(node_weights.values())) * 3000
                  for node in G.nodes()]

    # 计算节点颜色
    norm_weights = [(node_weights[node] / max(node_weights.values())) for node in G.nodes()]

    # 创建自定义的红色渐变色谱 (用于节点)
    reds = [(1.0, 0.9, 0.9), (1.0, 0.7, 0.7), (1.0, 0.5, 0.5), (0.9, 0.3, 0.3), (0.7, 0.0, 0.0)]
    red_cmap = LinearSegmentedColormap.from_list("custom_reds", reds)

    # 创建自定义的绿色渐变色谱 (用于边)
    greens = [(0.4, 0.8, 0.4), (0.2, 0.6, 0.2), (0.0, 0.4, 0.0)]
    green_cmap = LinearSegmentedColormap.from_list("custom_greens", greens)

    # 获取当前所有节点的布局位置
    pos = nx.spring_layout(G, seed=42)  # 或使用你原本的布局方式，比如 fruchterman_reingold_layout 等

    # 设置“知识重构类”节点位于画布中心
    pos["知识重构类"] = (0.0, 0.0)  # 设定为画布中心坐标
    pos["知识规划类"] = (0.7, -0.7)

    # 绘制节点 - 使用红色色谱
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=norm_weights,
        cmap=red_cmap,  # 使用红色色谱
        alpha=0.9,
        # edgecolors='darkred',  # 深红色边框
        linewidths=1.5
    )

    # 添加节点标签
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
        edge_cmap=green_cmap,  # 使用绿色色谱
        alpha=0.9,
        arrowsize=15,  # 增大箭头
        arrowstyle='-|>',  # 使用更明显的箭头样式
        connectionstyle=[f'arc3,rad={0.3 * G[u][v]['two']}' for u, v in G.edges()],
        min_source_margin=25,  # 从源节点边缘的更大距离
        min_target_margin=25  # 到目标节点边缘的更大距离
        # 使用列表推导式生成不同弧度
    )

    # 绘制边标签（概率）- 恢复使用原始方法自动放置
    edge_labels = {(u, v): f"{G[u][v]['probability']:.1%}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=12,  # 增大字号提高可读性
        font_weight='bold',  # 加粗文字
        font_color='darkgreen',  # 深绿色文字
        bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.9),  # 添加白色背景和细边框
        font_family=prop.get_name(),
        label_pos=0.6,  # 放置在边的中点
        rotate=False  # 不旋转标签
    )

    # plt.title(title, fontsize=18, pad=20, fontproperties=chinese_font)
    plt.axis('off')

    # 添加图例 - 更新色彩说明
    # plt.text(
    #     0.95, 0.05,
    #     "节点：红色，大小代表动词重要性\n边：绿色，宽度代表转移频率\n箭头方向：表示转移方向\n边标签：转移概率",
    #     transform=plt.gca().transAxes,
    #     fontsize=12,
    #     horizontalalignment='right',
    #     verticalalignment='bottom',
    #     bbox=dict(facecolor='white', alpha=0.8, edgecolor='#999999'),
    #     fontproperties=chinese_font
    # )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=2000, bbox_inches='tight')
    print(f"已保存为: {output_filename}")
    return plt


if __name__ == '__main__':
    visualize_paper_ready_network()
