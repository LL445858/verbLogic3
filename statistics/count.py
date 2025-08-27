#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 
# @Author  : LiXiang
# @File    : help.py
# @Software: PyCharm

import ast
import re
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# def verb_count_pre():
#     # 根据哈工大同义词典计算每个动词出现的次数
#
#     def load_cilin_dict():
#         word_to_codes = {}
#         code_to_words = {}
#
#         with open(r"Y:\Project\PythonProject\VerbLogic\data\analysis\sys.txt", 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line or '@' in line:
#                     continue  # 忽略弱关系
#                 if '=' in line:
#                     code, words_str = line.split('=')
#                     words = words_str.strip().split()
#                     code_to_words[code] = words
#                     for word in words:
#                         if word in word_to_codes:
#                             word_to_codes[word].append(code)
#                         else:
#                             word_to_codes[word] = [code]
#                 if '#' in line:
#                     code, words_str = line.split('#')
#                     words = words_str.strip().split()
#                     code_to_words[code] = words
#                     for word in words:
#                         if word in word_to_codes:
#                             word_to_codes[word].append(code)
#                         else:
#                             word_to_codes[word] = [code]
#
#         return word_to_codes, code_to_words
#
#     word_to_codes, code_to_words = load_cilin_dict()
#     count_verb = dict()
#     sys_verb = dict(dict())
#     with open(r'Y:\Project\PythonProject\VerbLogic\data\extract\verbs\gold.txt', 'r', encoding='utf-8') as f:
#         verbs = json.load(f)
#     for data in verbs:
#         for verb in verbs[data]:
#             if verb in word_to_codes:
#                 verb_s = code_to_words[word_to_codes[verb][0]][0]
#             else:
#                 verb_s = verb
#             if verb_s in count_verb.keys():
#                 count_verb[verb_s] += 1
#             else:
#                 count_verb[verb_s] = 1
#             if verb_s in sys_verb.keys():
#                 if verb in sys_verb[verb_s].keys():
#                     sys_verb[verb_s][verb] += 1
#                 else:
#                     sys_verb[verb_s][verb] = 1
#             else:
#                 sys_verb[verb_s] = {verb: 1}
#
#     with open(r'Y:\Project\PythonProject\VerbLogic\data\analysis\verb.txt', 'w', encoding='utf-8') as f:
#         for key, value in sorted(count_verb.items(), key=lambda item: item[1], reverse=True):
#             f.write(f"共出现{value}次:\t")
#             for k, v in sorted(sys_verb[key].items(), key=lambda item: item[1], reverse=True):
#                 f.write(f"{k}({v}次)、")
#             f.write("\n")
#
#
# def c_verb_plot(sj):
#     # 绘制动词类别柱状图
#     # 设置画布
#     plt.figure(figsize=(14, 6), dpi=100)
#     ax = plt.subplot(111)
#
#     # 获取所有唯一的属性类型
#     # all_attributes = set()
#     # for category in sj.values():
#     #     for name, _ in category:
#     #         all_attributes.add(name)
#     # print(all_attributes)
#
#     # 为每个属性类型分配唯一颜色（使用tab20配色方案）
#     # cmap = get_cmap('tab20')
#     # # pyplot.get_cmap()
#     # attribute_colors = {}
#     # for i, attr_name in enumerate(all_attributes):
#     #     attribute_colors[attr_name] = cmap(i % 20)  # tab20提供20种不同颜色
#
#     # 获取类别列表
#     categories = list(sj.keys())
#     n_categories = len(categories)
#
#     # 布局参数设置
#     bar_width = 0.35  # 柱子宽度
#     inner_gap = 0.15  # 大类内部柱子间距
#     outer_gap = 0.0  # 大类之间的间隔[6](@ref)
#
#     # 计算大类中心位置
#     x_centers = np.arange(0, n_categories * (3 + outer_gap), (3 + outer_gap))[:n_categories]
#
#     # 绘制柱状图
#     for i, category in enumerate(categories):
#         # 计算当前大类内部的柱子位置（对称分布）
#         x_positions = x_centers[i] + np.array([-2 * (bar_width + inner_gap),
#                                                -(bar_width + inner_gap),
#                                                0,
#                                                bar_width + inner_gap,
#                                                2 * (bar_width + inner_gap)])
#
#         # 获取当前大类的属性数据
#         sj_data = sj[category]
#
#         # 绘制当前大类的五个柱子
#         for j in range(5):
#             name, value = sj_data[j]
#             # color = attribute_colors[name]
#             color = "#9DD6E1"
#             bar = ax.bar(x_positions[j], value, width=bar_width,
#                          color=color, edgecolor=color, linewidth=0.7)
#
#             # 在柱子顶部添加属性标签（旋转45度避免重叠）
#             plt.text(x_positions[j], value + 0.1, name,
#                      ha='center', va='bottom',
#                      fontsize=8,
#                      # rotation=45,
#                      fontweight='bold', alpha=0.9)
#
#     # 设置坐标轴和标签
#     plt.xticks(x_centers, categories, fontsize=12, fontweight='bold')
#     plt.ylabel('动词频次', fontsize=12, fontweight='bold')
#     plt.ylim(0, 25)  # 设置Y轴范围
#
#     # 添加网格线和样式优化
#     plt.grid(axis='y', alpha=0.2, linestyle='--')
#     plt.tight_layout()
#
#     # 显示图表
#     plt.show()
#
#
# def c_attr_plot(sj):
#     # 绘取属性类别柱状图
#     plt.figure(figsize=(14, 6), dpi=100)
#     ax = plt.subplot(111)
#
#     # 获取所有唯一的属性类型
#     # all_attributes = set()
#     # for category in sj.values():
#     #     for name, _ in category:
#     #         all_attributes.add(name)
#     # print(all_attributes)
#     #
#     # # 为每个属性类型分配唯一颜色（使用tab20配色方案）
#     # cmap = get_cmap('tab20')
#     # # pyplot.get_cmap()
#     # attribute_colors = {}
#     # for i, attr_name in enumerate(all_attributes):
#     #     attribute_colors[attr_name] = cmap(i % 20)  # tab20提供20种不同颜色
#
#     # 获取类别列表
#     categories = list(sj.keys())
#     n_categories = len(categories)
#
#     # 布局参数设置
#     bar_width = 0.35  # 柱子宽度
#     inner_gap = 0.15  # 大类内部柱子间距
#     outer_gap = 0.0  # 大类之间的间隔[6](@ref)
#
#     # 计算大类中心位置
#     x_centers = np.arange(0, n_categories * (3 + outer_gap), (3 + outer_gap))[:n_categories]
#
#     # 绘制柱状图
#     for i, category in enumerate(categories):
#         # 计算当前大类内部的柱子位置（对称分布）
#         x_positions = x_centers[i] + np.array([-2 * (bar_width + inner_gap),
#                                                -(bar_width + inner_gap),
#                                                0,
#                                                bar_width + inner_gap,
#                                                2 * (bar_width + inner_gap)])
#
#         # 获取当前大类的属性数据
#         sj_data = sj[category]
#
#         # 绘制当前大类的五个柱子
#         for j in range(5):
#             name, value = sj_data[j]
#             # color = attribute_colors[name]
#             color = '#9DD6E1'
#             bar = ax.bar(x_positions[j], value, width=bar_width,
#                          color=color, edgecolor=color, linewidth=0.7)
#
#             # 在柱子顶部添加属性标签（旋转45度避免重叠）
#             plt.text(x_positions[j], value + 5, name,
#                      ha='center', va='bottom',
#                      fontsize=8, rotation=45,
#                      fontweight='bold', alpha=0.9)
#
#     # 设置坐标轴和标签
#     plt.xticks(x_centers, categories, fontsize=12, fontweight='bold')
#     plt.ylabel('属性类别频次', fontsize=12, fontweight='bold')
#     plt.ylim(0, 330)  # 设置Y轴范围
#
#     # 添加网格线和样式优化
#     plt.grid(axis='y', alpha=0.2, linestyle='--')
#     plt.tight_layout()
#
#     # 显示图表
#     plt.show()
#
#
# def c_verb_plot_pre(sj):
#     # 绘取动词类别百分比柱状图
#     # 设置画布
#     plt.figure(figsize=(14, 6), dpi=100)
#     ax = plt.subplot(111)
#
#     # 获取所有唯一的属性类型
#     # all_attributes = set()
#     # for category in sj.values():
#     #     for name, _ in category:
#     #         all_attributes.add(name)
#     # print(all_attributes)
#
#     # 为每个属性类型分配唯一颜色（使用tab20配色方案）
#     # cmap = get_cmap('tab20')
#     # # pyplot.get_cmap()
#     # attribute_colors = {}
#     # for i, attr_name in enumerate(all_attributes):
#     #     attribute_colors[attr_name] = cmap(i % 20)  # tab20提供20种不同颜色
#
#     # 获取类别列表
#     categories = list(sj.keys())
#     n_categories = len(categories)
#
#     # 布局参数设置
#     bar_width = 0.35  # 柱子宽度
#     inner_gap = 0.15  # 大类内部柱子间距
#     outer_gap = 0.0  # 大类之间的间隔[6](@ref)
#
#     # 计算大类中心位置
#     x_centers = np.arange(0, n_categories * (3 + outer_gap), (3 + outer_gap))[:n_categories]
#
#     # 绘制柱状图
#     for i, category in enumerate(categories):
#         # 计算当前大类内部的柱子位置（对称分布）
#         x_positions = x_centers[i] + np.array([-2 * (bar_width + inner_gap),
#                                                -(bar_width + inner_gap),
#                                                0,
#                                                bar_width + inner_gap,
#                                                2 * (bar_width + inner_gap)])
#
#         # 获取当前大类的属性数据
#         sj_data = sj[category]
#
#         # 绘制当前大类的五个柱子
#         for j in range(5):
#             name, value = sj_data[j]
#             # color = attribute_colors[name]
#             color = "#9DD6E1"
#             bar = ax.bar(x_positions[j], value, width=bar_width,
#                          color=color, edgecolor=color, linewidth=0.7)
#
#             # 在柱子顶部添加属性标签（旋转45度避免重叠）
#             plt.text(x_positions[j], value + 0.1, name,
#                      ha='center', va='bottom',
#                      fontsize=8,
#                      # rotation=45,
#                      fontweight='bold', alpha=0.9)
#
#     # 设置坐标轴和标签
#     plt.xticks(x_centers, categories, fontsize=12, fontweight='bold')
#     plt.ylabel('动词频次占比（%）', fontsize=12, fontweight='bold')
#     plt.ylim(0, 50)  # 设置Y轴范围
#
#     # 添加网格线和样式优化
#     plt.grid(axis='y', alpha=0.2, linestyle='--')
#     plt.tight_layout()
#
#     # 显示图表
#     plt.show()
#
#
# def c_attr_pre_plot(sj):
#     # 绘取属性类别百分比柱状图
#     # 设置画布
#     plt.figure(figsize=(14, 6), dpi=100)
#     ax = plt.subplot(111)
#
#     # 获取所有唯一的属性类型
#     # all_attributes = set()
#     # for category in sj.values():
#     #     for name, _ in category:
#     #         all_attributes.add(name)
#     # print(all_attributes)
#     #
#     # # 为每个属性类型分配唯一颜色（使用tab20配色方案）
#     # cmap = get_cmap('tab20')
#     # # pyplot.get_cmap()
#     # attribute_colors = {}
#     # for i, attr_name in enumerate(all_attributes):
#     #     attribute_colors[attr_name] = cmap(i % 20)  # tab20提供20种不同颜色
#
#     # 获取类别列表
#     categories = list(sj.keys())
#     n_categories = len(categories)
#
#     # 布局参数设置
#     bar_width = 0.35  # 柱子宽度
#     inner_gap = 0.15  # 大类内部柱子间距
#     outer_gap = 0.0  # 大类之间的间隔[6](@ref)
#
#     # 计算大类中心位置
#     x_centers = np.arange(0, n_categories * (3 + outer_gap), (3 + outer_gap))[:n_categories]
#
#     # 绘制柱状图
#     for i, category in enumerate(categories):
#         # 计算当前大类内部的柱子位置（对称分布）
#         x_positions = x_centers[i] + np.array([-2 * (bar_width + inner_gap),
#                                                -(bar_width + inner_gap),
#                                                0,
#                                                bar_width + inner_gap,
#                                                2 * (bar_width + inner_gap)])
#
#         # 获取当前大类的属性数据
#         sj_data = sj[category]
#
#         # 绘制当前大类的五个柱子
#         for j in range(5):
#             name, value = sj_data[j]
#             # color = attribute_colors[name]
#             color = '#9DD6E1'
#             bar = ax.bar(x_positions[j], value, width=bar_width,
#                          color=color, edgecolor=color, linewidth=0.7)
#
#             # 在柱子顶部添加属性标签（旋转45度避免重叠）
#             plt.text(x_positions[j], value + 1, name,
#                      ha='center', va='bottom',
#                      fontsize=8, rotation=45,
#                      fontweight='bold', alpha=0.9)
#
#     # 设置坐标轴和标签
#     plt.xticks(x_centers, categories, fontsize=12, fontweight='bold')
#     plt.ylabel('属性类别频次占比（%）', fontsize=12, fontweight='bold')
#     plt.ylim(0, 50)  # 设置Y轴范围
#
#     # 添加网格线和样式优化
#     plt.grid(axis='y', alpha=0.2, linestyle='--')
#     plt.tight_layout()
#
#     # 显示图表
#     plt.show()
#
#
# def pie_attr():
#     # 数据准备
#     labels = ['执行者', '受事对象', '成果', '施事对象', '合作者', '开始时间', '方法', '结论', '受事人', '影响', '其他']
#     percentages = [36.33, 23.80, 7.49, 4.74, 3.24, 3.18, 3.12, 2.52, 2.52, 2.46]
#     # 计算剩余部分占比
#     remaining = 100 - sum(percentages)
#     percentages.append(round(remaining, 2))  # 添加剩余部分，保留两位小数
#
#     # 设置颜色（10种鲜明色系 + 灰色表示其他）
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#cccccc']
#
#     # 解决中文显示问题
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 多字体兼容
#     plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
#
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(10, 7))
#
#     # 绘制饼图（关键参数说明）
#     wedges, texts, autotexts = ax.pie(
#         percentages,
#         labels=labels,
#         colors=colors,
#         autopct='%1.1f%%',  # 显示百分比格式
#         startangle=140,  # 起始角度（36.33%从顶部开始）
#         pctdistance=0.85,  # 百分比标签距离中心位置
#         labeldistance=1.05,  # 字段标签基础距离
#         wedgeprops={'linewidth': 1, 'edgecolor': 'white'},  # 区块白边
#         textprops={'fontsize': 9}  # 标签字体
#     )
#
#     # 优化标签位置（折线引至外部）
#     bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="gray", lw=0.5, alpha=0.8)  # 标签背景框样式
#     for i, p in enumerate(wedges):
#         ang = (p.theta2 - p.theta1) / 2. + p.theta1  # 计算区块中心角度
#         y = np.sin(np.deg2rad(ang))  # 极坐标转直角坐标
#         x = np.cos(np.deg2rad(ang))
#
#         # 动态调整长标签位置（水平/垂直对齐方式）
#         horizontal = 'left' if x < 0 else 'right' if x > 0 else 'center'
#         vertical = 'bottom' if y > 0 else 'top' if y < 0 else 'center'
#
#         # 长标签添加折线引导
#         connectionstyle = f"angle,angleA=0,angleB={ang}"
#         kw = dict(arrowprops=dict(arrowstyle="-", color="gray", connectionstyle=connectionstyle),
#                   bbox=bbox_props, zorder=0, ha=horizontal, va=vertical)
#
#         # 仅对需要外部标注的标签调整（避免过度重叠）
#         if i < 10:  # 前10个主要标签
#             ax.annotate(
#                 f"{labels[i]}: {percentages[i]}%",
#                 xy=(x, y),
#                 xytext=(1.3 * np.sign(x), 1.3 * y),
#                 **kw
#             )
#             texts[i].set_visible(False)  # 隐藏原始内部标签
#
#     # 添加标题和图例
#     plt.title('字段分布百分比饼图', fontsize=15, pad=20)
#     plt.legend(wedges, [f"{l}: {p}%" for l, p in zip(labels, percentages)],
#                loc="center left",
#                bbox_to_anchor=(1, 0, 0.5, 1),
#                fontsize=9)
#
#     # 保存高清图像
#     plt.tight_layout()
#     plt.savefig('字段分布饼图.png', dpi=300, bbox_inches='tight')
#     plt.show()
# def network():
#     def v_set_c(s1, s2, path):
#         verb_dict = verb_sys(path)
#         with open(r'Y:\Project\PythonProject\VerbLogic\data\extract\gold\c.txt', 'r', encoding='utf-8') as f:
#             c_content = f.read()
#         for v in verb_dict:
#             c_content = c_content.replace(v, verb_dict[v])
#         c_content = parse_data(c_content)
#
#         # 计算从a类顺承b类的动词列表
#         v_set = set()
#         for v_c in c_content.values():
#             v_list = [v for v in v_c.keys()]
#             for i in range(len(v_list)):
#                 if '_' in v_list[i]:
#                     v_list[i] = v_list[i].split('_')[0]
#             c_list = [v_c[v] for v in v_list]
#             for i in range(len(v_list) - 1):
#                 if v_list[i] in verb_dict.keys() and v_list[i + 1] in verb_dict.keys():
#                     # if c_list[i] == s1 and c_list[i + 1] == s2 and v_list[i] in verb_dict.keys() and v_list[i + 1] in verb_dict.keys():
#                     v_set.add(v_list[i])
#                     v_set.add(v_list[i + 1])
#
#         return list(v_set)
#
#     def matrix_save(s1, s2):
#         path = r'Y:\Project\PythonProject\VerbLogic\data\analysis\verb_del2.txt'
#         v_set = v_set_c(s1, s2, path)
#         # print(v_set)
#         word_count = {v: 0 for v in v_set}
#         co_matrix = np.zeros((len(v_set), len(v_set)), dtype=int)
#         verb_dict = verb_sys(path)
#
#         with open(r'Y:\Project\PythonProject\VerbLogic\data\extract\gold\c.txt', 'r', encoding='utf-8') as f:
#             c_content = f.read()
#         for v in verb_dict.keys():
#             c_content = c_content.replace(v, verb_dict[v])
#         c_content = parse_data(c_content)
#
#         # 统计从a类顺承b类的动词次数
#         for v2c in c_content.values():
#             v_list = [v for v in v2c.keys()]
#             for i in range(len(v_list)):
#                 if '_' in v_list[i]:
#                     v_list[i] = v_list[i].split('_')[0]
#             c_list = [v2c[v] for v in v_list]
#             for i in range(len(v_list) - 1):
#                 if v_list[i] in v_set and v_list[i + 1] in v_set:
#                     # if c_list[i] == s1 and c_list[i + 1] == s2 and v_list[i] in v_set and v_list[i + 1] in v_set:
#                     co_matrix[v_set.index(v_list[i])][v_set.index(v_list[i + 1])] += 1
#                     # co_matrix[v_set.index(v_list[i + 1])][v_set.index(v_list[i])] += 1
#             for i in range(len(v_list)):
#                 if v_list[i] in v_set:
#                     # if (c_list[i] == s1 or c_list[i] == s2) and v_list[i] in v_set:
#                     word_count[v_list[i]] += 1
#
#         df = pd.DataFrame(co_matrix, index=v_set, columns=v_set)
#         df.to_excel(r"Y:\Project\PythonProject\VerbLogic\data\analysis\co_matrix.xlsx", index=True)
#         # matrix = np.zeros_like(co_matrix, dtype=float)
#         #
#         # for i in range(len(v_set)):
#         #     for j in range(len(v_set)):
#         #         if i != j and co_matrix[i][j] > 0:
#         #             Cij = co_matrix[i][j]
#         #             # Ci = word_count[v_set[i]]
#         #             # Cj = word_count[v_set[j]]
#         #             Ci = sum(co_matrix[i])
#         #             Cj = sum(co_matrix[j])
#         #             if Ci > 0 and Cj > 0:
#         #                 matrix[i][j] = Cij / (Ci * Cj) ** 0.5
#         #
#         # df = pd.DataFrame(matrix, index=v_set, columns=v_set)
#         # df.to_excel(r"Y:\Project\PythonProject\VerbLogic\data\analysis\matrix.xlsx", index=True)
#
#     def load_matrix(file_path):
#         df = pd.read_excel(file_path, sheet_name='Sheet1', index_col=0)
#         # 确保索引和列名的一致性
#         if not df.index.tolist() == df.columns.tolist():
#             raise ValueError("词语列表不一致，请检查数据")
#         return df
#
#     # 创建有向图
#     def create_graph(df, threshold=0):
#         G = nx.DiGraph()
#         words = df.index.tolist()
#         # print(words)
#         G.add_nodes_from(words)
#
#         # 添加带权重的边
#         for i, source in enumerate(words):
#             for j, target in enumerate(words):
#                 if i == j:
#                     continue
#                 weight = df.iloc[i, j]
#
#                 # isinstance(weight, (int, float)) and
#                 if weight > threshold:
#                     print(weight)
#                     G.add_edge(source, target, weight=float(weight))
#         return G
#
#     # 可视化有向图
#     def visualize_graph(G, s1, s2):
#         # 设置节点布局
#         pos = nx.spring_layout(G, k=1, iterations=100)
#
#         # 创建图形
#         plt.figure(figsize=(20, 18), dpi=100)
#         ax = plt.gca()
#
#         # 创建渐变色
#         cmap = LinearSegmentedColormap.from_list("custom", ["#FFEEAD", "#FF9E76"])
#
#         # 修正后的颜色计算（修复了括号问题）
#         node_degrees = dict(G.degree(weight='weight'))
#         print(node_degrees)
#         max_degree = max(node_degrees.values()) if node_degrees.values() else 1
#         colors = [cmap(d / max_degree) for d in node_degrees.values()]
#
#         # 计算边的宽度
#         edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
#
#         # 节点绘制
#         nx.draw_networkx_nodes(
#             G, pos,
#             node_size=1200,
#             node_color=colors,
#             alpha=0.9,
#             edgecolors='#4B6584',
#             linewidths=1,
#             ax=ax
#         )
#
#         # 边绘制（带箭头）
#         edges = nx.draw_networkx_edges(
#             G, pos,
#             edgelist=G.edges(),
#             width=edge_weights,
#             edge_color='#4B6584',
#             alpha=0.6,
#             arrowstyle='->',
#             arrowsize=25,
#             connectionstyle='arc3,rad=0.15',  # 添加曲线避免重叠
#             min_source_margin=15,
#             min_target_margin=15,
#             ax=ax
#         )
#
#         # 标签位置调整
#         label_pos = {k: [v[0], v[1]] for k, v in pos.items()}
#         nx.draw_networkx_labels(
#             G, label_pos,
#             font_size=12,
#             font_weight='bold',
#             font_family='SimHei',
#             verticalalignment='center'
#         )
#
#         # 添加图例
#         plt.rcParams['font.sans-serif'] = ['SimHei']
#         plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_degree))
#         sm.set_array([])
#         cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
#         cbar.set_label('节点权重度', fontsize=14)
#         if s1 == s2:
#             title = f"{s1}动词转移图谱"
#         else:
#             title = f"{s1}到{s2}动词转移图谱"
#         plt.title(title, fontsize=24, pad=20)
#         plt.tight_layout()
#         plt.axis('off')
#         plt.savefig(f'Y:\\Project\\PythonProject\\VerbLogic\\data\\analysis\\fig\\动词转移网络\\sum.png',
#                     bbox_inches='tight')
#         # plt.savefig(f'Y:\\Project\\PythonProject\\VerbLogic\\data\\analysis\\fig\\动词转移网络\\{s1}\\{title}.png', bbox_inches='tight')
#         # plt.show()
#         plt.close()
#
#     c_list = ["知识规划类", "知识整合类", "资源获取类", "知识协作类", "知识重构类", "成果发布类", "成果影响类",
#               "知识整合类", "知识整合类"]
#     # for s1 in c_list:
#     #     for s2 in c_list:
#     #         matrix_save(s1, s2)
#     #         df = load_matrix('matrix.xlsx')
#     #         graph = create_graph(df, threshold=0.01)  # 使用更宽松的阈值确保测试显示更多边
#     #         visualize_graph(graph, s1, s2)
#     s1, s2 = '', ''
#     matrix_save(s1, s2)
#     # df = load_matrix(r"Y:\Project\PythonProject\VerbLogic\data\analysis\单向转移频次矩阵.xlsx")
#     # graph = create_graph(df, threshold=0.01)  # 使用更宽松的阈值确保测试显示更多边
#     # visualize_graph(graph, s1, s2)

def parse_data(content):
    """解析c.txt文件内容，处理重复动词"""
    data_blocks = re.findall(r'"(data\d+)":\s*\{(.*?)\}(?=,\s*"data\d+":|\s*\})', content, re.DOTALL)
    parsed_data = {}
    for key, block in data_blocks:
        items = re.findall(r'"(.*?)":"(.*?)"', block)
        counter = defaultdict(int)
        word_labels = {}
        for word, label in items:
            counter[word] += 1
            suffix = f"_{counter[word]}" if counter[word] > 1 else ""
            word_labels[f"{word}{suffix}"] = label
        parsed_data[key] = word_labels
    return parsed_data


def verb_sys(path):
    """
    返回同义词典
    """
    v_dict = {}
    with open(path, 'r', encoding='utf-8') as vf:
        for line in vf:
            n, v_list = line.strip().split()
            # n = int(n[3:-2])
            v_list = [v[:2] for v in v_list.split('、') if v]
            for v in v_list:
                v_dict[v] = v_list[0]
    return v_dict


def category_count():
    # 计算种类数量、种类高频动词和转移概率
    category_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
                     "5": 0, "6": 0}
    category_index = {"知识规划类": 0, "知识整合类": 1, "资源获取类": 2, "知识协作类": 3, "知识重构类": 4,
                      "成果发布类": 5, "成果影响类": 6}
    category_matrix = [[0 for _ in range(7)] for _ in range(7)]

    with open(r'Y:\Project\PythonProject\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()
    c_content = parse_data(c_content)

    for i in range(1, 42):
        category = c_content[f'data{i}']
        c_v = list(category.values())
        for j in range(len(c_v) - 1):
            category_matrix[category_index[c_v[j]]][category_index[c_v[j + 1]]] += 1

        for j in range(len(c_v)):
            category_dict[str(category_index[c_v[j]])] += 1

    # c = 7
    # for i in range(c):
    #     for j in range(c):
    #         # category_matrix[i][j] = (category_matrix[i][j] + 0.1) / (category_dict[str(i)] + 0.1 * c)
    #         category_matrix[i][j] = category_matrix[i][j] / category_dict[str(i)]

    print(category_matrix)
    df = pd.DataFrame(category_matrix, index=list(category_index.keys()), columns=list(category_index.keys()))
    df.to_excel(r"Y:\Project\PythonProject\VerbLogic\data\excel\阶段转移频次.xlsx", index=True)

    #     for key, value in category.items():
    #         category_dict[value] += 1
    #         if value == "知识规划类":
    #             zsgh[verb_dict[key]] = zsgh.get(verb_dict[key], 0) + 1
    #         elif value == "知识整合类":
    #             zszh[verb_dict[key]] = zszh.get(verb_dict[key], 0) + 1
    #         elif value == "资源获取类":
    #             zyhq[verb_dict[key]] = zyhq.get(verb_dict[key], 0) + 1
    #         elif value == "知识协作类":
    #             zsxz[verb_dict[key]] = zsxz.get(verb_dict[key], 0) + 1
    #         elif value == "知识重构类":
    #             zscg[verb_dict[key]] = zscg.get(verb_dict[key], 0) + 1
    #         elif value == "成果发布类":
    #             cgfb[verb_dict[key]] = cgfb.get(verb_dict[key], 0) + 1
    #         elif value == "成果影响类":
    #             cgyx[verb_dict[key]] = cgyx.get(verb_dict[key], 0) + 1
    # for c in [zsgh, zszh, zyhq, zsxz, zscg, cgfb, cgyx]:
    #     print('-------------------------------------')
    #     i = 0
    #     for k, v in sorted(c.items(), key=lambda item: item[1], reverse=True):
    #         i += 1
    #         print(f'{k}({v / sum(c.values()) * 100:.2f}次)、', end='')
    #         if i == 5:
    #             break
    #     print()
    #
    categories = ["知识规划", "知识整合", "资源获取", "知识协作", "知识重构", "成果发布", "成果影响"]
    category_matrix = np.array(category_matrix, dtype=float)

    plt.figure(figsize=(7.5, 6))
    # sns.set(font='SimHei')  # 设置中文字体

    # 绘制热力图（小数格式）
    ax = sns.heatmap(
        category_matrix,
        annot=True,
        fmt='.0f',
        cmap='Reds',
        xticklabels=categories,
        yticklabels=categories,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': '转移频次'}
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # plt.title("类别转移概率热力图", fontsize=14)
    plt.xlabel("转移到的类别", fontsize=12)
    plt.ylabel("起始类别", fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"Y:\Project\PythonProject\VerbLogic\data\figure\阶段转移热力图.svg", bbox_inches='tight', dpi=2000)
    print(category_dict)


def cate_attr_count():
    # 属性类别数量计算
    def parse_a_content(content):
        """解析a.txt文件内容，处理重复动词并提取属性类别"""
        # 使用ast安全解析非标准JSON格式
        try:
            data_dict = ast.literal_eval(a_content)
        except Exception as e:
            print("Parse error:", e)
            return None

        parsed_data = {}
        for data_key, verb_dict in data_dict.items():
            verb_counter = defaultdict(int)
            data_verbs = {}

            for verb, attributes in verb_dict.items():
                # 处理重复动词
                verb_counter[verb] += 1
                suffix = f"_{verb_counter[verb]}" if verb_counter[verb] > 1 else ""
                unique_verb = f"{verb}{suffix}"

                # 提取属性类别（字典键）
                attribute_categories = list(attributes.keys())
                data_verbs[unique_verb] = attribute_categories

            parsed_data[data_key] = data_verbs

        return parsed_data

    def count_attribute_categories(a_content, c_content):
        """统计每个类别下各个属性类别的出现次数"""
        # 解析两个文件
        a_data = parse_a_content(a_content)
        c_data = parse_data(c_content)

        # 创建统计字典
        category_stats = defaultdict(lambda: defaultdict(int))

        # 遍历所有data块
        for data_key in a_data:
            if data_key not in c_data:
                continue

            a_verbs = a_data[data_key]
            c_verbs = c_data[data_key]

            # 遍历当前data块中的所有动词
            for verb, attributes in a_verbs.items():
                if verb in c_verbs:
                    verb_class = c_verbs[verb]
                    # 统计每个属性类别
                    for attr in attributes:
                        category_stats[verb_class][attr] += 1

        # 转换为普通字典
        return {cls: dict(attrs) for cls, attrs in category_stats.items()}

    # 读取文件内容
    with open(r'Y:\Project\PythonProject\VerbLogic\data\extract\gold\a.txt', 'r', encoding='utf-8') as f:
        a_content = f.read()

    with open(r'Y:\Project\PythonProject\VerbLogic\data\extract\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()

    # 执行统计
    result = count_attribute_categories(a_content, c_content)

    total = {}
    # 输出结果
    for k, v in result.items():
        # print(f'\n{k}: ')
        # sum_v = 0
        # for vk, vv in v.items():
        #     sum_v += vv
        # for vk, vv in v.items():
        #     v[vk] = vv / sum_v * 100
        # for vk, vv in sorted(v.items(), key=lambda v: v[1], reverse=True):
        #     print(f"{vk},{vv:.2f}  ", end='')
        for vk, vv in v.items():
            total[vk] = total.get(vk, 0) + vv
    for k, v in sorted(total.items(), key=lambda t: t[1], reverse=True):
        print(f"{k}, {v / sum(total.values()) * 100:.2f}")


def verb_cate_move():
    v_c_set = set()
    verb_dict = verb_sys(r"Y:\Project\PythonProject\VerbLogic\data\analysis\verb_lulu.txt")
    with open(r'Y:\Project\PythonProject\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()
    for v in verb_dict.keys():
        c_content = c_content.replace(v, verb_dict[v])
    c_content = parse_data(c_content)
    for v2c in c_content.values():
        for v, c in v2c.items():
            if '_' in v:
                v = v.split('_')[0]
            if v in verb_dict.keys():
                v_c_set.add((v, c))

    v_c_list = list(v_c_set)
    # v_c_num = len(v_c_list)
    v_c_dict = {(v, c): 0 for v, c in v_c_list}
    v2v_matrix = np.zeros((len(v_c_list), len(v_c_list)))
    for v2c in c_content.values():
        v_list = list(v2c.keys())
        c_list = list(v2c.values())

        for i in range(len(v_list)):
            if '_' in v_list[i]:
                v_list[i] = v_list[i].split('_')[0]

        for i in range(len(v_list) - 1):
            v_c_1 = (v_list[i], c_list[i])
            v_c_2 = (v_list[i + 1], c_list[i + 1])
            if v_list[i] == '探索' and v_list[i + 1] == '建议':
                print(list(c_content.values()).index(v2c) + 1)
            if v_c_1 in v_c_list and v_c_2 in v_c_list:
                v2v_matrix[v_c_list.index(v_c_1), v_c_list.index(v_c_2)] += 1

        for i in range(len(v_list)):
            vc = (v_list[i], c_list[i])
            v_c_dict[vc] += 1

    v2v_p_matrix = np.zeros((len(v_c_list), len(v_c_list)))
    for i in range(len(v_c_list)):
        for j in range(len(v_c_list)):
            v2v_p_matrix[i, j] = v2v_matrix[i, j] / v_c_dict[v_c_list[i]]

    df = pd.DataFrame(v2v_p_matrix, index=v_c_list, columns=v_c_list)
    df.to_excel(r"Y:\Project\PythonProject\VerbLogic\data\excel\动词类别转移概率.xlsx", index=True)

    # pre_c, next_c = '资源获取类', '知识重构类'
    # move_num = {}
    # for i in range(len(v_c_list)):
    #     for j in range(len(v_c_list)):
    #         if v_c_list[i][1] == pre_c and v_c_list[j][1] == next_c:
    #             move_num[(i, j)] = v2v_matrix[i, j]
    #
    # print(f"{pre_c} 到 {next_c}的转移统计：")
    # for k, v in sorted(move_num.items(), key=lambda t: t[1], reverse=True):
    #     if v == 0:
    #         continue
    #     print(f"{v_c_list[k[0]][0]}->{v_c_list[k[1]][0]}\t频次:{v}\t转移概率:{v2v_p_matrix[k[0]][k[1]] * 100:.2f}%")


if __name__ == '__main__':
    verb_cate_move()
