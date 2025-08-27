#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14 
# @Author  : LiXiang
# @File    : test2.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 给定的动词频率数据
data = [
    ('知识重构类', 'a', 369),
    ('知识协作类', 'a', 87),
    ('成果影响类', 'a', 74),
    ('知识规划类', 'a', 80),
    ('成果发布类', 'a', 59),
    ('资源获取类', 'a', 54),
    ('知识整合类', 'a', 37),
]

# 按频率降序排序
data.sort(key=lambda x: x[2], reverse=True)

# 准备数据
verbs = [item[0] for item in data]
frequencies = [item[2] for item in data]
categories = [item[1] for item in data]

# 为每个类别定义唯一的颜色
category_colors = {
    'a': '#DBEDC5'
}

# 为每个动词分配对应类别的颜色
colors = [category_colors[cat] for cat in categories]

# 创建图表
plt.figure(figsize=(6, 6))

# 创建垂直柱状图 (x轴为动词，y轴为频率)
bars = plt.bar(np.arange(len(verbs)), frequencies,
               color=colors, width=0.4,
               edgecolor='none',  # 移除柱子边框
               alpha=1,
               capstyle='round',
               zorder=10)

ax = plt.gca()

# 保留底部边框，移除其余三边
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.5)  # 可选：设置下框线宽度
ax.spines['bottom'].set_color('#A3A3A3')

# 添加标题和标签
plt.ylabel('出现次数', fontsize=15)
plt.xlabel('阶段类别', fontsize=15)
plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontsize=14, rotation=45)
plt.ylim(0, max(frequencies) + 3)

# 在柱条顶部添加数值标签
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 0.5, str(freq),
             va='bottom', ha='center',
             fontsize=14, weight='bold')

# 创建图例
# legend_handles = [plt.Rectangle((0, 0), 1, 1, color=category_colors[cat], alpha=1, label=cat)
#                   for cat in category_colors]
# plt.legend(handles=legend_handles,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 1.15),
#            ncol=len(category_colors),
#            frameon=False,
#            prop={'size': 12})

# 添加网格线（水平）
plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)

# 设置x轴范围，留出空间
plt.xlim(-0.5, len(verbs) - 0.5)

# 调整布局，增加底部空间避免标签被裁切
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# 保存高清图片
# plt.show()
plt.savefig('cat_plot.png', dpi=1200)
