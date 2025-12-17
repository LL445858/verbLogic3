#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14 
# @Author  : YinLuLu
# @File    : plotcat.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = [
    ('知识重构类', 'a', 369),
    ('知识协作类', 'a', 87),
    ('成果影响类', 'a', 74),
    ('知识规划类', 'a', 80),
    ('成果发布类', 'a', 59),
    ('资源获取类', 'a', 54),
    ('知识整合类', 'a', 37),
]

data.sort(key=lambda x: x[2], reverse=True)
verbs = [item[0] for item in data]
frequencies = [item[2] for item in data]
categories = [item[1] for item in data]
category_colors = {
    'a': '#DBEDC5'
}

colors = [category_colors[cat] for cat in categories]
plt.figure(figsize=(6, 6))
bars = plt.bar(np.arange(len(verbs)), frequencies,
               color=colors, width=0.4,
               edgecolor='none',
               alpha=1,
               capstyle='round',
               zorder=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#A3A3A3')

plt.ylabel('出现次数', fontsize=15)
plt.xlabel('阶段类别', fontsize=15)
plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontsize=14, rotation=45)
plt.ylim(0, max(frequencies) + 3)
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 0.5, str(freq),
             va='bottom', ha='center',
             fontsize=14, weight='bold')


plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
plt.xlim(-0.5, len(verbs) - 0.5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# plt.show()
plt.savefig('cat_plot.png', dpi=1200)
