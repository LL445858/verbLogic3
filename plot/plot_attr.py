#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14 
# @Author  : YinLuLu
# @File    : plot_attr.py
# @Software: PyCharm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
data = [
    ('执行者', 'a', 606),
    ('受事对象', 'a', 397),
    ('成果', 'a', 125),
    ('施事对象', 'a', 79),
    ('合作者', 'a', 54),
    ('开始时间', 'a', 53),
    ('方法', 'a', 52),
    ('受事人', 'a', 42),
    ('结论', 'a', 42),
    ('影响', 'a', 41),
    ('发生地点', 'a', 37),
    ('学术背景', 'a', 30),
    ('学术载体', 'a', 19),
    ('理论', 'a', 18),
    ('工具', 'a', 16),
    ('资料', 'a', 16),
    ('状况', 'a', 13),
    ('数据', 'a', 11),
    ('持续时间', 'a', 9),
    ('其他', 'a', 8),
]

data.sort(key=lambda x: x[2], reverse=True)
verbs = [item[0] for item in data]
frequencies = [item[2] for item in data]
categories = [item[1] for item in data]
category_colors = {
    'a': '#C4E2EC'
}

colors = [category_colors[cat] for cat in categories]
plt.figure(figsize=(14, 6))
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
plt.xlabel('属性类别', fontsize=15)
plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontsize=14, rotation=45)
plt.ylim(0, max(frequencies) + 3)

for i, freq in enumerate(frequencies):
    plt.text(i, freq + 0.5, str(freq),
             va='bottom', ha='center',
             fontsize=14, weight='bold')
plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
plt.xlim(-0.5, len(verbs) - 0.5)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.show()
# plt.savefig('attr_plot.svg', dpi=1200)