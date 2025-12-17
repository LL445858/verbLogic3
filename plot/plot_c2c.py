#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/14
# @Author  : YinLuLu
# @File    : plot_c2c.py
# @Software: PyCharm

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_chord_diagram import chord_diagram

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data1 = pd.read_csv(r"Y:\Project\PythonProject\VerbLogic\data\excel\阶段转移概率_和弦.csv")
data2 = data1.to_numpy()


colors = ['#8989BD', '#B7A8CF', '#E7BDC7', '#FECEA0', '#F0A586', '#F0E68C', '#FFD700']
names = ["知识规划类", "知识整合类", "资源获取类", "知识协作类", "知识重构类", "成果发布类", "成果影响类"]

font = {
    'family': 'Microsoft YaHei',
    'style': 'normal',
    'weight': 'normal',
    'color': 'black',
    'size': 25
}

chord_diagram(data2,
              names,
              pad=5,
              directed=False,
              colors=colors,
              fontsize=8,
              rotate_names=[True, True, True, True, True, True, True],
              )

plt.savefig('test_1' + '.png',
            dpi=500,
            bbox_inches='tight'
            )
