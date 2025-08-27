import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 给定的动词频率数据
data = [
    ('计算', '知识重构类', 27),
    ('采用', '资源获取类', 22),
    ('发表', '成果发布类', 22),
    ('分析', '知识重构类', 19),
    ('研究', '知识重构类', 18),
    ('发现', '知识重构类', 17),
    ('讨论', '知识协作类', 15),
    ('试验', '知识重构类', 15),
    ('获取', '知识重构类', 14),
    ('决定', '知识规划类', 13),
    ('观察', '知识重构类', 13),
    ('带领', '知识协作类', 11),
    ('研制', '知识重构类', 10),
    ('提出', '知识重构类', 10),
    ('证明', '知识重构类', 9),
    ('测量', '知识重构类', 9),
    ('指导', '知识协作类', 9),
    ('考虑', '知识重构类', 8),
    ('探索', '知识重构类', 7),
    ('解决', '知识重构类', 7),
    ('组织', '知识协作类', 7),
    ('根据', '资源获取类', 7),
    ('建立', '知识规划类', 7),
    ('改进', '知识重构类', 6),
    ('引起', '成果影响类', 6),
    ('认为', '知识重构类', 6),
    ('证明', '成果影响类', 6),
    ('确定', '知识规划类', 6),
    ('设计', '知识重构类', 6),
    ('表明', '知识重构类', 6),
    ('建立', '知识重构类', 6),
    ('成为', '成果影响类', 6),
    ('提出', '成果发布类', 6),
]

# 按频率降序排序
data.sort(key=lambda x: x[2], reverse=True)

# 准备数据
verbs = [item[0] for item in data]
frequencies = [item[2] for item in data]
categories = [item[1] for item in data]

# 为每个类别定义唯一的颜色
category_colors = {
    '知识重构类': '#8989BD',
    '资源获取类': '#B7A8CF',
    '成果发布类': '#E7BDC7',
    '知识协作类': '#FECEA0',
    '知识规划类': '#F0A586',
    '成果影响类': '#F0E68C'
}

# 为每个动词分配对应类别的颜色
colors = [category_colors[cat] for cat in categories]

# 创建图表
plt.figure(figsize=(20, 6))

# 创建垂直柱状图 (x轴为动词，y轴为频率)
bars = plt.bar(np.arange(len(verbs)), frequencies,
               color=colors, width=0.6,
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
ax.spines['bottom'].set_linewidth(1)  # 可选：设置下框线宽度
ax.spines['bottom'].set_color('#A3A3A3')

# 添加标题和标签
plt.ylabel('出现次数', fontsize=15)
plt.xlabel('动词名称', fontsize=15)
plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontsize=14, rotation=45)
plt.ylim(0, max(frequencies) + 3)

# 在柱条顶部添加数值标签
for i, freq in enumerate(frequencies):
    plt.text(i, freq + 0.5, str(freq),
             va='bottom', ha='center',
             fontsize=14, weight='bold')

# 创建图例
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=category_colors[cat], alpha=1, label=cat)
                  for cat in category_colors]
plt.legend(handles=legend_handles,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.15),
           ncol=len(category_colors),
           frameon=False,
           prop={'size': 15})

# 添加网格线（水平）
plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)

# 设置x轴范围，留出空间
plt.xlim(-0.5, len(verbs) - 0.5)

# 调整布局，增加底部空间避免标签被裁切
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)

# 保存高清图片
# plt.show()
plt.savefig('verb_plot.png', dpi=1200)
