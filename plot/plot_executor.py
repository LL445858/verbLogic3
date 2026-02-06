#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import importlib.util
spec = importlib.util.spec_from_file_location("executor", os.path.join(project_root, "statistics", "executor.py"))
executor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor_module)

count_executor_verbs = executor_module.count_executor_verbs
count_executor_verbs_by_stage = executor_module.count_executor_verbs_by_stage

mpl.use('TkAgg')

font_cn = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=14)
font_cn_large = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=16)
font_en = FontProperties(fname=r'C:\Windows\Fonts\times.ttf', size=14)
plt.rcParams['axes.unicode_minus'] = False

category_colors = {
    '成果发布类': '#E7BDC7',
    '成果影响类': '#F0E68C',
    '知识协作类': '#FECEA0',
    '知识整合类': '#B7A8CF',
    '知识规划类': '#F0A586',
    '知识重构类': '#8989BD',
    '资源获取类': '#C4E2EC'
}


def plot_top10_executor_verbs_all_stages(executor_counts: dict, save_path: str = None):
    top_10 = sorted(executor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    verbs = [item[0] for item in top_10]
    frequencies = [item[1] for item in top_10]
    
    colors = ['#8989BD'] * len(verbs)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(verbs)), frequencies,
                   color=colors, width=0.6,
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
    
    plt.ylabel('出现次数', fontproperties=font_cn_large, fontsize=15)
    plt.xlabel('动词名称', fontproperties=font_cn_large, fontsize=15)
    plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontproperties=font_cn, fontsize=14)
    plt.yticks(fontproperties=font_en, fontsize=12)
    plt.ylim(0, max(frequencies) + 2)
    
    for i, freq in enumerate(frequencies):
        plt.text(i, freq + 0.3, str(freq),
                 va='bottom', ha='center',
                 fontproperties=font_en, fontsize=12, weight='bold')
    
    plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
    plt.xlim(-0.5, len(verbs) - 0.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_top10_executor_verbs_by_stage(stage_counts: dict, save_dir: str = None):
    for stage in sorted(stage_counts.keys()):
        stage_verbs = stage_counts[stage]
        top_10 = sorted(stage_verbs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        verbs = [item[0] for item in top_10]
        frequencies = [item[1] for item in top_10]
        color = category_colors.get(stage, '#8989BD')
        colors = [color] * len(verbs)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(np.arange(len(verbs)), frequencies,
                       color=colors, width=0.6,
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
        
        plt.ylabel('出现次数', fontproperties=font_cn_large, fontsize=15)
        plt.xlabel('动词名称', fontproperties=font_cn_large, fontsize=15)
        plt.title(f'{stage}', fontproperties=font_cn_large, fontsize=16, pad=20)
        plt.xticks(np.arange(len(verbs)), verbs, ha='center', fontproperties=font_cn, fontsize=14)
        plt.yticks(fontproperties=font_en, fontsize=12)
        plt.ylim(0, max(frequencies) + 1 if max(frequencies) > 5 else max(frequencies) + 0.5)
        
        for i, freq in enumerate(frequencies):
            plt.text(i, freq + 0.2, str(freq),
                     va='bottom', ha='center',
                     fontproperties=font_en, fontsize=12, weight='bold')
        
        plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
        plt.xlim(-0.5, len(verbs) - 0.5)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        
        if save_dir:
            safe_stage_name = stage.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(save_dir, f'top10_{safe_stage_name}.png')
            plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def plot_executor_verb_types_by_stage(stage_counts: dict, save_path: str = None):
    stages = sorted(stage_counts.keys())
    verb_counts = [len(stage_counts[stage]) for stage in stages]
    colors = [category_colors.get(stage, '#8989BD') for stage in stages]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(stages)), verb_counts,
                   color=colors, width=0.6,
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
    
    plt.ylabel('动词种类数量', fontproperties=font_cn_large, fontsize=15)
    plt.xlabel('阶段名称', fontproperties=font_cn_large, fontsize=15)
    plt.xticks(np.arange(len(stages)), stages, ha='center', fontproperties=font_cn, fontsize=14, rotation=45)
    plt.yticks(fontproperties=font_en, fontsize=12)
    plt.ylim(0, max(verb_counts) + 5)
    
    for i, count in enumerate(verb_counts):
        plt.text(i, count + 0.5, str(count),
                 va='bottom', ha='center',
                 fontproperties=font_en, fontsize=12, weight='bold')
    
    plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
    plt.xlim(-0.5, len(stages) - 0.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_executor_verb_frequencies_by_stage(stage_counts: dict, save_path: str = None):
    stages = sorted(stage_counts.keys())
    total_frequencies = [sum(stage_counts[stage].values()) for stage in stages]
    colors = [category_colors.get(stage, '#8989BD') for stage in stages]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(stages)), total_frequencies,
                   color=colors, width=0.6,
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
    
    plt.ylabel('出现总频次', fontproperties=font_cn_large, fontsize=15)
    plt.xlabel('阶段名称', fontproperties=font_cn_large, fontsize=15)
    plt.xticks(np.arange(len(stages)), stages, ha='center', fontproperties=font_cn, fontsize=14, rotation=45)
    plt.yticks(fontproperties=font_en, fontsize=12)
    plt.ylim(0, max(total_frequencies) + 20)
    
    for i, freq in enumerate(total_frequencies):
        plt.text(i, freq + 2, str(freq),
                 va='bottom', ha='center',
                 fontproperties=font_en, fontsize=12, weight='bold')
    
    plt.grid(axis='y', linestyle='-', alpha=0.5, zorder=0)
    plt.xlim(-0.5, len(stages) - 0.5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'result', 'gold')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'figure')
    
    v_file = os.path.join(base_dir, 'v.txt')
    a_file = os.path.join(base_dir, 'a.txt')
    c_file = os.path.join(base_dir, 'c.txt')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        executor_counts = count_executor_verbs(v_file, a_file)
        stage_counts = count_executor_verbs_by_stage(v_file, a_file, c_file)
        
        print("正在生成图表...")
        
        plot_top10_executor_verbs_all_stages(
            executor_counts,
            save_path=os.path.join(output_dir, 'top10_executor_verbs_all_stages.png')
        )
        print("已生成: top10_executor_verbs_all_stages.png")
        
        plot_top10_executor_verbs_by_stage(
            stage_counts,
            save_dir=output_dir
        )
        print("已生成各阶段Top10动词图表")
        
        plot_executor_verb_types_by_stage(
            stage_counts,
            save_path=os.path.join(output_dir, 'executor_verb_types_by_stage.png')
        )
        print("已生成: executor_verb_types_by_stage.png")
        
        plot_executor_verb_frequencies_by_stage(
            stage_counts,
            save_path=os.path.join(output_dir, 'executor_verb_frequencies_by_stage.png')
        )
        print("已生成: executor_verb_frequencies_by_stage.png")
        
        print("\n所有图表生成完成！")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
