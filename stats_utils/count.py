#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 
# @Author  : YinLuLu
# @File    : count.py
# @Software: PyCharm

import ast
import re
import seaborn as sns
import matplotlib
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def verb_count_pre():
    def load_cilin_dict():
        word_to_codes = {}
        code_to_words = {}

        with open(r"Y:\Project\Python\VerbLogic\data\analysis\sys_dict.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '@' in line:
                    continue
                if '=' in line:
                    code, words_str = line.split('=')
                    words = words_str.strip().split()
                    code_to_words[code] = words
                    for word in words:
                        if word in word_to_codes:
                            word_to_codes[word].append(code)
                        else:
                            word_to_codes[word] = [code]
                if '#' in line:
                    code, words_str = line.split('#')
                    words = words_str.strip().split()
                    code_to_words[code] = words
                    for word in words:
                        if word in word_to_codes:
                            word_to_codes[word].append(code)
                        else:
                            word_to_codes[word] = [code]

        return word_to_codes, code_to_words

    word_to_codes, code_to_words = load_cilin_dict()
    count_verb = dict()
    sys_verb = dict(dict())
    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\v.txt', 'r', encoding='utf-8') as f:
        verbs = json.load(f)
    for data in verbs:
        for verb in verbs[data]:
            if verb in word_to_codes:
                verb_s = code_to_words[word_to_codes[verb][0]][0]
            else:
                verb_s = verb
            if verb_s in count_verb.keys():
                count_verb[verb_s] += 1
            else:
                count_verb[verb_s] = 1
            if verb_s in sys_verb.keys():
                if verb in sys_verb[verb_s].keys():
                    sys_verb[verb_s][verb] += 1
                else:
                    sys_verb[verb_s][verb] = 1
            else:
                sys_verb[verb_s] = {verb: 1}

    with open(r'Y:\Project\Python\VerbLogic\data\analysis\verb_sys.txt', 'w', encoding='utf-8') as f:
        for key, value in sorted(count_verb.items(), key=lambda item: item[1], reverse=True):
            f.write(f"共出现{value}次:\t")
            for k, v in sorted(sys_verb[key].items(), key=lambda item: item[1], reverse=True):
                f.write(f"{k}({v}次)、")
            f.write("\n")


def parse_data(content):
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
    v_dict = {}
    with open(path, 'r', encoding='utf-8') as vf:
        for line in vf:
            n, v_list = line.strip().split()
            v_list = [v[:2] for v in v_list.split('、') if v]
            for v in v_list:
                v_dict[v] = v_list[0]
    return v_dict


def category_count():
    category_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
                     "5": 0, "6": 0}
    category_index = {"知识规划类": 0, "知识整合类": 1, "资源获取类": 2, "知识协作类": 3, "知识重构类": 4,
                      "成果发布类": 5, "成果影响类": 6}
    category_matrix = [[0 for _ in range(7)] for _ in range(7)]

    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()
    c_content = parse_data(c_content)

    for i in range(1, 42):
        category = c_content[f'data{i}']
        c_v = list(category.values())
        for j in range(len(c_v) - 1):
            category_matrix[category_index[c_v[j]]][category_index[c_v[j + 1]]] += 1

        for j in range(len(c_v)):
            category_dict[str(category_index[c_v[j]])] += 1

    df = pd.DataFrame(category_matrix, index=list(category_index.keys()), columns=list(category_index.keys()))
    df.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\阶段转移频次.xlsx", index=True)

    categories = ["知识规划", "知识整合", "资源获取", "知识协作", "知识重构", "成果发布", "成果影响"]
    category_matrix = np.array(category_matrix, dtype=float)

    plt.figure(figsize=(7.5, 6))
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
    plt.xlabel("转移到的类别", fontsize=12)
    plt.ylabel("起始类别", fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig(r"Y:\Project\Python\VerbLogic\data\figure\阶段转移热力图.svg", bbox_inches='tight', dpi=2000)
    print(category_dict)


def cate_attr_count():
    def parse_a_content(content):
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
                verb_counter[verb] += 1
                suffix = f"_{verb_counter[verb]}" if verb_counter[verb] > 1 else ""
                unique_verb = f"{verb}{suffix}"
                attribute_categories = list(attributes.keys())
                data_verbs[unique_verb] = attribute_categories

            parsed_data[data_key] = data_verbs

        return parsed_data

    def count_attribute_categories(a_content, c_content):
        a_data = parse_a_content(a_content)
        c_data = parse_data(c_content)
        category_stats = defaultdict(lambda: defaultdict(int))

        for data_key in a_data:
            if data_key not in c_data:
                continue

            a_verbs = a_data[data_key]
            c_verbs = c_data[data_key]
            for verb, attributes in a_verbs.items():
                if verb in c_verbs:
                    verb_class = c_verbs[verb]
                    for attr in attributes:
                        category_stats[verb_class][attr] += 1
        return {cls: dict(attrs) for cls, attrs in category_stats.items()}

    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\a.txt', 'r', encoding='utf-8') as f:
        a_content = f.read()

    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()

    result = count_attribute_categories(a_content, c_content)
    total = {}
    for k, v in result.items():
        print(f'\n{k}: ')
        sum_v = 0
        for vk, vv in v.items():
            sum_v += vv
        for vk, vv in v.items():
            v[vk] = vv / sum_v * 100
        for vk, vv in sorted(v.items(), key=lambda v: v[1], reverse=True):
            print(f"{vk},{vv:.2f}%  ", end='')
        for vk, vv in v.items():
            total[vk] = total.get(vk, 0) + vv

    print("\n\n\n总：")
    for k, v in sorted(total.items(), key=lambda t: t[1], reverse=True):
        print(f"{k}, {v}次, 占比{v / sum(total.values()) * 100:.2f}%")


def verb_cate_move(markov=True):
    v_c_set = set()
    verb_dict = verb_sys(r"Y:\Project\Python\VerbLogic\data\analysis\verb_lulu.txt")
    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
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
            if v_c_1 in v_c_list and v_c_2 in v_c_list:
                v2v_matrix[v_c_list.index(v_c_1), v_c_list.index(v_c_2)] += 1

        for i in range(len(v_list) - 1):
            vc = (v_list[i], c_list[i])
            v_c_dict[vc] += 1

    df = pd.DataFrame(v2v_matrix, index=v_c_list, columns=v_c_list)
    df.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移频次.xlsx", index=True)

    v2v_p_matrix = np.zeros((len(v_c_list), len(v_c_list)))
    for i in range(len(v_c_list)):
        row_sum = np.sum(v2v_matrix[i, :])
        if row_sum > 0:
            # 正常情况：有出边转移，按频次计算概率
            for j in range(len(v_c_list)):
                v2v_p_matrix[i, j] = v2v_matrix[i, j] / row_sum
        elif markov:
            # 特殊情况：该节点没有出边转移（只被其他节点转移过来）
            # 设置自转移概率为1，符合马尔可夫转移概率矩阵要求
            v2v_p_matrix[i, i] = 1.0

    df = pd.DataFrame(v2v_p_matrix, index=v_c_list, columns=v_c_list)
    df.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移概率_非马尔可夫.xlsx", index=True)
    
    # 将v_c_dict转换为三列表格：动词、类别、频次
    v_c_data = []
    for vc, count in v_c_dict.items():
        v_c_data.append({'动词阶段': vc, '频次': count})
    df_vc = pd.DataFrame(v_c_data)
    df_vc = df_vc.sort_values(by='频次', ascending=False)
    df_vc.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别频次.xlsx", index=False)

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


def verb_cate_move_part(markov=True):
    v_c_set = set()
    verb_dict = verb_sys(r"Y:\Project\Python\VerbLogic\data\analysis\verb_lulu.txt")
    with open(r'Y:\Project\Python\VerbLogic\data\result\gold\c.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()

    for v in verb_dict.keys():
        c_content = c_content.replace(v, verb_dict[v])
    c_content = parse_data(c_content)

    for i in list(range(1, 17)) + list(range(39, 42)):
        del c_content[f"data{i}"]

    for v2c in c_content.values():
        for v, c in v2c.items():
            if '_' in v:
                v = v.split('_')[0]
            if v in verb_dict.keys():
                v_c_set.add((v, c))

    v_c_list = list(v_c_set)
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
            if v_c_1 in v_c_list and v_c_2 in v_c_list:
                v2v_matrix[v_c_list.index(v_c_1), v_c_list.index(v_c_2)] += 1

        for i in range(len(v_list) - 1):
            vc = (v_list[i], c_list[i])
            v_c_dict[vc] += 1

    df = pd.DataFrame(v2v_matrix, index=v_c_list, columns=v_c_list)
    df.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移频次_部分.xlsx", index=True)

    v2v_p_matrix = np.zeros((len(v_c_list), len(v_c_list)))
    for i in range(len(v_c_list)):
        row_sum = np.sum(v2v_matrix[i, :])
        if row_sum > 0:
            # 正常情况：有出边转移，按频次计算概率
            for j in range(len(v_c_list)):
                v2v_p_matrix[i, j] = v2v_matrix[i, j] / row_sum
        elif markov:
            # 特殊情况：该节点没有出边转移（只被其他节点转移过来）
            # 设置自转移概率为1，符合马尔可夫转移概率矩阵要求
            v2v_p_matrix[i, i] = 1.0

    df = pd.DataFrame(v2v_p_matrix, index=v_c_list, columns=v_c_list)
    df.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移概率_部分.xlsx", index=True)
    
    # 将v_c_dict转换为三列表格：动词、类别、频次
    v_c_data = []
    for vc, count in v_c_dict.items():
        v_c_data.append({'动词阶段': vc, '频次': count})
    df_vc = pd.DataFrame(v_c_data)
    df_vc = df_vc.sort_values(by='频次', ascending=False)
    df_vc.to_excel(r"Y:\Project\Python\VerbLogic\data\excel\动词类别频次_部分.xlsx", index=False)

if __name__ == '__main__':
    verb_cate_move_part()
