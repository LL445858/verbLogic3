#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/8 
# @Author  : LiXiang
# @File    : F1.py
# @Software: PyCharm

import re
import json
import jieba
import random
import matplotlib

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def category_score(model_name):
    # Step 1: Load raw content with preserved duplicates using regex

    with open(r"Y:\Project\PythonProject\VerbLogic\data\result\category\gold.txt", "r", encoding="utf-8") as f:
        content1 = f.read()
    with open(f"Y:\\Project\\PythonProject\\VerbLogic\\data\\result\\category\\{model_name}.txt",
              encoding='utf-8') as f:
        content2 = f.read()

    def parse_data(content):
        data_blocks = re.findall(r'"(data\d+)":\s*\{(.*?)\}(?=,\s*"data\d+":|\s*\})', content, re.DOTALL)
        parsed_data = {}
        for key, block in data_blocks:
            # Handle duplicates by appending _1, _2, etc.
            items = re.findall(r'"(.*?)":"(.*?)"', block)
            counter = defaultdict(int)
            word_labels = {}
            for word, label in items:
                counter[word] += 1
                suffix = f"_{counter[word]}" if counter[word] > 1 else ""
                word_labels[f"{word}{suffix}"] = label
            parsed_data[key] = word_labels
        # print(parsed_data)
        return parsed_data

    data1 = parse_data(content1)
    data2 = parse_data(content2)

    # Step 2: Compute Cohen Kappa for each data item
    kappas = []
    for i in range(1, 42):
        key = f"data{i}"
        d1 = data1.get(key, {})
        d2 = data2.get(key, {})
        common_keys = sorted(set(d1.keys()) & set(d2.keys()))
        if not common_keys:
            # kappas.append(None)
            continue
        y_true = [d1[k] for k in common_keys]
        y_pred = [d2[k] for k in common_keys]
        score = cohen_kappa_score(y_true, y_pred)
        kappas.append(score)

    # Step 3: Global confusion matrix
    all_true = []
    all_pred = []
    for key in data1:
        d1 = data1[key]
        d2 = data2.get(key, {})
        for word in d1:
            if word in d2:
                all_true.append(d1[word])
                all_pred.append(d2[word])
    labels = ['知识规划类', "知识整合类", "资源获取类", "知识协作类", "知识重构类", "成果发布类", "成果影响类"]
    cm = confusion_matrix(all_true, all_pred, labels=labels)

    # Step 4: Metrics
    # print(all_true)
    # print(all_pred)
    precision = precision_score(all_true, all_pred, labels=labels, average='weighted')
    recall = recall_score(all_true, all_pred, labels=labels, average='weighted')
    f1 = f1_score(all_true, all_pred, labels=labels, average='weighted')
    accuracy = accuracy_score(all_true, all_pred)
    print(f'\n{model_name}:')
    # print(f'一致性：{np.average(kappas) * 100:.2f}%')
    print(f'A值: {accuracy * 100:.2f}%, P值: {precision * 100:.2f}%, R值: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%')

    # Step 5: Plot heatmap
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.xlabel('yinlulu')
    # plt.ylabel('lixiang')
    # plt.title('7x7 Confusion Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(f"{model_name}.png")
    # plt.show()


def verb_score(model_name):
    def compute_metrics(a, b):
        # 多重集合统计
        counter_a = Counter(a)
        counter_b = Counter(b)

        # 所有可能的实体标签（全集）
        all_entities = sorted(set(counter_a.keys()) | set(counter_b.keys()))

        # 构造布尔标签向量，用于 Kappa 计算
        vec_a = []
        vec_b = []
        for entity in all_entities:
            count = max(counter_a[entity], counter_b[entity])
            a = random.randint(1, 10)
            vec_a.extend([a] * counter_a[entity] + [0] * (count - counter_a[entity]))
            vec_b.extend([a] * counter_b[entity] + [0] * (count - counter_b[entity]))

        kappa = cohen_kappa_score(vec_a, vec_b)

        # 计算 PRF (以 a 为 gold standard)
        tp = sum(min(counter_a[ent], counter_b[ent]) for ent in all_entities)
        fp = sum(counter_b[ent] - min(counter_a[ent], counter_b[ent]) for ent in all_entities)
        fn = sum(counter_a[ent] - min(counter_a[ent], counter_b[ent]) for ent in all_entities)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return kappa, precision, recall, f1

    kappa_score = []
    precision_score = []
    recall_score = []
    f1_score = []
    with open(r"Y:\Project\PythonProject\VerbLogic\data\result\verbs\gold.txt", encoding='utf-8') as f:
        json_a = json.load(f)
    with open(f"Y:\\Project\\PythonProject\\VerbLogic\\data\\result\\verbs\\{model_name}.txt", encoding='utf-8') as f:
        json_b = json.load(f)
    for i in range(1, 35):
        v_a = json_a[f"data{i}"]
        v_b = json_b[f"data{i}"]
        kappa, precision, recall, f1 = compute_metrics(v_a, v_b)
        kappa_score.append(kappa)
        precision_score.append(precision)
        recall_score.append(recall)
        f1_score.append(f1)
    # for i in range(len(kappa_score)):
    #     print(f"第{i+1}篇:\t{kappa_score[i]*100:.2f}%")
    # print('动词标注一致性平均值', average(kappa_score))
    print(f'{model_name}:')
    print(f'动词标注准确率: {np.average(precision_score) * 100:.2f}%', end='\t\t')
    print(f'动词标注召回率: {np.average(recall_score) * 100:.2f}%', end='\t\t')
    print(f'动词标注F1:    {np.average(f1_score) * 100:.2f}%')


def attr_score(model):
    def is_value_match(a, b):
        a_words = set(a)
        b_words = set(b)
        return len(a_words & b_words) > 0

    def parse_custom_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = f.read()

        # Find all 'dataX' blocks using a more reliable regex split
        block_splits = re.split(r'"(data\d+)":\s*{', raw)[1:]  # drop content before first "dataX"
        parsed_data = {}

        for i in range(0, len(block_splits), 2):
            data_id = block_splits[i]
            content = block_splits[i + 1]

            # Remove trailing brace that might belong to the enclosing object
            content = re.sub(r'}\s*$', '', content.strip())

            # Match each verb block
            verb_matches = re.finditer(r'"(.*?)"\s*:\s*{(.*?)}(?=,\s*"(.*?)"\s*:\s*{|\s*}$)', content, re.DOTALL)
            verbs = defaultdict(list)

            for match in verb_matches:
                verb, attr_block = match.group(1), match.group(2)
                attr_dict = {}
                for attr_match in re.finditer(r'"(.*?)"\s*:\s*"(.*?)"', attr_block):
                    attr, val = attr_match.group(1), attr_match.group(2)
                    attr_dict[attr] = val
                verbs[verb].append(attr_dict)

            # Final verb block (edge case not captured above)
            tail_match = re.findall(r'"(.*?)"\s*:\s*{(.*?)}\s*$', content.strip(), re.DOTALL)
            for verb, attr_block in tail_match:
                attr_dict = {}
                for attr_match in re.finditer(r'"(.*?)"\s*:\s*"(.*?)"', attr_block):
                    attr, val = attr_match.group(1), attr_match.group(2)
                    attr_dict[attr] = val
                verbs[verb].append(attr_dict)

            numbered_verbs = {}
            for verb, items in verbs.items():
                for idx, item in enumerate(items, 1):
                    numbered_verbs[f"{verb}#{idx}"] = item

            parsed_data[data_id] = numbered_verbs
        return parsed_data

    data_a = parse_custom_json(r'Y:\Project\PythonProject\VerbLogic\data\extract\attr\gold.txt')
    data_b = parse_custom_json(f'Y:\\Project\\PythonProject\\VerbLogic\\data\\extract\\attr\\{model}.txt')

    kappa_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    rouge = Rouge()
    smooth_fn = SmoothingFunction().method2
    bleu_scores_1 = []
    bleu_scores_2 = []
    rouge_scores = [[], [], [], [], [], [], [], [], []]

    for index in range(1, 42):
        text_id = f"data{index}"
        verbs_a = data_a.get(text_id, {})
        verbs_b = data_b.get(text_id, {})

        # 一致性
        # all_verbs = set(verbs_a.keys()).union(verbs_b.keys())
        # attr_a_list = []
        # attr_b_list = []
        # for verb in all_verbs:
        #     attrs_a = verbs_a.get(verb, {})
        #     attrs_b = verbs_b.get(verb, {})
        #     all_attrs = set(attrs_a.keys()).union(attrs_b.keys())
        #
        #     for attr in all_attrs:
        #         val_a = attrs_a.get(attr, "")
        #         val_b = attrs_b.get(attr, "")
        #         # if is_value_match(val_a, val_b):
        #         if val_a == val_b:
        #             num = random.randint(1, 10)
        #             attr_a_list.append(num)
        #             attr_b_list.append(num)
        #         else:
        #             attr_a_list.append(1)
        #             attr_b_list.append(-1)
        #
        # kappa_results.append((text_id, cohen_kappa_score(attr_a_list, attr_b_list)))

        # P/R/F1
        true_items = {(verb, attr, val) for verb, attr_dict in verbs_a.items() for attr, val in attr_dict.items()}
        pred_items = {(verb, attr, val) for verb, attr_dict in verbs_b.items() for attr, val in attr_dict.items()}

        tp = 0
        used_preds = set()
        for t in true_items:
            for p in pred_items:
                if p not in used_preds and (t[0] == p[0]) and (t[1] == p[1]):
                    # if p not in used_preds and (t[0] == p[0]) and (t[1] == p[1]) and t[2] == p[2]:
                    # if p not in used_preds and (t[0] == p[0]) and (t[1] == p[1]) and is_value_match(t[2], p[2]):
                    tp += 1
                    used_preds.add(p)
                    break
        fp = len(pred_items) - tp
        fn = len(true_items) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_results.append((text_id, precision))
        recall_results.append((text_id, recall))
        f1_results.append((text_id, f1))

        # blue
        for verb in sorted(set(verbs_a.keys()).union(verbs_b.keys())):
            ref_attrs = verbs_a.get(verb, {})
            hyp_attrs = verbs_b.get(verb, {})

            for attr in sorted(set(ref_attrs.keys()).union(hyp_attrs.keys())):
                ref_val = ref_attrs.get(attr, "")
                hyp_val = hyp_attrs.get(attr, "")

                if ref_val and hyp_val:
                    bleu_1 = sentence_bleu([list(jieba.cut(hyp_val))], list(jieba.cut(hyp_val)),
                                           smoothing_function=smooth_fn, weights=[1, 0, 0, 0])
                    bleu_scores_1.append(bleu_1)
                    bleu_2 = sentence_bleu([list(jieba.cut(hyp_val))], list(jieba.cut(hyp_val)),
                                           smoothing_function=smooth_fn, weights=[0.5, 0.5, 0, 0])
                    bleu_scores_2.append(bleu_2)

                    rouge_result = rouge.get_scores(" ".join(jieba.cut(hyp_val)), " ".join(jieba.cut(ref_val)), avg=True)
                    rouge_scores[0].append(rouge_result['rouge-1']['r'])
                    rouge_scores[1].append(rouge_result['rouge-1']['p'])
                    rouge_scores[2].append(rouge_result['rouge-1']['f'])
                    rouge_scores[3].append(rouge_result['rouge-2']['r'])
                    rouge_scores[4].append(rouge_result['rouge-2']['p'])
                    rouge_scores[5].append(rouge_result['rouge-2']['f'])
                    rouge_scores[6].append(rouge_result['rouge-l']['r'])
                    rouge_scores[7].append(rouge_result['rouge-l']['p'])
                    rouge_scores[8].append(2*rouge_scores[6][-1]*rouge_scores[7][-1]/(rouge_scores[6][-1]+rouge_scores[7][-1])
                                           if rouge_scores[7][-1] + rouge_scores[6][-1] != 0 else 0)

                    if rouge_scores[8][-1] <= rouge_scores[7][-1] and rouge_scores[8][-1] <= rouge_scores[6][-1]:
                        print(ref_val, hyp_val, rouge_result['rouge-l'])

    # print(f'一致性：{np.average([k[1] for k in kappa_results]) * 100: .2f} % , r:{rouge_scores}
    print(f'\n{model}:')
    print(
        f'P值：{np.average([p[1] for p in precision_results]) * 100:.2f}%  R值：{np.average([r[1] for r in recall_results]) * 100:.2f}% F1值：{np.average([f[1] for f in f1_results]) * 100:.2f}%')
    # print(f'bleu-1 : {np.average(bleu_scores_1) * 100:.2f}%\nbleu-2 : {np.average(bleu_scores_2) * 100:.2f}%')
    # print(R_results)
    # print(
    #     f'rouge-1: P:{np.average(rouge_scores[1]) * 100:.2f}%\tR:{np.average(rouge_scores[0]) * 100:.2f}%\tF1:{np.average(rouge_scores[2]) * 100:.2f}%')
    # print(
    #     f'rouge-2: P:{np.average(rouge_scores[4]) * 100:.2f}%\tR:{np.average(rouge_scores[3]) * 100:.2f}%\tF1:{np.average(rouge_scores[5]) * 100:.2f}%')
    print(
        f'rouge-l: P:{np.average(rouge_scores[7]) * 100:.2f}%\tR:{np.average(rouge_scores[6]) * 100:.2f}%\tF:{np.average(rouge_scores[8]) * 100:.2f}%\n')


if __name__ == "__main__":
    model_list = ['baichuan3', 'baichuan4', 'glm4', 'glmz', 'deepseek_r1', 'deepseek_v3', 'doubao_15', 'doubao_16',
                  'qwen3', 'qwen_plus']
    for i in model_list:
        category_score(i)
    # category_score('deepseek_v3')
