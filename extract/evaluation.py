#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/8 
# @Author  : YinLuLu
# @File    : evaluation.py
# @Software: PyCharm

import json
import random
import re
from collections import Counter, defaultdict
import jieba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rouge import Rouge
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def category_score(model_name):
    with open(r"Y:\Project\Python\VerbLogic\data\result\gold\c.txt", "r", encoding="utf-8") as f:
        content1 = f.read()
    with open(f"Y:\\Project\\Python\\VerbLogic\\data\\result\\category\\{model_name}.txt",
              encoding='utf-8') as f:
        content2 = f.read()

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

    data1 = parse_data(content1)
    data2 = parse_data(content2)
    kappas = []
    for i in range(1, 4):
        key = f"data{i}"
        d1 = data1.get(key, {})
        d2 = data2.get(key, {})
        common_keys = sorted(set(d1.keys()) & set(d2.keys()))
        if not common_keys:
            continue
        y_true = [d1[k] for k in common_keys]
        y_pred = [d2[k] for k in common_keys]
        score = cohen_kappa_score(y_true, y_pred)
        kappas.append(score)

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
    precision = precision_score(all_true, all_pred, labels=labels, average='weighted')
    recall = recall_score(all_true, all_pred, labels=labels, average='weighted')
    f1 = f1_score(all_true, all_pred, labels=labels, average='weighted')
    accuracy = accuracy_score(all_true, all_pred)
    print(f'\n{model_name}:')
    print(f'Kappa值：{np.average(kappas) * 100:.2f}%, A值: {accuracy * 100:.2f}%, P值: {precision * 100:.2f}%, R值: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%')

    # cm = confusion_matrix(all_true, all_pred, labels=labels)
    # plt.figure(figsize=(7.5, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.tight_layout()
    # plt.savefig(f"{model_name}.png")
    return kappas, precision, recall, f1


def verb_score(model_name):
    def compute_metrics(a, b):
        counter_a = Counter(a)
        counter_b = Counter(b)
        all_entities = sorted(set(counter_a.keys()) | set(counter_b.keys()))
        vec_a = []
        vec_b = []
        for entity in all_entities:
            count = max(counter_a[entity], counter_b[entity])
            a = random.randint(1, 10)
            vec_a.extend([a] * counter_a[entity] + [0] * (count - counter_a[entity]))
            vec_b.extend([a] * counter_b[entity] + [0] * (count - counter_b[entity]))

        kappa = cohen_kappa_score(vec_a, vec_b)
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
    with open(r"Y:\Project\Python\VerbLogic\data\result\gold\v.txt", encoding='utf-8') as f:
        json_a = json.load(f)
    with open(f"Y:\\Project\\Python\\VerbLogic\\data\\result\\verbs\\{model_name}.txt", encoding='utf-8') as f:
        json_b = json.load(f)
    for i in range(1, 5):
        v_a = json_a[f"data{i}"]
        v_b = json_b[f"data{i}"]
        kappa, precision, recall, f1 = compute_metrics(v_a, v_b)
        kappa_score.append(kappa)
        precision_score.append(precision)
        recall_score.append(recall)
        f1_score.append(f1)
    print(f'{model_name}:')
    print(f'动词标注Kappa:  {np.average(kappa_score) * 100:.2f}%', end='\t\t')
    print(f'动词标注准确率: {np.average(precision_score) * 100:.2f}%', end='\t\t')
    print(f'动词标注召回率: {np.average(recall_score) * 100:.2f}%', end='\t\t')
    print(f'动词标注F1:    {np.average(f1_score) * 100:.2f}%')
    return kappa_score, precision_score, recall_score, f1_score


def attr_score(model):
    def parse_custom_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = f.read()
        block_splits = re.split(r'"(data\d+)":\s*{', raw)[1:]  # drop content before first "dataX"
        parsed_data = {}

        for i in range(0, len(block_splits), 2):
            data_id = block_splits[i]
            content = block_splits[i + 1]
            content = re.sub(r'}\s*$', '', content.strip())
            verb_matches = re.finditer(r'"(.*?)"\s*:\s*{(.*?)}(?=,\s*"(.*?)"\s*:\s*{|\s*}$)', content, re.DOTALL)
            verbs = defaultdict(list)

            for match in verb_matches:
                verb, attr_block = match.group(1), match.group(2)
                attr_dict = {}
                for attr_match in re.finditer(r'"(.*?)"\s*:\s*"(.*?)"', attr_block):
                    attr, val = attr_match.group(1), attr_match.group(2)
                    attr_dict[attr] = val
                verbs[verb].append(attr_dict)

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

    data_a = parse_custom_json(r'Y:\Project\Python\VerbLogic\data\result\gold\a.txt')
    data_b = parse_custom_json(f'Y:\\Project\\Python\\VerbLogic\\data\\result\\attribute\\{model}.txt')

    kappa_score = []
    rouge = Rouge()
    precision_score, recall_score, f1_score = [], [], []

    for index in range(1, 4):
        ka, kb, p, r, f1 = [], [], [], [], []
        text_id = f"data{index}"
        verbs_a = data_a.get(text_id, {})
        verbs_b = data_b.get(text_id, {})
        # print(f"{text_id}:\n{verbs_a}\n{verbs_b}")

        for verb in sorted(set(verbs_a.keys()).union(verbs_b.keys())):
            ref_attrs = verbs_a.get(verb, {})
            hyp_attrs = verbs_b.get(verb, {})

            for attr in sorted(set(ref_attrs.keys()).union(hyp_attrs.keys())):
                ref_val = ref_attrs.get(attr, "")
                hyp_val = hyp_attrs.get(attr, "")
                if ref_val and hyp_val:
                    if len(set(ref_val).intersection(set(hyp_val))) > 0:
                        ka.append(len(set(ref_val).intersection(set(hyp_val))))
                        kb.append(len(set(ref_val).intersection(set(hyp_val))))
                    else:
                        ka.append(len(ref_val))
                        kb.append(-len(hyp_val))
                    rouge_result = rouge.get_scores(" ".join(jieba.cut(hyp_val)), " ".join(jieba.cut(ref_val)), avg=True)
                    r.append(rouge_result['rouge-l']['r'])
                    p.append(rouge_result['rouge-l']['p'])
                    f1.append(rouge_result['rouge-l']['f'])
                elif ref_val and not hyp_val:
                    ka.append(len(ref_val))
                    kb.append(0)
                else:
                    ka.append(0)
                    kb.append(len(hyp_val))


        kappa_score.append(cohen_kappa_score(ka, kb))
        # print("\n\n", kappa_score[-1], "\n", ka, "\n", kb)
        precision_score.append(float(np.average(p)))
        recall_score.append(float(np.average(r)))
        f1_score.append(float(np.average(f1)))

    print(f'\n{model}:')
    print(f'Kappa: {np.average(kappa_score) * 100:.2f}%, P:{np.average(precision_score) * 100:.2f}%\tR:{np.average(recall_score) * 100:.2f}%\tF:{np.average(f1_score) * 100:.2f}%\n')
    return kappa_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    k, p, r, f = attr_score('glm47f')
    print(np.std(k, ddof=1))
    print(k, p, r, f)
