#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 
# @Author  : YinLuLu
# @File    : cluster.py
# @Software: PyCharm

import matplotlib
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def hierarchical():
    file_path = r"Y:\Project\PythonProject\VerbLogic\data\analysis\matrix.xlsx"
    df = pd.read_excel(file_path, index_col=0)
    cooccurrence_values = df.values
    min_val, max_val = np.min(cooccurrence_values), np.max(cooccurrence_values)
    distance_matrix = 1 - (cooccurrence_values - min_val) / (max_val - min_val)
    linkage_matrix = sch.linkage(distance_matrix, method='ward')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage_matrix, labels=df.index.tolist(), leaf_rotation=90, leaf_font_size=10)
    plt.title("层次聚类树状图")
    plt.xlabel("词语")
    plt.ylabel("聚类距离")
    plt.show()


def kmeans():
    def cwc():
        file_path = r"Y:\Project\PythonProject\VerbLogic\data\excel\动词类别转移概率.xlsx"
        df = pd.read_excel(file_path, index_col=0)
        cooccurrence_values = df.values
        min_val, max_val = np.min(cooccurrence_values), np.max(cooccurrence_values)
        distance_matrix = 1 - (cooccurrence_values - min_val) / (max_val - min_val)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(distance_matrix)
        inertia_values = []
        K_range = range(1, 10)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, n_init=1000)
            kmeans.fit(scaled_data)
            inertia_values.append(kmeans.inertia_)
        #
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertia_values, marker="o", linestyle="--")
        plt.xlabel("聚类数")
        plt.ylabel("簇内误差平方和")
        plt.title("簇内误差平方和与聚类数的关系")
        plt.show()

    def lkxs():
        file_path = r"Y:\Project\PythonProject\VerbLogic\data\excel\动词类别转移概率.xlsx"
        df = pd.read_excel(file_path, index_col=0)
        cooccurrence_values = df.values
        min_val, max_val = np.min(cooccurrence_values), np.max(cooccurrence_values)
        distance_matrix = 1 - (cooccurrence_values - min_val) / (max_val - min_val)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(distance_matrix)
        best_score = -1
        silhouette_scores = []

        K_range = range(2, 10)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, n_init=1000)
            labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                best_k = k

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, silhouette_scores, marker="o", linestyle="--")
        plt.xlabel("聚类数 K")
        plt.ylabel("轮廓系数")
        plt.show()

    def k_c(best_k=3):
        file_path = r"Y:\Project\PythonProject\VerbLogic\data\excel\动词类别转移概率.xlsx"
        df = pd.read_excel(file_path, index_col=0)
        cooccurrence_values = df.values
        min_val, max_val = np.min(cooccurrence_values), np.max(cooccurrence_values)
        distance_matrix = 1 - (cooccurrence_values - min_val) / (max_val - min_val)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(distance_matrix)
        kmeans = KMeans(n_clusters=best_k, n_init=10000)
        clusters = kmeans.fit_predict(scaled_data)
        cluster_groups = {i: [] for i in range(best_k)}
        for i, word in enumerate(df.index):
            cluster_groups[clusters[i]].append(word)

        print(f"\nK 值: {best_k}")
        for cluster_id, words in cluster_groups.items():
            print(f"\n簇 {cluster_id + 1} ({len(words)} 个词): {', '.join(words)}")

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        plt.figure(figsize=(8, 6))
        for i in range(best_k):
            plt.scatter(reduced_data[clusters == i, 0], reduced_data[clusters == i, 1], label=f"簇 {i + 1}")

        for i, word in enumerate(df.index):
            plt.annotate(word, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, alpha=0.75)

        plt.xlabel("PCA 维度 1")
        plt.ylabel("PCA 维度 2")
        plt.title(f"K-Means 聚类结果（K={best_k}）")
        plt.legend()
        plt.show()

    # cwc()
    # lkxs()
    k_c(3)


if __name__ == '__main__':
    # hierarchical()
    kmeans()
