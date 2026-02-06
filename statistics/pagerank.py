"""
PageRank 算法实现

基于一阶马尔可夫链的随机游走模型，用于评估动词阶段节点在知识网络中的重要性。

算法核心思想：
- 将动词阶段节点看作有向图的节点
- 节点间的转移概率构成马尔可夫转移矩阵
- 通过迭代计算收敛到平稳分布，即为各节点的 PageRank 值

迭代公式：
    R_{t+1} = (d * M + (1-d)/n * E) * R_t

其中：
    - M: 马尔可夫转移矩阵 (n×n)
    - R: 节点重要性向量 (n×1)
    - d: 阻尼因子，通常取 0.85
    - E: 全1矩阵 (n×n)
    - n: 节点数量
"""

import pandas as pd
import numpy as np


def compute_pagerank(
    transition_matrix_path: str,
    initial_vector_path: str,
    damping_factor: float = 0.85,
    epsilon: float = 1e-8,
    max_iterations: int = 1000
) -> pd.DataFrame:
    """
    计算 PageRank 值

    参数:
        transition_matrix_path: 马尔可夫转移矩阵文件路径 (Excel格式)
        initial_vector_path: 初始词频统计向量文件路径 (Excel格式)
        damping_factor: 阻尼因子 d，默认 0.85
        epsilon: 收敛阈值，默认 1e-8
        max_iterations: 最大迭代次数，默认 1000

    返回:
        DataFrame: 包含动词阶段及其 PageRank 值的结果
    """

    # 读取马尔可夫转移矩阵 M
    df_m = pd.read_excel(transition_matrix_path)

    # 第一列是索引（动词阶段名称），提取节点列表
    node_names = df_m.iloc[:, 0].tolist()

    # 提取转移矩阵（去掉第一列索引）
    M = df_m.iloc[:, 1:].values.astype(float)

    n = len(node_names)
    print(f"节点数量: {n}")
    print(f"转移矩阵形状: {M.shape}")

    # 读取初始词频统计向量
    df_r = pd.read_excel(initial_vector_path)

    # 构建初始向量 R0（按频次归一化）
    # 创建动词阶段到频次的映射
    freq_dict = dict(zip(df_r['动词阶段'], df_r['频次']))

    # 按照转移矩阵的节点顺序构建初始向量
    R = np.zeros(n)
    for i, node in enumerate(node_names):
        R[i] = freq_dict.get(node, 0)

    # 归一化初始向量（使其成为概率分布）
    total_freq = R.sum()
    if total_freq > 0:
        R = R / total_freq
    else:
        # 如果总频次为0，使用均匀分布
        R = np.ones(n) / n

    print(f"初始向量总和: {R.sum():.6f}")

    # 构建全1矩阵 E 的缩放部分: (1-d)/n * E
    # 实际上只需要计算 (1-d)/n * E * R_t = (1-d)/n * sum(R_t) = (1-d)/n
    # 因为 R_t 是概率分布，sum(R_t) = 1
    teleport_prob = (1 - damping_factor) / n

    # 迭代计算 PageRank
    iteration = 0
    diff = float('inf')

    print(f"\n开始迭代计算 (阻尼因子 d={damping_factor}, 阈值 ε={epsilon})...")

    while diff > epsilon and iteration < max_iterations:
        # 计算 R_{t+1} = d * M^T * R_t + (1-d)/n * 1
        # 注意：M[i,j] 表示从节点i转移到节点j的概率
        # 所以计算节点j的得分需要累加所有指向j的节点的贡献：sum_i M[i,j] * R[i]
        # 即 R_new = M^T @ R

        R_new = damping_factor * (M.T @ R) + teleport_prob

        # 归一化（防止数值误差导致总和不为1）
        R_new = R_new / R_new.sum()

        # 计算变化量（L2范数）
        diff = np.linalg.norm(R_new - R, 2)

        R = R_new
        iteration += 1

        if iteration % 100 == 0:
            print(f"  迭代 {iteration}: 变化量 = {diff:.10f}")

    print(f"\n迭代完成！")
    print(f"总迭代次数: {iteration}")
    print(f"最终变化量: {diff:.10e}")
    print(f"PageRank 值总和: {R.sum():.6f}")

    # 构建结果 DataFrame
    result = pd.DataFrame({
        '动词阶段': node_names,
        'PageRank': R,
        '初始频次': [freq_dict.get(node, 0) for node in node_names]
    })

    # 按 PageRank 值降序排序
    result = result.sort_values(by='PageRank', ascending=False).reset_index(drop=True)

    return result


def main():
    """主函数：执行 PageRank 计算并保存结果"""

    # 文件路径
    transition_matrix_path = r"Y:\Project\Python\VerbLogic\data\excel\动词类别转移概率_非马尔可夫.xlsx"
    initial_vector_path = r"Y:\Project\Python\VerbLogic\data\excel\动词类别频次.xlsx"
    output_path = r"Y:\Project\Python\VerbLogic\data\excel\PageRank.xlsx"

    print("=" * 60)
    print("PageRank 算法计算")
    print("=" * 60)

    # 计算 PageRank
    pagerank_result = compute_pagerank(
        transition_matrix_path=transition_matrix_path,
        initial_vector_path=initial_vector_path,
        damping_factor=0.85,
        epsilon=1e-10,
        max_iterations=1000
    )

    # 显示前10个结果
    print(f"\nTop 10 重要节点:")
    print(pagerank_result.head(10).to_string(index=False))

    # 保存结果到 Excel
    pagerank_result.to_excel(output_path, index=False)
    print(f"\n结果已保存至: {output_path}")
    print(f"共 {len(pagerank_result)} 个节点")


if __name__ == '__main__':
    main()
