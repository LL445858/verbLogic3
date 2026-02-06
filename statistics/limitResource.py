from __future__ import annotations
import json
from collections import Counter, OrderedDict
from typing import Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
import math
from typing import List, Dict, Optional, Literal, Tuple
import numpy as np



plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def resource_acquire_share_per_data(
    json_input: Union[str, Dict[str, Any]],
    target_label1: str = "资源获取类",
    target_label2: str = "知识重构类",
    sort_key: bool = True
) -> Dict[str, Dict[str, float]]:

    data = json.loads(json_input)

    # 兼容：顶层可能就是 {"data1": {...}, ...}，也可能是 {"data": {...}} 之类
    # 这里按你给的格式（顶层就是 data1..dataN）处理；如果不是，再自己改一下 key。
    items = data.items()

    result = {}
    for k, v in items:
        if not isinstance(v, dict):
            # 若某个 dataX 不是 dict，则跳过或报错，这里选择跳过
            continue

        labels = list(v.values())
        total = len(labels)
        cnt1 = sum(1 for lab in labels if lab == target_label1)
        cnt2 = sum(1 for lab in labels if lab == target_label2)
        share = cnt1 / total 
        share2 = cnt2 / total
        share3 = 0
        for i in range(len(labels) - 1):
            if labels[i] == target_label2 and labels[i+1] == target_label2:
                share3 += 1
        share3 /= (total - 1)


        result[k] = {"资源获取类占比": float(share), "知识重构类占比": float(share2), "知识重构类自转移概率": float(share3)}

    # 排序：data1, data2... 更直观
    if sort_key:
        def _data_num(x: str) -> Tuple[int, str]:
            # 提取 data 后面的数字用于排序
            import re
            m = re.search(r"(\d+)$", x)
            return (int(m.group(1)) if m else 10**9, x)
        result = OrderedDict(sorted(result.items(), key=lambda kv: _data_num(kv[0])))

    return dict(result)


def plot_discrete_pmf_of_shares(
    share_dict: Dict[str, Dict[str, float]],
    bin_width: float = 0.05,
    title: str = "资源获取类占比分布（PMF）"
) -> Dict[float, float]:
    """
    把每个 data 的 share 视为样本，按 bin_width 分箱后画 PMF（离散概率质量函数）。
    返回：{bin_center: probability}
    """
    shares = [v["资源获取类占比"] for v in share_dict.values()]
    if not shares:
        raise ValueError("No share values to plot.")

    # 分箱到离散格点：round(share / bin_width) * bin_width
    binned = [round(s / bin_width) * bin_width for s in shares]

    c = Counter(binned)
    n = len(binned)
    pmf = {k: v / n for k, v in sorted(c.items(), key=lambda x: x[0])}

    xs = list(pmf.keys())
    ps = list(pmf.values())

    plt.figure()
    plt.bar(xs, ps, width=bin_width * 0.9, align="center")
    plt.xlabel(f"资源获取类占比（按 {bin_width:.2f} 分箱）")
    plt.ylabel("概率（频率归一化）")
    plt.title(title)
    plt.xticks(xs, rotation=45)
    plt.tight_layout()
    plt.show()

    return pmf

def _is_finite(a: float) -> bool:
    return isinstance(a, (int, float)) and math.isfinite(a)

def _pearson_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    sxx = sum(v*v for v in dx)
    syy = sum(v*v for v in dy)
    if sxx == 0 or syy == 0:
        raise ValueError("x 或 y 方差为 0（常量序列），Pearson 相关系数未定义。")
    sxy = sum(a*b for a, b in zip(dx, dy))
    return sxy / math.sqrt(sxx * syy)

def _t_cdf(t: float, df: int) -> float:
    """
    t 分布 CDF。优先用 scipy；若没有 scipy，则报错（你也可按需换成 mpmath）。
    """
    try:
        from scipy.stats import t as t_dist  # type: ignore
        return float(t_dist.cdf(t, df))
    except Exception as e:
        raise ImportError(
            "需要 scipy 才能计算 t 分布 CDF：pip install scipy。"
        ) from e

def corr_negative_test(
    x: List[float],
    y: List[float],
    method: Literal["pearson"] = "pearson",
    nan_policy: Literal["raise", "omit"] = "omit",
) -> Dict[str, float]:
    """
    计算相关系数 + 负相关单侧检验（H1: rho < 0）。

    返回字典字段：
      - n: 样本量
      - r: 相关系数
      - t: t 统计量
      - df: 自由度
      - p_one_sided: 单侧 p 值（负相关方向）
      - p_two_sided: 双侧 p 值（rho != 0）
      - min_alpha_to_claim_negative: “最低错误率阈值”（等于 p_one_sided）
    """
    if len(x) != len(y):
        raise ValueError(f"x,y 长度不一致：len(x)={len(x)}, len(y)={len(y)}")
    if len(x) < 3:
        raise ValueError("样本量至少需要 3 才能做 Pearson 相关的 t 检验。")

    # 清洗 / 处理缺失
    if nan_policy == "omit":
        xy = [(float(a), float(b)) for a, b in zip(x, y) if _is_finite(a) and _is_finite(b)]
        if len(xy) < 3:
            raise ValueError("剔除非有限值后样本量不足 3。")
        x2, y2 = zip(*xy)
        x2, y2 = list(x2), list(y2)
    elif nan_policy == "raise":
        if not all(_is_finite(a) for a in x) or not all(_is_finite(b) for b in y):
            raise ValueError("x 或 y 中存在 NaN/Inf，且 nan_policy='raise'。")
        x2, y2 = [float(a) for a in x], [float(b) for b in y]
    else:
        raise ValueError("nan_policy 只能是 'omit' 或 'raise'。")

    n = len(x2)

    if method != "pearson":
        raise NotImplementedError("当前实现仅支持 pearson。需要 spearman 我也可以给你加。")

    r = _pearson_r(x2, y2)

    # t 检验：t = r * sqrt((n-2)/(1-r^2)), df=n-2
    df = n - 2
    if abs(r) >= 1.0:
        # r=±1 时 1-r^2=0，会爆；这通常意味着完美线性关系
        t_stat = math.copysign(float("inf"), r)
    else:
        t_stat = r * math.sqrt(df / (1.0 - r*r))

    # 单侧：H1: rho < 0 => p = P(T_df <= t_stat)
    p_one = _t_cdf(t_stat, df)

    # 双侧 p：2 * min(CDF(t), 1-CDF(t))
    p_two = 2.0 * min(p_one, 1.0 - p_one)

    return {
        "n": float(n),
        "r": float(r),
        "t": float(t_stat),
        "df": float(df),
        "p_one_sided": float(p_one),
        "p_two_sided": float(p_two),
        "min_alpha_to_claim_negative": float(p_one),
    }

def plot(x, y):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("资源获取类占比")
    plt.ylabel("知识重构类自转移概率")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 读取文件内容
    with open(r"data\result\gold\c.txt", "r", encoding="utf-8") as f:
        json_content = f.read()
    stats = resource_acquire_share_per_data(json_content)
    zy = [data["资源获取类占比"] for data in stats.values()]
    cg = [data["知识重构类占比"] for data in stats.values()]
    cgt = [data["知识重构类自转移概率"] for data in stats.values()]
    pmf = plot_discrete_pmf_of_shares(stats, bin_width=0.05)
    print(corr_negative_test(zy,cgt))
    plot(zy, cgt)
