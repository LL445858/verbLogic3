import pandas as pd
import numpy as np
import os


def load_transition_matrix(file_path: str = None) -> pd.DataFrame:
    """
    加载动词类别转移概率矩阵
    """
    if file_path is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'excel')
        file_path = os.path.join(base_dir, '动词类别转移概率.xlsx')
    
    df = pd.read_excel(file_path, index_col=0, engine='openpyxl')
    return df


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """
    应用温度参数调整概率分布
    P_T(j|i) = P(j|i)^(1/T) / sum_k(P(k|i)^(1/T))
    """
    if temperature <= 0:
        raise ValueError("温度T必须大于0")
    
    # 将概率转换为温度调整后的概率
    adjusted_probs = np.power(probabilities, 1.0 / temperature)
    
    # 归一化
    sum_probs = np.sum(adjusted_probs)
    if sum_probs > 0:
        adjusted_probs = adjusted_probs / sum_probs
    
    return adjusted_probs


def sample_next_state(current_state: str, transition_df: pd.DataFrame, 
                      temperature: float) -> str:
    """
    根据当前状态采样下一个状态
    """
    # 获取当前状态的转移概率
    if current_state not in transition_df.index:
        return None
    
    probs = transition_df.loc[current_state].values
    states = transition_df.columns.tolist()
    
    # 应用温度调整
    adjusted_probs = apply_temperature(probs, temperature)
    
    # 采样下一个状态
    next_state = np.random.choice(states, p=adjusted_probs)
    
    return next_state


def generate_markov_chain(verb: str, category: str, temperature: float, 
                          length: int, transition_file: str = None) -> list:
    """
    生成马尔可夫链序列
    
    参数:
        verb: 动词
        category: 类别
        temperature: 温度参数T
        length: 生成序列的长度n
        transition_file: 转移概率矩阵文件路径（可选）
    
    返回:
        生成的(动词, 类别)元组序列列表
    """
    # 加载转移概率矩阵
    transition_df = load_transition_matrix(transition_file)
    
    # 构建初始状态元组字符串
    initial_state = f"('{verb}', '{category}')"
    
    # 检查初始状态是否存在于转移矩阵中
    if initial_state not in transition_df.index:
        available_states = transition_df.index.tolist()
        print(f"错误: 输入的状态 '{initial_state}' 不存在于转移概率矩阵中。")
        print(f"请重新输入，可用的状态示例: {available_states[:5]}...")
        return None
    
    # 生成序列
    sequence = []
    current_state = initial_state
    
    for i in range(length):
        # 解析当前状态的动词和类别
        # 状态格式为 "('动词', '类别')"
        try:
            current_verb, current_cat = eval(current_state)
        except:
            # 如果解析失败，保持原样
            current_verb, current_cat = current_state, ""
        
        sequence.append((current_verb, current_cat))
        
        # 采样下一个状态
        next_state = sample_next_state(current_state, transition_df, temperature)
        
        if next_state is None:
            print(f"警告: 状态 '{current_state}' 没有可转移的下一状态，提前结束序列生成。")
            break
        
        current_state = next_state
    
    return sequence


def format_sequence(sequence: list) -> str:
    """
    格式化序列为可读字符串
    """
    if not sequence:
        return "序列为空"
    
    parts = []
    for i, (verb, category) in enumerate(sequence, 1):
        # parts.append(f"{i}. ({verb}, {category})")
        parts.append(f"{verb}_{category}")
    
    return " -> ".join(parts)


if __name__ == "__main__":

    verb = "采用"
    category = "资源获取类"
    length = 10
    temperatures = [0.5, 1.0, 2.0]
    
    for t in temperatures:
        print(f"\n温度 T = {t}:")
        result = generate_markov_chain(verb, category, t, length)
        if result:
            print(format_sequence(result))

