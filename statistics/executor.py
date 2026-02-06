import json
from collections import Counter
from typing import Dict, List, Tuple
import os

import pandas as pd


def load_json_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {e}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")


def count_executor_verbs(v_file: str, a_file: str) -> Dict[str, int]:
    v_data = load_json_file(v_file)
    a_data = load_json_file(a_file)
    
    executor_verb_counts = Counter()
    
    for data_key in v_data:
        if data_key not in a_data:
            continue
        
        verbs = v_data[data_key]
        attributes = a_data[data_key]
        
        for verb in verbs:
            if verb in attributes and '执行者' in attributes[verb]:
                executor_verb_counts[verb] += 1
    
    return dict(executor_verb_counts)


def count_executor_verbs_by_stage(v_file: str, a_file: str, c_file: str) -> Dict[str, Dict[str, int]]:
    v_data = load_json_file(v_file)
    a_data = load_json_file(a_file)
    c_data = load_json_file(c_file)
    
    stage_verb_counts = {}
    
    for data_key in v_data:
        if data_key not in a_data or data_key not in c_data:
            continue
        
        verbs = v_data[data_key]
        attributes = a_data[data_key]
        categories = c_data[data_key]
        
        for verb in verbs:
            if verb in attributes and '执行者' in attributes[verb] and verb in categories:
                stage = categories[verb]
                
                if stage not in stage_verb_counts:
                    stage_verb_counts[stage] = Counter()
                
                stage_verb_counts[stage][verb] += 1
    
    for stage in stage_verb_counts:
        stage_verb_counts[stage] = dict(stage_verb_counts[stage])
    
    return stage_verb_counts


def format_and_print_results(executor_counts: Dict[str, int], stage_counts: Dict[str, Dict[str, int]]):
    print("\n" + "-" * 80)
    print("出现频次前10的动词:")
    print("-" * 80)
    top_10 = sorted(executor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (verb, count) in enumerate(top_10, 1):
        print(f"  {i:2d}. {verb}: {count}次")
    
    print("\n" + "-" * 80)
    print("各阶段出现频次前10的动词:")
    print("-" * 80)
    for stage in sorted(stage_counts.keys()):
        stage_verbs = stage_counts[stage]
        stage_total = sum(stage_verbs.values())
        stage_unique = len(stage_verbs)
        
        print(f"\n【{stage}】(共{stage_unique}个动词, 总频次{stage_total}次)")
        top_10_stage = sorted(stage_verbs.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (verb, count) in enumerate(top_10_stage, 1):
            print(f"  {i:2d}. {verb}: {count}次")
    
    print("\n" + "=" * 80)
    print("统计完成")
    print("=" * 80)


def generate_executor_excel(v_file: str, a_file: str, c_file: str, output_path: str = None):
    """
    生成执行者统计的Excel表
    第一列：所有包含执行者属性的动词（允许重复）
    第二列：执行者属性值
    第三列：该动词所处类
    """
    v_data = load_json_file(v_file)
    a_data = load_json_file(a_file)
    c_data = load_json_file(c_file)
    
    # 存储每一行的数据
    rows = []
    
    for data_key in v_data:
        if data_key not in a_data or data_key not in c_data:
            continue
        
        verbs = v_data[data_key]
        attributes = a_data[data_key]
        categories = c_data[data_key]
        
        for verb in verbs:
            if verb in attributes and '执行者' in attributes[verb] and verb in categories:
                executor_value = attributes[verb]['执行者']
                category = categories[verb]
                rows.append({
                    '动词': verb,
                    '执行者': executor_value,
                    '阶段类别': category
                })
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=['动词', '执行者', '阶段类别'])
    
    # 设置默认输出路径
    if output_path is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'excel')
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, '执行者统计.xlsx')
    
    # 保存到Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n执行者统计表已生成: {output_path}")
    print(f"共统计 {len(rows)} 条记录")
    
    return output_path


def analyze_stage_executor_category(input_file: str = None, output_file: str = None):
    """
    读取执行者统计_角色标注.xlsx，统计不同阶段类别和执行者类别的数量
    生成二维表格（阶段类别 x 执行者类别）
    """
    # 设置默认输入路径
    if input_file is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'excel')
        input_file = os.path.join(base_dir, '执行者统计_角色标注.xlsx')
    
    # 设置默认输出路径
    if output_file is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'excel')
        output_file = os.path.join(base_dir, '执行者_阶段.xlsx')
    
    # 读取Excel文件
    df = pd.read_excel(input_file, sheet_name='Sheet1', engine='openpyxl')
    
    # 检查必要的列是否存在
    if '阶段类别' not in df.columns or '执行者类别' not in df.columns:
        raise ValueError("Excel文件中缺少'阶段类别'或'执行者类别'列")
    
    # 创建二维统计表（透视表）
    pivot_table = pd.crosstab(df['阶段类别'], df['执行者类别'], margins=True, margins_name='总计')
    
    # 保存到Excel
    pivot_table.to_excel(output_file, engine='openpyxl')
    
    print(f"\n阶段-执行者类别统计表已生成: {output_file}")
    print(f"\n统计结果预览:")
    print(pivot_table)
    
    return pivot_table


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'result', 'gold')
    
    v_file = os.path.join(base_dir, 'v.txt')
    a_file = os.path.join(base_dir, 'a.txt')
    c_file = os.path.join(base_dir, 'c.txt')
    
    try:
        executor_counts = count_executor_verbs(v_file, a_file)
        stage_counts = count_executor_verbs_by_stage(v_file, a_file, c_file)
        format_and_print_results(executor_counts, stage_counts)
        
        # 生成执行者统计Excel表
        generate_executor_excel(v_file, a_file, c_file)
        
        # 分析阶段类别和执行者类别的二维统计
        analyze_stage_executor_category()
    except Exception as e:
        print(f"错误: {e}")

