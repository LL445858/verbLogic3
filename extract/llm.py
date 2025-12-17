#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 
# @Author  : YinLuLu
# @File    : llm.py
# @Software: PyCharm

import json
import time
from api import *


def extract_last_big_brace(s):
    stack = []
    pairs = []
    for i, c in enumerate(s):
        if c == '{':
            stack.append(i)
        elif c == '}':
            if stack:
                left = stack.pop()
                if not stack:
                    pairs.append((left, i))
    if not pairs:
        return s
    last_left, last_right = pairs[-1]
    return s[last_left:last_right + 1]


def verbs_extract(texts, model_name):
    model = {'deepseek_r1': deepseek_r1, 'deepseek_v3': deepseek_v3,
             'qwen_plus': qwen_plus, 'qwen3': qwen3,
             'doubao_16': doubao_16, 'doubao_15': doubao_15,
             'glm4': chatglm4, 'glmz': chatglmz1,
             "baichuan4": baichuan4, "baichuan3": baichuan3}
    result = dict()
    system_content = '''
# 指令
您是科技文本分析专家，擅长识别和提取知识创造过程中的关键动词。请精读输入的知识创造文本，理解其所描述的知识创造过程，并提取知识创造过程事件中的核心动词群。
提取要求：
1）仅提取与知识创造密切相关的动词，确保动词序列覆盖整个过程的时间轴。忽略无关动词或不具知识创造主题意义的动词。
2）所有动词统一为二字动词原型，允许语义补全单字动词（如“看”→“看见”，“帮”→“帮助”）并提取，允许出现重复动词（如多次出现的实验）。
3）请将抽取到的全部动词，按原文出现顺序排列，仿照以下JSON 格式结构输出最终结果，不要添加除json以外的任何信息:{"verb1", "verb2", "verb3", ......}

# 示例
输入示例：
许多学者在钼单晶研究领域开展了有益的尝试，如Andrade曾利用新型的炉子结构(细导线被包裹在真空的熔融石英管中，通电流使其保持一个高且均匀的温度)获得更快的高熔点金属α铁单晶的生长速度。Tsien和Chow也用同样的方法成功制得了0.25mm直径的钼单晶。总结以上经验，陈能宽在该结构基本原理的基础上进行了改良，使用了缠绕镍铬合金的附加炉子，这样电子加热区就是一个有1/2英寸宽的圆筒形区域，可以通过所连接的发动机电子控制器控制速度从每小时0.5到15英寸，来进行升高或降低的调节。这种改良后的方法采用极细的金属丝克服了Andrade方法中的缺陷，就能够将这种方法运用到大的钼单晶或者其他任何难熔金属的单晶生长研究中去。1951年，按照合同，由海军研究办公室赞助，陈能宽作为约翰斯·霍普斯金大学机械工程系物理冶金的研究员跟R.Maddin教授和R.B.Pond教授在AIME金属学报上共同发表了《钼单晶生长过程研究》的论文，引起了多方关注。对于钼单晶生长的研究，在很大程度上对为工业发展指明了方向。
输出示例：
{"开展", "利用", "保持", "获得", "制得", "总结", "改良", "使用", "控制", "采用", "克服", "运用", "赞助", "发表", "引起", "指明"}
'''
    user_content = "接下面是需要你进行动词实体抽取的文本:\n\""
    for text in texts:
        with open(data_path + f"text\\data{text}.txt", "r", encoding="utf-8") as f:
            response = str(model[model_name](system_content, user_content + f.read() + "\""))
        response = extract_last_big_brace(response)
        response = response.replace("{", "[")
        response = response.replace("}", "]")
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        response = response.replace(' ', '')
        result[f"data{text}"] = response
        print(model_name, f"\tdata{text}: ", response)

    with open(data_path + f"extract\\verbs\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(result)
        i = 0
        for k, v in result.items():
            if v:
                f.write(f"\"{k}\":{v}")
            else:
                f.write(f"\"{k}\":[]")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


def attributes_extract(texts, model_name):
    model = {'deepseek_r1': deepseek_r1, 'deepseek_v3': deepseek_v3,
             'qwen_plus': qwen_plus, 'qwen3': qwen3,
             'doubao_16': doubao_16, 'doubao_15': doubao_15,
             'glm4': chatglm4, 'glmz': chatglmz1,
             "baichuan4": baichuan4, "baichuan3": baichuan3}
    result = dict()
    system_content = '''
# 指令
您是科技文本分析专家，擅长对文本中关键动词的属性实体进行抽取。请根据输入的知识创造文本，对提供的动词列表中的每一个词，进行属性实体抽取
抽取要求：
1）抽取的属性实体仅可以在以下范围内
主体类：[执行者,合作者,推动者,反对者,施事对象]
客体类：[受事人,受事对象]
背景类：[历史背景,社会背景,学术背景,开始时间,结束时间,持续时间,发生地点,学术载体]	
逻辑类：[工具,理论,方法,资料,结论,影响,数据,状况,成果]
2）提供的动词列表中的动词为原文出现顺序，可能含有重复动词和原文单字动词补全后的结果（如"担任"对应原文的"任"），注意按出现顺序进行依次抽取
3）所给的每一个动词都应该根据全文内容尽量抽取所有可能存在的属性实体，将代词查找替换为原始的属性实体名，未找到的属性不需要显示
4）请将抽取到的全部属性，仿照以下JSON 格式结构输出最终结果，不要添加除json以外的任何信息:
{"动词1":{"属性1"："属性实体1",...}, "动词2":...}

# 示例
输入示例：
文本："许多学者在钼单晶研究领域开展了有益的尝试，如Andrade曾利用新型的炉子结构(细导线被包裹在真空的熔融石英管中，通电流使其保持一个高且均匀的温度)获得更快的高熔点金属α铁单晶的生长速度。Tsien和Chow也用同样的方法成功制得了0.25mm直径的钼单晶。总结以上经验，陈能宽在该结构基本原理的基础上进行了改良，使用了缠绕镍铬合金的附加炉子，这样电子加热区就是一个有1/2英寸宽的圆筒形区域，可以通过所连接的发动机电子控制器控制速度从每小时0.5到15英寸，来进行升高或降低的调节。这种改良后的方法采用极细的金属丝克服了Andrade方法中的缺陷，就能够将这种方法运用到大的钼单晶或者其他任何难熔金属的单晶生长研究中去。1951年，按照合同，由海军研究办公室赞助，陈能宽作为约翰斯·霍普斯金大学机械工程系物理冶金的研究员跟R.Maddin教授和R.B.Pond教授在AIME金属学报上共同发表了《钼单晶生长过程研究》的论文，引起了多方关注。对于钼单晶生长的研究，在很大程度上为工业发展指明了方向。"
动词列表：["开展","利用","保持","获得","制得","总结","改良","使用","控制","采用","克服","运用","赞助","发表","引起","指明"]
输出示例：
{
  "开展": {"执行者": "许多学者","受事对象": "有益的尝试","学术背景": "钼单晶研究领域"},
  "利用": {"执行者": "Andrade" , "工具": "新型的炉子结构"},
  "保持": {"施事对象": "细导线","状况": "高且均匀的温度","方法": "通电流"},
  "获得": {"执行者": "Andrade", "成果": "更快的高熔点金属α铁单晶的生长速度"},
  "制得": {"执行者": "Tsien和Chow","方法": "同样的方法","成果": "0.25mm直径的钼单晶"},
  "总结": {"执行者": "陈能宽","资料": "以上经验"},
  "改良": {"执行者": "陈能宽","理论": "该结构基本原理","成果": "改良后的方法"},
  "使用": {"执行者": "陈能宽","工具": "缠绕镍铬合金的附加炉子"},
  "控制": {"工具": "发动机电子控制器","状况": "速度从每小时0.5到15英寸"},
  "采用": {"执行者": "陈能宽","工具": "极细的金属丝"},
  "克服": {"施事对象": "改良后的方法","受事对象": "Andrade方法中的缺陷"},
  "运用": {"方法": "这种方法","受事对象": "大的钼单晶或者其他任何难熔金属的单晶生长研究"},
  "赞助": {"执行者": "海军研究办公室","受事人": "陈能宽","开始时间": "1951年"},
  "发表": {"执行者": "陈能宽","合作者": "R.Maddin教授和R.B.Pond教授","成果": "《钼单晶生长过程研究》的论文","学术载体": "AIME金属学报","开始时间": "1951年"},
  "引起": {"施事对象":《钼单晶生长过程研究》的论文","影响": "多方关注"},
  "指明": {"施事对象":"对于钼单晶生长的研究","影响": "工业发展方向"}
} "引起": {"影响": "多方关注"},
  "指明": {"影响": "工业发展方向"}
}
'''
    user_content = f"接下面是需要你进行动词属性抽取的文本,"

    with open(data_path + f"extract\\verbs\\gold.txt", "r", encoding="utf-8") as f:
        verbs = json.load(f)

    for text in texts:
        verbs_list = verbs[f"data{text}"]
        with open(data_path + f"text\\data{text}.txt", "r", encoding="utf-8") as f:
            response = model[model_name](system_content,
                                         user_content + f"其中的动词列表为{verbs_list}:\n\"" + f.read() + "\"")
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        response = response.replace(' ', '')
        response = response.replace('},', '},\n')
        print(model_name, f": data{text}:\n", response)
        result[f"data{text}"] = extract_last_big_brace(response)

    with open(data_path + f"extract\\attr\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(result)
        i = 0
        for k, v in result.items():
            if v:
                f.write(f"\"{k}\":\n{v}")
            else:
                f.write(f"\"{k}\":{{}}")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


# 动词的属性抽取
def category_extract(texts, model_name):
    model = {'deepseek_r1': deepseek_r1, 'deepseek_v3': deepseek_v3,
             'qwen_plus': qwen_plus, 'qwen3': qwen3,
             'doubao_16': doubao_16, 'doubao_15': doubao_15,
             'glm4': chatglm4, 'glmz': chatglmz1,
             "baichuan4": baichuan4, "baichuan3": baichuan3}
    result = dict()
    system_content = '''
# 指令
您是科技文本分析专家，擅长识别和提取知识创造过程中的关键动词的所处阶段类别。请根据输入的知识创造文本，对提供的动词列表中的每一个词，结合其语境进行类别判断，同样的动词可能因为语境不同而分类不同
判断要求：
1）判断的类别仅可以在以下范围内
    准备阶段：
    a. 知识规划类:明确创新目标，设计实施路径，预判风险与资源需求
    b. 知识整合类:系统搜索、学习和整理已有的相关领域背景知识、核心技术、历史研究数据等
    研究阶段：
    c. 资源获取类:系统获取、调配、借助所需理论依据、实践经验、技术方法、数据及物质设备等关键资源
    d. 知识协作类:通过跨学科/跨领域合作，促进知识共享、观点碰撞与协同创造
    e. 知识重构类:深度研究、剖析、解构、测试现有及所提的知识体系或方法，重新组合或突破局限以形成新认知或解决方案
    实现阶段：
    f. 成果发布类:通过学术/行业渠道（如论文、专利、产品、报告等）正式产出并传播创新知识成果
    g. 成果影响类:评估并阐述成果在学术理论、技术实践、产业应用或社会层面的价值与影响，以及所获得的荣誉
2）提供的动词列表中的动词为原文出现顺序，可能含有重复动词和原文单字动词补全后的结果（如"担任"对应原文的"任"），注意按实际出现顺序进行依次抽取，不要遗漏每一个词
3）请将抽取到的全部属性，仿照以下JSON 格式结构按原有的动词顺利逐个输出最终结果，不要添加除json以外的任何信息:
{"动词1":"类别1", "动词2"："类别2", ...}

## 示例
接下面是需要你进行动词类别判断的文本，其中的动词列表为["开展","利用","保持","获得","制得","总结","改良","使用","控制","采用","克服","运用","赞助","发表","引起","指明"]：
许多学者在钼单晶研究领域开展了有益的尝试，如Andrade曾利用新型的炉子结构(细导线被包裹在真空的熔融石英管中，通电流使其保持一个高且均匀的温度)获得更快的高熔点金属α铁单晶的生长速度。Tsien和Chow也用同样的方法成功制得了0.25mm直径的钼单晶。总结以上经验，陈能宽在该结构基本原理的基础上进行了改良，使用了缠绕镍铬合金的附加炉子，这样电子加热区就是一个有1/2英寸宽的圆筒形区域，可以通过所连接的发动机电子控制器控制速度从每小时0.5到15英寸，来进行升高或降低的调节。这种改良后的方法采用极细的金属丝克服了Andrade方法中的缺陷，就能够将这种方法运用到大的钼单晶或者其他任何难熔金属的单晶生长研究中去。1951年，按照合同，由海军研究办公室赞助，陈能宽作为约翰斯·霍普斯金大学机械工程系物理冶金的研究员跟R.Maddin教授和R.B.Pond教授在AIME金属学报上共同发表了《钼单晶生长过程研究》的论文，引起了多方关注。对于钼单晶生长的研究，在很大程度上为工业发展指明了方向。
输出：
{"开展":"知识规划类", "利用":"资源获取类", "保持":"知识重构类", "获得":"知识重构类", "制得":"成果发布类", "总结":"知识整合类", "改良":"知识重构类", "使用":"资源获取类", "控制":"知识重构类", "采用":"资源获取类", "克服":"知识重构类", "运用":"资源获取类", "赞助":"资源获取类", "发表":"成果发布类", "引起":"成果影响类", "指明":"成果影响类"}
'''
    user_content = f"接下面是需要你进行动词类别判断的文本,"

    with open(data_path + f"extract\\verbs\\gold.txt", "r", encoding="utf-8") as f:
        verbs = json.load(f)

    for text in texts:
        verbs_list = verbs[f"data{text}"]
        with open(data_path + f"text\\data{text}.txt", "r", encoding="utf-8") as f:
            response = model[model_name](system_content, user_content + f"其中的动词列表为{verbs_list}:\n\"" + f.read() + "\"")
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        response = response.replace(' ', '')
        response = response.replace('},', '},\n')
        print(model_name, f": \"data{text}\":\n", response.encode('gbk', 'ignore').decode('gbk'))
        result[f"data{text}"] = extract_last_big_brace(response)
        time.sleep(3)

    with open(data_path + f"extract\\category\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(result)
        i = 0
        for k, v in result.items():
            if v:
                f.write(f"\"{k}\":\n{v}")
            else:
                f.write(f"\"{k}\":{{}}")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


if __name__ == '__main__':
    data_path = f"Y:\\Project\\PythonProject\\VerbLogic\\data\\"
    # model_list = ['deepseek_r1', 'deepseek_v3', 'qwen_plus', 'qwen3', 'doubao_16', 'doubao_15', 'glm4', 'glmz', "baichuan4", "baichuan3"]
    model_list = ["deepseek_r1", "deepseek_v3"]
    for model in model_list:
        category_extract([i for i in range(1, 42)], model)

