#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 
# @Author  : YinLuLu
# @File    : llm.py
# @Software: PyCharm

import json
import time
from api import *

model = {'deepseek_r1': deepseek_r1, 'deepseek_v32': deepseek_v32,
             'qwen_plus': qwen_plus, 'qwen3': qwen3,
             'doubao_16': doubao_16, 'doubao_18': doubao_18,
             'glm47': chatglm_47, 'glm47f': chatglm_47_flash,
             "baichuan4": baichuan4, "baichuan3": baichuan3}

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
    system_content = '''
# 指令
您是科技文本分析专家，擅长识别和提取知识创造过程中的关键动词。请精读输入的知识创造文本，理解其所描述的知识创造过程，并提取知识创造过程事件中的核心动词群。
提取要求：
1）仅提取与知识创造密切相关的最能概括所在句流程的核心动词，确保动词序列覆盖整个知识创造过程的时间轴。忽略无关动词、不具知识创造主题意义、或者背景介绍里的的动词（如"为了解决..."中的"解决"）；
2）对于虚动词+名词的组合，需要提取能当作动词的实义名词作为动词（如"进行了报告/实验/推导"，提取的应是"报告/实验/推导"，而不是"进行"）；
3）所有动词**必须**统一为**二个汉字**的动词原型，若为单字动词则根据语义进行补全（如“看”→“看见”，“帮”→“帮助”），允许出现重复动词（如前后多次出现的"实验"）,允许从四字成语中总结隐含动词（"绞尽脑汁"->"思考"）；
4）请将抽取到的全部动词，按原文出现顺序排列，仿照以下JSON 格式结构，**以"动词"为键，动词序列为值**，输出格式为{"动词": ["verb1", "verb2", "verb3", ......]}的最终结果，即使没有动词也返回值为空列表的词典，不要添加除json以外的任何信息:

# 示例
输入示例：
为了解决如何控制和利用原子弹能量这一高难度的问题，于敏首先分析了原子弹爆炸所释放的各种能量形式，比较了它们的特性与在总能量中所占的比例，明确了一种比较容易控制、驾驭的能量形式。然后，他想出了一个减少这种能量损失、提高其利用率的精巧的结构，估计了有多少能量可以被利用，如何利用其有利因素，控制、避免其不利因素，又使于敏绞尽脑汁。10月下旬，于敏向在上海出差的全体同志作了系列的“氢弹原理设想”的学术报告。于敏从辐射流体力学、中子扩散和热核反应动力学的基本方程出发，结合以前的理论探索和最新的计算结果，时而进行严密的推导，时而进行量纲分析和粗估，列举了实现热核材料自持燃烧的各种可能途径，比较了它们的优劣利弊，详尽地论证了实现热核材料自持燃烧的内因和必要条件，提出了两级氢弹的原理和构形的设想。但新原理刚刚提出有一大堆问题有待研究论证，于是为完成这项新任务立即成立了一个新原理的研究小组。新原理小组在于敏的领导下又做了许多工作。于敏一面对氢弹设计中的问题进行物理分解，一面由新原理小组组织数值模拟计算。于敏在对计算结果进行分析后，又向大家报告，经过热烈讨论产生许多新的想法，根据这些新想法立即开辟新课题。经过这段时间的系统工作，发现了一批重要的物理现象和规律。这些规律对以后氢弹的物理设计和核试验诊断都有重要指导意义。这样，逐步形成了从氢弹初级到能量传输到氢弹次级的原理和构形基本完整的物理方案。
输出示例：
{"动词": ["分析","比较","明确","想出","估计","思考","报告","推导","分析","粗估","列举","比较","论证","提出","成立","领导","分解","组织","分析","报告","讨论","产生","开辟","发现","形成"]}
'''
    user_content = "接下面是需要你进行动词实体抽取的文本:\n\""

    with open(data_path + f"result\\verbs\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(texts)
        i = 0
        for text in texts:
            time.sleep(3)
            with open(data_path + f"corpus\\D{text}.txt", "r", encoding="utf-8") as f1:
                t = f1.read()
            if t:
                try:
                    response = model[model_name](system_content, user_content + t + "\"")
                except Exception as e:
                    print(f"{type(e)}: {e}")
                    response = "{\"动词\":[\"结果错误\"]}"
            else:
                response = "{\"动词\":[\"文本空白\"]}"
            print(model_name, f"\tdata{text}: ", response)
            response = extract_last_big_brace(response)
            # response = response.replace("{", "[")
            # response = response.replace("}", "]")
            response = response.replace('\n', '')
            response = response.replace('\t', '')
            response = response.replace(' ', '')
            if response:
                try:
                    response = eval(response).get("动词", [])
                    f.write(f"\"data{text}\":{response}")
                except Exception as e:
                    print(f"{type(e)}: {e}")
                    f.write(f"\"data{text}\":[\"解析错误\"]")
            else:
                f.write(f"\"data{text}\":[\"结果空白\"]")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


def attributes_extract(texts, model_name):
    system_content = '''
# 指令
您是科技文本分析专家，擅长对文本中关键动词的属性实体进行抽取。请根据输入的知识创造文本，对提供的动词列表中的每一个词，进行属性实体抽取
抽取要求：
1）抽取的属性实体仅可以在以下范围内
主体类：[执行者,合作者,推动者,反对者,施事对象]
客体类：[受事人,受事对象]
背景类：[历史背景,社会背景,学术背景,开始时间,结束时间,持续时间,发生地点,学术载体]	
逻辑类：[工具,理论,方法,资料,结论,影响,数据,状况,成果,问题]
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
  "引起": {"施事对象":"《钼单晶生长过程研究》的论文","影响": "多方关注"},
  "指明": {"施事对象":"对于钼单晶生长的研究","影响": "工业发展方向"}
}
'''
    user_content = f"接下面是需要你进行动词属性抽取的文本,"

    with open(data_path + f"result\\gold\\v.txt", "r", encoding="utf-8") as f:
        verbs = json.load(f)

    with open(data_path + f"result\\attribute\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(texts)
        i = 0
        for text in texts:
            with open(data_path + f"corpus\\D{text}.txt", "r", encoding="utf-8") as f1:
                t = f1.read()
            verbs_list = verbs[f"data{text}"]
            if t:
                try:
                    response = model[model_name](system_content, user_content + f"其中的动词列表为{verbs_list}:\n\"" + t + "\"")
                except Exception as e:
                    print(f"{type(e)}: {e}")
                    response = "{\"结果错误\":{}}"
            else:
                response = "{\"文本空白\":{}}"
            response = response.replace('\n', '')
            response = response.replace('\t', '')
            response = response.replace(' ', '')
            response = response.replace('},', '},\n')
            response = extract_last_big_brace(response)
            print(model_name, f": data{text}:\n", response)
            if response:
                f.write(f"\"data{text}\":\n{response}")
            else:
                f.write(f"\"data{text}\":{{\"结果空白\":{{}}}}")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


# 动词的属性抽取
def category_extract(texts, model_name):
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
2）提供的动词列表中的动词为原文出现顺序，可能含有重复动词、原文单字动词补全后的结果（如"担任"对应原文的"任"）以及概括出的动词（"绞尽脑汁"->"思考"），注意按实际出现顺序进行依次抽取，不要遗漏每一个词
3）请将抽取到的全部属性，仿照以下JSON 格式结构按原有的动词顺利逐个输出最终结果，不要添加除json以外的任何信息:
{"动词1":"类别1", "动词2"："类别2", ...}

## 示例
接下面是需要你进行动词类别判断的文本，其中的动词列表为["开展","利用","保持","获得","制得","总结","改良","使用","控制","采用","克服","运用","赞助","发表","引起","指明"]：
许多学者在钼单晶研究领域开展了有益的尝试，如Andrade曾利用新型的炉子结构(细导线被包裹在真空的熔融石英管中，通电流使其保持一个高且均匀的温度)获得更快的高熔点金属α铁单晶的生长速度。Tsien和Chow也用同样的方法成功制得了0.25mm直径的钼单晶。总结以上经验，陈能宽在该结构基本原理的基础上进行了改良，使用了缠绕镍铬合金的附加炉子，这样电子加热区就是一个有1/2英寸宽的圆筒形区域，可以通过所连接的发动机电子控制器控制速度从每小时0.5到15英寸，来进行升高或降低的调节。这种改良后的方法采用极细的金属丝克服了Andrade方法中的缺陷，就能够将这种方法运用到大的钼单晶或者其他任何难熔金属的单晶生长研究中去。1951年，按照合同，由海军研究办公室赞助，陈能宽作为约翰斯·霍普斯金大学机械工程系物理冶金的研究员跟R.Maddin教授和R.B.Pond教授在AIME金属学报上共同发表了《钼单晶生长过程研究》的论文，引起了多方关注。对于钼单晶生长的研究，在很大程度上为工业发展指明了方向。
输出：
{"开展":"知识规划类", "利用":"资源获取类", "保持":"知识重构类", "获得":"知识重构类", "制得":"成果发布类", "总结":"知识整合类", "改良":"知识重构类", "使用":"资源获取类", "控制":"知识重构类", "采用":"资源获取类", "克服":"知识重构类", "运用":"资源获取类", "赞助":"资源获取类", "发表":"成果发布类", "引起":"成果影响类", "指明":"成果影响类"}
'''
    user_content = f"接下面是需要你进行动词类别判断的文本,"

    with open(data_path + f"result\\gold\\v.txt", "r", encoding="utf-8") as f:
        verbs = json.load(f)


    with open(data_path + f"result\\category\\{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("{\n")
        n = len(texts)
        i = 0
        for text in texts:
            with open(data_path + f"corpus\\D{text}.txt", "r", encoding="utf-8") as f1:
                t = f1.read()
            verbs_list = verbs[f"data{text}"]
            if t:
                try:
                    response = model[model_name](system_content, user_content + f"其中的动词列表为{verbs_list}:\n\"" + t + "\"")
                except:
                    response = "{\"结果错误\":\"\"}"
            else:
                response = "{\"文本空白\":\"\"}"
            response = response.replace('\n', '')
            response = response.replace('\t', '')
            response = response.replace(' ', '')
            response = response.replace('},', '},\n')
            print(model_name, f": \"data{text}\":\n", response.encode('gbk', 'ignore').decode('gbk'))
            response = extract_last_big_brace(response)
            if response:
                f.write(f"\"data{text}\":\n{response}")
            else:
                f.write(f"\"data{text}\":{{\"空白\":\"\"}}")
            i += 1
            if i < n:
                f.write(',\n')
        f.write("\n}")


if __name__ == '__main__':
    model_list = ['qwen_plus', 'qwen3', 'deepseek_r1', 'deepseek_v32', "baichuan4", "baichuan3", 'glm47f','glm47', ]
    data_path = f"Y:\\Project\\Python\\VerbLogic\\data\\"
    for model_name in model_list:
        verbs_extract([i for i in range(1, 42)], model_name)


