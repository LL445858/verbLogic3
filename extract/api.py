#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 
# @Author  : LiXiang
# @File    : api.py
# @Software: PyCharm

from openai import OpenAI


def baichuan4(system_content, user_content):
    client = OpenAI(
        api_key="sk-3652f7d1480a1233cf50f2c289cf40a1",
        base_url="https://api.baichuan-ai.com/v1/",
    )

    completion = client.chat.completions.create(
        model="Baichuan4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
        # print(chunk.choices[0].delta)
    return content


def baichuan3(system_content, user_content):
    client = OpenAI(
        api_key="sk-3652f7d1480a1233cf50f2c289cf40a1",
        base_url="https://api.baichuan-ai.com/v1/",
    )

    completion = client.chat.completions.create(
        model="Baichuan3-Turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
        # print(chunk.choices[0].delta)
    return content


def chatglm4(system_content, user_content):
    client = OpenAI(
        api_key="d2c75957dcec44589c5ae9b1a1c5817c.yPynLdLChtsTNCM0",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    completion = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.9,
        stream=True,
        max_tokens=8192
    )

    # print(completion.choices[0].message)

    content = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content

    return content


def chatglmz1(system_content, user_content):
    client = OpenAI(
        api_key="d2c75957dcec44589c5ae9b1a1c5817c.yPynLdLChtsTNCM0",
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    completion = client.chat.completions.create(
        model="glm-z1-air",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.9,
        stream=True,
        max_tokens=8192
    )

    # print(completion.choices[0].message)

    content = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content

    return content


def doubao_16(system_content, user_content):
    client = OpenAI(
        # 从环境变量中读取您的方舟API Key
        api_key="dafcc4ce-6810-4161-b22e-cb243105ca5a",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="doubao-seed-1.6-thinking-250615",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        # thinking = 'enable',
        max_tokens=8192,
        temperature=0.8,
        stream=True,
        # response_format={'type': 'json_object'}
    )
    reasoning_content = ""
    content = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
            # print(chunk.choices[0].delta.reasoning_content, end="")
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                # print(chunk.choices[0].delta.content, end="")
    return content


def doubao_15(system_content, user_content):
    client = OpenAI(
        # 从环境变量中读取您的方舟API Key
        api_key="dafcc4ce-6810-4161-b22e-cb243105ca5a",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="doubao-1.5-thinking-pro-250415",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        # thinking = 'enable',
        max_tokens=8192,
        temperature=0.8,
        stream=True,
        # response_format={'type': 'json_object'}
    )
    reasoning_content = ""
    content = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
            # print(chunk.choices[0].delta.reasoning_content, end="")
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                # print(chunk.choices[0].delta.content, end="")
    return content


def deepseek_r1(system_content, user_content):
    client = OpenAI(
        api_key="sk-ce23d310eef443b2be123fbe8db61807",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        max_tokens=8192,
        temperature=0.8,
    )
    reasoning_content = ""
    content = ""

    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        else:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    return content


def deepseek_v3(system_content, user_content):
    client = OpenAI(
        api_key="sk-ce23d310eef443b2be123fbe8db61807",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        # stream=True,
        max_tokens=8192,
        temperature=0.8,
    )

    return response.choices[0].message.content


def qwen_plus(system_content, user_content):
    client = OpenAI(
        api_key='sk-8180c63bbca8429f86b3b1fa6caa7ee5',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus-latest",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        extra_body={"enable_thinking": True},
        stream=True,
        max_tokens=8192,
        temperature=0.8
    )

    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = False  # 是否进入回复阶段
    # print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            # print("\nUsage:")
            # print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 只收集思考内容
        # if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        #     if not is_answering:
        #         print(delta.reasoning_content, end="", flush=True)
        #     reasoning_content += delta.reasoning_content

        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # print(delta.content, end="", flush=True)
            answer_content += delta.content

    return answer_content


def qwen3(system_content, user_content):
    client = OpenAI(
        api_key='sk-8180c63bbca8429f86b3b1fa6caa7ee5',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        extra_body={"enable_thinking": True},
        stream=True,
        max_tokens=8192,
        temperature=0.8
    )

    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = False  # 是否进入回复阶段
    # print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        if not chunk.choices:
            # print("\nUsage:")
            # print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 只收集思考内容
        # if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        #     if not is_answering:
        #         print(delta.reasoning_content, end="", flush=True)
        #     reasoning_content += delta.reasoning_content

        # 收到content，开始进行回复
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # print(delta.content, end="", flush=True)
            answer_content += delta.content

    return answer_content
