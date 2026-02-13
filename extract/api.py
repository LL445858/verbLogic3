#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 
# @Author  : YinLuLu
# @File    : api.py
# @Software: PyCharm

import toml
from openai import OpenAI
from volcenginesdkarkruntime import Ark
from zai import ZhipuAiClient
from dashscope import Generation
import dashscope

api_config = toml.load("ApiConfig.toml")
temperature = 0.6
max_tokens = 4096


def baichuan4(system_content, user_content):
    client = OpenAI(
        api_key=api_config["BaichuanKey"],
        base_url=api_config["BaichuanUrl"],
    )

    completion = client.chat.completions.create(
        model="Baichuan4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
    return content


def baichuan3(system_content, user_content):
    client = OpenAI(
        api_key=api_config["BaichuanKey"],
        base_url=api_config["BaichuanUrl"],
    )

    completion = client.chat.completions.create(
        model="Baichuan3-Turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    content = ""
    for chunk in completion:
        content += chunk.choices[0].delta.content
    return content


def chatglm_47(system_content, user_content):
    client = ZhipuAiClient(api_key=api_config["ChatGLMKey"])

    response = client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        thinking={
            "type": "enabled",
        },
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content


def chatglm_47_flash(system_content, user_content):
    client = ZhipuAiClient(api_key=api_config["ChatGLMKey"])

    response = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        thinking={
            "type": "enabled",
        },
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content


def doubao_16(system_content, user_content):
    client = Ark(
        api_key=api_config["DoubaoKey"],
    )

    completion = client.chat.completions.create(
        model="doubao-seed-1-6-251015",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        thinking={"type": "enabled"},
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content



def doubao_18(system_content, user_content):
    client = Ark(
        api_key=api_config["DoubaoKey"],
    )

    completion = client.chat.completions.create(
        model="doubao-seed-1-8-251228",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        thinking={"type": "enabled"},
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


def deepseek_r1(system_content, user_content):
    client = Ark(
        api_key=api_config["DoubaoKey"],
    )

    completion = client.chat.completions.create(
        model="deepseek-r1-250528",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        thinking={"type": "enabled"},
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


def deepseek_v32(system_content, user_content):
    client = Ark(
        api_key=api_config["DoubaoKey"],
    )

    completion = client.chat.completions.create(
        model="deepseek-v3-2-251201",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        thinking={"type": "enabled"},
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


# def deepseek_r1(system_content, user_content):
#     client = OpenAI(
#         api_key=api_config["DeepseekKey"],
#         base_url=api_config["DeepseekUrl"],
#     )
#
#     response = client.chat.completions.create(
#         model="deepseek-reasoner",
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_content},
#         ],
#         stream=True,
#         max_tokens=max_tokens,
#         temperature=temperature,
#     )
#     reasoning_content = ""
#     content = ""
#
#     for chunk in response:
#         if chunk.choices[0].delta.reasoning_content:
#             reasoning_content += chunk.choices[0].delta.reasoning_content
#         else:
#             if chunk.choices[0].delta.content:
#                 content += chunk.choices[0].delta.content
#     return content
#
#
# def deepseek_v32(system_content, user_content):
#     client = OpenAI(
#         api_key=api_config["DeepseekKey"],
#         base_url=api_config["DeepseekUrl"],
#     )
#
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_content},
#         ],
#         max_tokens=max_tokens,
#         temperature=temperature,
#     )
#
#     return response.choices[0].message.content


def qwen_plus(system_content, user_content):
    response = Generation.call(
        api_key=api_config["QwenKey"],
        model="qwen-plus-2025-12-01",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=True,
        result_format="message",
        # response_format={"type": "json_object"}
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"{response.status_code}; {response.code}; {response.message}"



def qwen3(system_content, user_content):
    response = Generation.call(
        api_key=api_config["QwenKey"],
        model="qwen3-max-2026-01-23",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=True,
        result_format="message",
        # response_format={"type": "json_object"}
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        return f"{response.status_code}; {response.code}; {response.message}"


if __name__ == "__main__":
    result = qwen_plus("你是问答助手", "1+1等于几")
    print(result)