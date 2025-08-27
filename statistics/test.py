#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/19 
# @Author  : LiXiang
# @File    : test.py
# @Software: PyCharm

import json

for i in range(1, 42):
    with open(f'Y:\\Project\\PythonProject\\VerbLogic\\data\\corpus\\data{i}.txt', 'r', encoding='utf-8') as f:
        c_content = f.read()
    if '研制' in c_content and '比较' in c_content:
        print(i)