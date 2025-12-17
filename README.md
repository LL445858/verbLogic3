## VerbLogic

### data 文本文件
#### analysis
- 同义词典 sys_dict.txt
- 全部同义词归纳 verb_sys.txt
- 删除出现一次的同义词归纳 verb_del1.txt
- 删除出现两次的同义词归纳 verb_del2.txt
- 人工调整后的同义词归纳 verb_lulu.txt
- 动词转移概率记录 verb_move.txt

#### corpos
- dataX.txt 语料

#### excel
- XXXX.xlsx 数据表格

### fig
- XXX.png/emf 数据图

### result
- verb  各个大模型动词抽取结果
- attribute  各个大模型属性抽取结果
- category  各个大模型分类结果
- gold 人工处理后的黄金语料

### extract 大模型抽取代码
- api.py 各个模型接口
- llm.py 提示词及接口调用
- evaluation.py 抽取结果评价计算

### plot 数据可视化

### statistics 数据统计