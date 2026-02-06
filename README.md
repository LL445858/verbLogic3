## 动词逻辑视角下科学家知识创新的叙事文本语义挖掘与行为分析研究
> 南京工业大学 经济与管理学院

### data：文本文件
#### data/analysis
- 同义词典 sys_dict.txt
- 全部同义词归纳 verb_sys.txt
- 删除出现一次的同义词归纳 verb_del1.txt
- 删除出现两次的同义词归纳 verb_del2.txt
- 人工调整后的同义词归纳 verb_lulu.txt
- 动词转移概率记录 verb_move.txt

#### data/corpos
- dataX.txt 语料

#### data/excel
- XXXX.xlsx 数据表格

### data/fig
- XXX.png/emf 数据图

### data/result
- verb  各个大模型动词抽取结果
- attribute  各个大模型属性抽取结果
- category  各个大模型分类结果
- gold 人工处理后的黄金语料
- - a.txt 人工处理后的属性结果
- - c.txt 人工处理后的分类结果
- - v.txt 人工处理后的动词抽取结果

### extract 大模型抽取代码
- api.py 各个模型接口
- llm.py 提示词及接口调用
- evaluation.py 抽取结果评价计算

### plot 数据可视化
- plot_attr.py 属性统计柱状图
- plot_verb.py 动词统计柱状图
- plot_cat.py 分类统计柱状图
- plot_c2c.py 分类转移和弦图
- plot_c2c——net.py 动词转移和弦图

### statistics 数据统计
- cluster.py 层级/kmeans聚类统计
- count.py 动词、属性、分类统计