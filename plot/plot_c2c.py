import pandas as pd  # 用来读取csv文件
import matplotlib as mpl
import matplotlib.pyplot as plt  # mpl_chord_diagram需要用到matplotlib的包
from mpl_chord_diagram import chord_diagram

mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据，从csv文件读取的数据是一个DataFrame类型
data1 = pd.read_csv(r"Y:\Project\PythonProject\VerbLogic\data\excel\阶段转移概率_和弦.csv")

# 将DataFrame类型转为一个2D的ndarray类型。
data2 = data1.to_numpy()

# 设定七种不同地类的显示颜色
# colors = ['#F0E68C', '#008000', '#006400', '#0000FF', '#BDB76B', '#5F9EA0', '#FFB6C1']
colors = ['#8989BD', '#B7A8CF', '#E7BDC7', '#FECEA0', '#F0A586', '#F0E68C', '#FFD700']

# 设置文字标注
names = ["知识规划类", "知识整合类", "资源获取类", "知识协作类", "知识重构类", "成果发布类", "成果影响类"]

# 设置字体
font = {
    'family': 'Microsoft YaHei',  # 这里有一个巨大的坑， ubuntu上要额外安装 Times New Roman 字体。
    'style': 'normal',
    'weight': 'normal',
    'color': 'black',
    'size': 25
}

# 使用chord_diagram函数绘图
chord_diagram(data2,  # 输入数据
              names,  # 用来显示文字标注
              pad=5,  # 每一段圆弧之间的距离
              directed=False,  # 用箭头显示流量方向,方便看出流量是从哪里流向哪里
              colors=colors,  # 每一段圆弧及其流量显示的颜色
              fontsize=8,  # 文字大小
              rotate_names=[True, True, True, True, True, True, True],  # 有些文字注释是倒转的，需要转90度
              )

# 保存图片到当前目录下的graph文件夹
plt.savefig('test_data1' + '.png',  # 保存的文件名，保存在当前目录的graph文件夹下
            dpi=500,  # 输出分辨率
            bbox_inches='tight'  # 把图片所有内容都保存，如果没有这个参数，图片的上边和下边会没有保存
            )
