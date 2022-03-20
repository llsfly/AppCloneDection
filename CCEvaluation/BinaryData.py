
class BinaryData:
    def __init__(self, src_key, tgt_key, default, pos_tag, pos_prob, neg_tag, neg_prob, true_tag):
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.pos_tag = pos_tag
        self.pos_prob = float(pos_prob)
        self.neg_tag = neg_tag
        self.neg_prob = float(neg_prob)
        self.true_tag = true_tag
        self.default = default

    def condTrue(self, pos_threshold):
        pred = self.pos_tag if self.pos_prob >= pos_threshold else self.neg_tag
        return pred == self.true_tag

    def __str__(self):
        items = [self.src_key, self.tgt_key, self.default, \
                 self.pos_tag, str(self.pos_prob), \
                 self.neg_tag, str(self.neg_prob), self.true_tag]
        strout = ' '.join(items)
        return strout


def parse(line):
    src_key, tgt_key, default, pos_tag, pos_prob, neg_tag, neg_prob, true_tag = line.split()
    data = BinaryData(src_key, tgt_key, default, pos_tag, pos_prob, neg_tag, neg_prob, true_tag)
    return data


import numpy as np
import matplotlib.pyplot as plt
import platform
import random

def draw_figure_and_save(xs, ys, labels, save_figure):
    '''
    画曲线
    :param xs: 坐标点的x轴坐标，list对象，xs中每个元素表示的是一条曲线的所有点的x轴坐标
    :param ys: 坐标点的y轴坐标，同xs
    :param labels: 标签，str的list
    :param save_figure: 图片保存路径
    :return:
    '''
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体，则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    # TODO: Make sure enough colors......

    number_of_colors = len(xs)
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

    for x, y, label, color in zip(xs, ys, labels, colors):
        plt.plot(x, y, color=color, linewidth=1.5, label=label)

    group_labels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # x轴刻度的标识
    plt.xticks(group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("1-R(Non-Clone)", fontsize=13, fontweight='bold')
    plt.ylabel("R(Clone)", fontsize=13, fontweight='bold')
    plt.xlim(0, 1)  # 设置x轴的范围
    plt.ylim(0, 1)
    # plt.legend()
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
    if save_figure:
        plt.savefig(save_figure, format='pdf')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    #if platform.system() == "Windows": plt.show()



