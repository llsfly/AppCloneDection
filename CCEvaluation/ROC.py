import argparse
from collections import Counter
import os
from BinaryData import *


def ROCEvaluate(datas, threshold):
    corrected = Counter()
    golded = Counter()
    pos_tag = datas[0].pos_tag
    for data in datas:
        if data.condTrue(threshold):
            corrected[data.true_tag] += 1
        golded[data.true_tag] += 1
    total_correct_pos = 0
    total_correct_neg = 0
    total_pos = 0
    total_neg = 0
    for tag, val in corrected.most_common():
        if tag == pos_tag:
            total_correct_pos += val
        else:
            total_correct_neg += val
    for tag, val in golded.most_common():
        if tag == pos_tag:
            total_pos += val
        else:
            total_neg += val
    R_neg = float(total_correct_neg / total_neg)
    R_pos = float(total_correct_pos / total_pos)

    return 1 - R_neg, R_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--sample', default=10, type=int, help='sample num')
    parser.add_argument('--save_fig', required=False, help='save figure path')

    args = parser.parse_args()

    input_file = args.input
    output_figure = args.save_fig

    thresholds = [0]
    interval = 1.0/args.sample
    for idx in range(args.sample):
        thresholds.append( (idx+1) * interval)

    xs = []
    ys = []
    labels = []
    for input_file in args.input:
        print('Processing file: ' + input_file)
        reader = open(input_file, 'r', encoding='utf-8')
        datas = []
        for line in reader.readlines():
            line = line.strip()
            if line != '' and line is not None:
                data = parse(line)
                datas.append(data)
        reader.close()

        x = []
        y = []
        for threshold in thresholds:
            coordinate_x, coordinate_y = ROCEvaluate(datas, threshold)
            x.append(coordinate_x)
            y.append(coordinate_y)

        # 输出坐标点
        for coordinate_x, coordinate_y in zip(x, y):
            print('(%.4f, %.4f)' % (coordinate_x, coordinate_y))
        print('=======')
        labels.append(input_file.split(os.path.sep)[::-1][0])
        xs.append(x)
        ys.append(y)

    # 不画图的话就注释掉下面这句
    draw_figure_and_save(xs, ys, labels, save_figure=output_figure)
