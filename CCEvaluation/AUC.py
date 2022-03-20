import argparse
from BinaryData import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import os

def AUCEvaluate(datas):
    gold_labels = []
    positive_scores = []
    pos_tag = datas[0].pos_tag
    for data in datas:
        positive_scores.append(data.pos_prob)
        cur_label = 1 if pos_tag == data.true_tag else 0
        gold_labels.append(cur_label)

    gold_labels = np.array(gold_labels)
    positive_scores = np.array(positive_scores)
    fpr, tpr, thresholds = roc_curve(gold_labels, positive_scores, pos_label=1)
    auc_score = roc_auc_score(gold_labels, positive_scores)

    return fpr, tpr, auc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--output', help='output file', required=True, default="result.txt")
    parser.add_argument('--save_fig', required=True, help='save figure path', default="result.pdf")

    args = parser.parse_args()

    input_file = args.input
    output_figure = args.save_fig

    output = open(args.output, 'w', encoding='utf-8')

    xs = []
    ys = []
    labels = []
    for input_file in args.input:
        print('Processing file: ' + input_file)
        output.write('Processing file: ' + input_file + '\n')
        reader = open(input_file, 'r', encoding='utf-8')
        datas = []
        for line in reader.readlines():
            line = line.strip()
            if line != '' and line is not None:
                data = parse(line)
                datas.append(data)
        reader.close()
        fpr, tpr, auc_score = AUCEvaluate(datas)
        str_auc = '%.6f' % auc_score
        print('AUC score: ' + str_auc)
        output.write('AUC score: ' + str_auc + '\n')

        points = set()
        for coordinate_x, coordinate_y in zip(fpr, tpr):
            str_point = '(%.4f, %.4f)\n' % (coordinate_x, coordinate_y)
            if str_point not in points:
                output.write(str_point)
                points.add(str_point)
        print('============')
        output.write('============\n')

        xs.append(fpr)
        ys.append(tpr)
        named_file = input_file.split(os.path.sep)[::-1][0]
        labels.append(named_file.split('.')[-1] + ', auc = ' + str_auc)

    output.close()
    draw_figure_and_save(xs, ys, labels, save_figure=output_figure)

