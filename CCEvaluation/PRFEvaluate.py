import argparse
from collections import Counter
import time
import os
import numpy as np
from BinaryData import *

def Evaluate(datas, threshold=0.5):
    corrected = Counter()
    detected = Counter()
    golded = Counter()
    pos_tag = datas[0].pos_tag
    for data in datas:
        pred = data.pos_tag if data.pos_prob >= threshold else data.neg_tag
        if pred == data.true_tag:
            corrected[data.true_tag] += 1
        golded[data.true_tag] += 1
        detected[pred] += 1
    total_correct_pos = 0
    total_correct_neg = 0
    total_detect_pos = 0
    total_detect_neg = 0
    total_pos = 0
    total_neg = 0
    for tag, val in corrected.most_common():
        if tag == pos_tag:
            total_correct_pos += val
        else:
            total_correct_neg += val
    for tag, val in detected.most_common():
        if tag == pos_tag:
            total_detect_pos += val
        else:
            total_detect_neg += val
    for tag, val in golded.most_common():
        if tag == pos_tag:
            total_pos += val
        else:
            total_neg += val

    R_neg = float(total_correct_neg / total_neg)
    R_pos = float(total_correct_pos / total_pos)
    P_pos = float(total_correct_pos / total_detect_pos)
    P_neg = float(total_correct_neg / total_detect_neg)
    F_pos = 200 * P_pos * R_pos / (R_pos + P_pos)
    F_neg = 200 * P_neg * R_neg / (P_neg + R_neg)

    print('pos P %.2f, R %.2f, F %.2f, neg P %.2f, R %.2f, F %.2f' %
          (100 * P_pos, 100 * R_pos, F_pos, 100 * P_neg, 100 * R_neg, F_neg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file', required=True)
    parser.add_argument('--threshold', type=float, help='input file', required=False, default='0.5')
    args = parser.parse_args()
    input_file = args.input


    print('Load data from %s' % (input_file))
    reader = open(input_file, 'r', encoding='utf-8')
    datas = []
    for line in reader.readlines():
        line = line.strip()
        if line != '' and line is not None:
            data = parse(line)
            datas.append(data)
    reader.close()

    Evaluate(datas, args.threshold)

