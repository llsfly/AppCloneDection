import argparse
import hashlib
import time
from collections import Counter
import numpy as np
from numpy import linalg
import codecs


class Instance:
    def __init__(self, src_key, tgt_key, tag):
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.tag = tag

    def __str__(self):
        output = self.src_key + ' ' + self.tgt_key + ' ' + self.tag
        return output


def read_corpus(clone_file):
    clone_data, non_clones = [], []
    with codecs.open(clone_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split(',')
            if len(items) != 3: continue
            src_key, tgt_key = items[0], items[1]
            curInst = Instance(src_key, tgt_key, items[2])
            if items[2] == '0': non_clones.append(curInst)
            else: clone_data.append(curInst)

    print("Total num: #clones:%d,  #nonclones: %d" % (len(clone_data), len(non_clones)))
    return clone_data, non_clones


def load_pretrained_reprs(represent_file):
    embedding_dim = -1
    code_represents = {}
    with open(represent_file, encoding='utf-8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split('\t')
            if len(items) != 2: continue
            index = items[0]
            if index in code_represents:
                print("Reduplicated key: " + items[0])
                continue
            vector = np.array(items[1].split(), dtype='float64')
            cur_dim = len(vector)
            if embedding_dim == -1:
                embedding_dim = cur_dim
                print("Represent dim: " + str(cur_dim))
            elif cur_dim != embedding_dim:
                print("Error dim: " + str(cur_dim))
                continue
            code_represents[index] = vector

    print("total codes: " + str(len(code_represents)))

    return embedding_dim, code_represents


def CCDetection(datas, code_represents, embedding_dim, output, display_interval):
    start = time.time()
    pseudo = np.zeros(embedding_dim, dtype='float64')
    tag_correct, tag_total = 0, 0
    pos_label, neg_label = '1', '0'
    for data in datas:
        src_repr = code_represents.get(data.src_key, pseudo)
        tgt_repr = code_represents.get(data.tgt_key, pseudo)
        cosine = np.dot(src_repr, tgt_repr) / (linalg.norm(src_repr) * linalg.norm(tgt_repr) + 1e-20)
        pos_prob = 0.5 + 0.5 * cosine
        neg_prob = 1 - pos_prob
        default = pos_label if pos_prob > 0.5 else neg_label
        items = [data.src_key, data.tgt_key, default, \
                 pos_label, str(pos_prob), \
                 neg_label, str(neg_prob), data.tag]
        output.write(' '.join(items) + '\n')
        if data.tag == default: tag_correct += 1
        tag_total += 1

        if tag_total % display_interval == 0:
            acc = tag_correct * 100.0 / tag_total
            end = time.time()
            during_time = float(end - start)
            print("processing: acc=%d/%d=%.2f, classifier time=%.2f" % (tag_correct, tag_total, acc, during_time))

    output.flush()
    acc = tag_correct * 100.0 / tag_total
    end = time.time()
    during_time = float(end - start)
    print("processing: acc=%d/%d=%.2f,  classifier time=%.2f " % (tag_correct, tag_total, acc, during_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--represent', help='representation file', required=True, default='bcb.represent')
    parser.add_argument('--input', help='code pair file', required=True, default='bcb_map.test.txt')
    parser.add_argument('--output', help='output file', required=False, default='bcb_map.test.txt.cosine')
    parser.add_argument('--interval', default=100, type=int, help='max training clone num')
    args = parser.parse_args()

    embedding_dim, code_represents = load_pretrained_reprs(args.represent)
    test_clone_data, test_non_clones = read_corpus(args.input)

    output = open(args.output, 'w', encoding='utf-8')
    CCDetection(test_clone_data, code_represents, embedding_dim, output, args.interval)
    CCDetection(test_non_clones, code_represents, embedding_dim, output, args.interval)
    output.close()





