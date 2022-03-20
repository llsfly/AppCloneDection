import argparse
import codecs
from PRFEvaluate import *
import math

def get_func_stat(in_file):
    oovs, totals = {}, {}
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            if len(items) != 3: continue
            oovs[items[0]] = int(items[1])
            totals[items[0]] = int(items[2])

    return oovs, totals

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stat', default='bcb.txt')
    argparser.add_argument('--input', default='result.txt')
    argparser.add_argument('--split', default=10, type=int, help='max function length')

    args, extra_args = argparser.parse_known_args()

    interval = 1.0 / args.split

    alldatas = []
    for idx in range(args.split):
        alldatas.append([])

    oovs, totals = get_func_stat(args.stat)
    print('Load data from %s' % (args.input))
    reader = open(args.input, 'r', encoding='utf-8')
    for line in reader.readlines():
        line = line.strip()
        if line != '' and line is not None:
            data = parse(line)
            bi_oov_count = oovs[data.src_key] + oovs[data.tgt_key]
            bi_total_count = totals[data.src_key] + totals[data.tgt_key]
            oov_ratio = float(bi_oov_count) * 1.0 / bi_total_count
            cur_count = int(math.floor(oov_ratio/interval))
            alldatas[cur_count].append(data)
    reader.close()

    for idx, datas in enumerate(alldatas):
        cur_count = len(datas)
        print(str(idx) + ":" + str(cur_count))
        if cur_count > 0:
            Evaluate(datas, 0.5)
