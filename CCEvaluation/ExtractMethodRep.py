import argparse
import hashlib
import time
from collections import Counter
import numpy as np
from numpy import linalg
import codecs

class Embedding:
    def __init__(self, embfile):
        self._embedding_dim = -1
        self._id2word = []
        allwords = set()
    
        allwords.add('<oov>')
        self._id2word.append('<oov>')
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2word.append(curword)
                    self._embedding_dim = len(values) - 1
        word_num = len(self._id2word)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained word self._embeddings: ' + str(self._embedding_dim) + '\n')
    
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
    
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")
    
        oov_id = self._word2id.get('<oov>')
        if 0 != oov_id:
            print("serious bug: oov word id is not correct, please check!")
    
        self._embeddings = np.zeros((word_num, self._embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    index = self._word2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    self._embeddings[index] = vector
                    self._embeddings[0] += vector
    
        self._embeddings[0] = self._embeddings[0] / word_num

    def word2embed(self, xs):
        ids = [self._word2id.get(x, 0) for x in xs]
        reprs = np.zeros(self._embedding_dim, dtype='float64')
        for id in ids:
            reprs += self._embeddings[id]
        reprs = reprs / len(ids)
        return reprs


def CDRepByName(embeds, infile, outfile, display_interval):
    start = time.time()
    output = open(outfile, 'w', encoding='utf-8')
    count = 0
    with codecs.open(infile, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip().lower()
            items = line.split('\t')
            if len(items) != 2: continue
            named_tokens = items[1].split()

            reprs = embeds.word2embed(named_tokens)
            str_out = [ str(val) for val in reprs]
            output.write(items[0] + '\t' + ' '.join(str_out))
            output.flush()
            count += 1
            if count % display_interval == 0:
                end = time.time()
                during_time = float(end - start)
                print("processing: %d, time=%.2f" % (count, during_time))

    end = time.time()
    during_time = float(end - start)
    print("processing: %d, time=%.2f" % (count, during_time))
    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='embedding file', default='clones/glove.6B.100d.txt')
    parser.add_argument('--input', help='code pair file', default='clones/bcb-names.txt')
    parser.add_argument('--output', help='output file', default='clones/bcb.names.represent')
    parser.add_argument('--interval', default=500, type=int, help='interval')
    args = parser.parse_args()

    embeds = Embedding(args.embedding)

    CDRepByName(embeds, args.input, args.output, args.interval)







