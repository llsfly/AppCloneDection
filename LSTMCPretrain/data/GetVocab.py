import argparse
from collections import Counter
import codecs

def get_vocab(in_file, out_file):
    name_counter = Counter()
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            for curword in items:
                name_counter[curword.lower()] += 1

    with codecs.open(out_file, 'w', encoding='utf8') as out_write:
        for name, count in name_counter.most_common():
            out_write.write(name + "\n")

    print("Total num: #words:%d" % (len(name_counter)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', default='in_file')
    argparser.add_argument('--output', default='out_file')

    args, extra_args = argparser.parse_known_args()

    get_vocab(args.input, args.output)



