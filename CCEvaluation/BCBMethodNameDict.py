import argparse
import codecs
from collections import Counter

def get_vocab(in_file, out_file):
    name_counter = Counter()
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip().lower()
            items = line.split('\t')
            for curword in items[1].split():
                name_counter[curword.lower()] += 1

    with codecs.open(out_file, 'w', encoding='utf8') as out_write:
        for name, count in name_counter.most_common():
            out_write.write(name + "\n")

    print("Total num: #words:%d" % (len(name_counter)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', default='clones/bcb-names.txt')
    argparser.add_argument('--output', default='clones/bcb.names.dict')

    args, extra_args = argparser.parse_known_args()

    get_vocab(args.input, args.output)







