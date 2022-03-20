import argparse
import codecs

def get_emb_words(in_file):
    words = set()
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            for curword in items:
                words.add(curword.lower())

    print("Total num: #words:%d" % (len(words)))

    return words

def get_bcb_words(in_file):
    words = set()
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            valid = False
            for curword in items:
                if valid:
                    words.add(curword.lower())
                if (valid is False) and (curword == '{'):
                    valid = True

    print("Total num: #words:%d" % (len(words)))

    return words


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bcb', default='bcb.txt')
    argparser.add_argument('--emb', default='simple.vec')

    args, extra_args = argparser.parse_known_args()

    bcb_words = get_bcb_words(args.bcb)
    emb_words = get_emb_words(args.emb)
    count = 0
    for bcb_word in bcb_words:
        if bcb_word in emb_words:
            count += 1

    bcb_count = len(bcb_words)
    oov_ratio = count * 100.00 / bcb_count
    print("oov ration=%d/%d=%.2f" % (bcb_count-count, bcb_count, oov_ratio))





