import argparse
import codecs

def get_emb_words(in_file):
    words = set()
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            if len(items) > 0:
                words.add(items[0].lower())

    print("Total num: #words:%d" % (len(words)))

    return words


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bcb', default='bcb.txt')
    argparser.add_argument('--emb', default='simple.vec')
    argparser.add_argument('--output', default='oov.txt')

    args, extra_args = argparser.parse_known_args()

    emb_words = get_emb_words(args.emb)
    output = open(args.output, 'w', encoding='utf-8')
    with codecs.open(args.bcb, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip().lower()
            items = line.split()
            if len(items) == 0: continue
            str_index = items[0]
            total_count, iv_count = 0, 0
            valid = False
            for curword in items:
                if valid:
                    total_count += 1
                    if curword in emb_words:
                        iv_count += 1
                if (valid is False) and (curword == '{'):
                    valid = True
            oov_count = total_count - iv_count
            output.write(str_index + ' ' + str(oov_count) + ' ' + str(total_count) + '\n')

    output.close()





