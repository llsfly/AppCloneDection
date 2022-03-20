import argparse
import fasttext
import codecs

def get_words(in_file):
    words = []
    with codecs.open(in_file, encoding='utf8') as in_read:
        for line in in_read.readlines():
            line = line.strip()
            items = line.split()
            for curword in items:
                words.append(curword.lower())

    print("Total num: #words:%d" % (len(words)))

    return words


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', default='wordfile')
    argparser.add_argument('--model', default='model.bin')
    argparser.add_argument('--output', default='words.vec')

    args, extra_args = argparser.parse_known_args()

    words = get_words(args.input)
    model = fasttext.load_model(args.model)
    with codecs.open(args.output, 'w', encoding='utf8') as out_write:
        for curword in words:
            outs = []
            outs.append(curword)
            vec = model.get_word_vector(curword)
            for idx in range(len(vec)):
                outs.append(str(vec[idx]))
            out_write.write(' '.join(outs) + "\n")




