import numpy as np


def load_pretrained_reprs(represent_file):
    embedding_dim = -1
    code_represents = {}
    with open(represent_file, encoding='utf-8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split('\t')
            if len(items) != 2: continue
            index = int(items[0])
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


import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input1', default='clone4/bcb.represent1')
    argparser.add_argument('--input2', default='clone4/bcb.represent2')
    argparser.add_argument('--output', default='clone4/bcb.represent.merge')

    args, extra_args = argparser.parse_known_args()
    embedding_dim1, code_represents1 = load_pretrained_reprs(args.input1)
    embedding_dim2, code_represents2 = load_pretrained_reprs(args.input2)

    pseudo1 = np.zeros(embedding_dim1, dtype='float64')
    pseudo2 = np.zeros(embedding_dim2, dtype='float64')

    allKeys = set()
    for curKey in code_represents1.keys():
        allKeys.add(curKey)
    for curKey in code_represents2.keys():
        allKeys.add(curKey)

    output = open(args.output, 'w')

    for curKey in allKeys:
        strnumbers = []
        if curKey in code_represents1:
            for curv in code_represents1[curKey]:
                strnumbers.append(str(curv))
        else:
            for curv in pseudo1:
                strnumbers.append(str(curv))

        if curKey in code_represents2:
            for curv in code_represents2[curKey]:
                strnumbers.append(str(curv))
        else:
            for curv in pseudo2:
                strnumbers.append(str(curv))

        assert(len(strnumbers) == (embedding_dim1 + embedding_dim2))

        strout = str(curKey) + '\t' + ' '.join(strnumbers)
        output.write(strout + "\n")

    output.close()



