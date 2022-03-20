from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
import codecs

def read_corpus(code_file):
    indexes = set()
    codes = []
    with codecs.open(code_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip().lower()
            items = line.split('\t')
            if len(items) != 2: continue
            index = int(items[0])
            tokens = items[1].split()
            tokens = tokens[:512]
            if index in indexes:
                print("Reduplicated key: " + items[0])
                continue
            cur_inst = Instance(tokens, index)
            codes.append(cur_inst)

    print("Total num: #codes:%d" % (len(codes)))
    return codes


def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)

def inst2id(inst, vocab):
    token_ids = vocab.token2id(inst.tokens)
    return token_ids, inst


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield insts


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  insts in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    max_len = len(batch[0].tokens)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_len = len(batch[b].tokens)
        if cur_len > max_len: max_len = cur_len

    tinst = TensorInstances(batch_size, max_len)

    b = 0
    for token_ids, inst in insts_numberize(batch, vocab):
        cur_len = len(inst.tokens)
        for index in range(cur_len):
            tinst.tokens[b, index] = token_ids[index]
            tinst.masks[b, index] = 1

        b += 1
    return tinst

