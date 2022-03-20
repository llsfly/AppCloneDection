from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import random
import codecs

def read_positive_corpus(code_file, max_length):
    codes = []
    with codecs.open(code_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip().lower()
            items = line.split('\t')
            if len(items) != 2: continue
            names = items[0].split()
            named_tokens = items[1].split()
            start, end = 0, len(named_tokens) - 1
            while start < end:
                if named_tokens[start] == '{':
                    break
                start = start + 1

            tokens = named_tokens[start+1:end]
            if len(tokens) > max_length: continue
            cur_inst = Instance(tokens, names, 1)
            codes.append(cur_inst)

    print("Total num: #codes:%d" % (len(codes)))
    return codes


def creat_vocab(pos_insts):
    name_counter = Counter()
    for inst in pos_insts:
        name_counter[inst.str_names] += 1

    return Vocab(name_counter)

def creat_negative_corpus(pos_insts, vocab, neg_count):
    max_int = vocab.name_size
    neg_codes = []
    for inst in pos_insts:
        repeat = 0
        while repeat < neg_count:
            idx = random.randint(0, max_int-1)
            str_names, names = vocab.id2names(idx)
            if str_names == inst.str_names: continue
            cur_inst = Instance(inst.tokens, names, -1)
            neg_codes.append(cur_inst)
            repeat = repeat + 1
    return neg_codes

def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)

def inst2id(inst, vocab):
    tokenids = vocab.token2id(inst.tokens)
    nameids = vocab.word2id(inst.names)
    tagid = inst.tag

    return tokenids, nameids, tagid, inst


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
    token_len, name_len = len(batch[0].tokens), len(batch[0].names)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_token_len, cur_name_len = len(batch[b].tokens), len(batch[b].names)
        if cur_token_len > token_len: token_len = cur_token_len
        if cur_name_len > name_len: name_len = cur_name_len

    tinst = TensorInstances(batch_size, token_len, name_len)

    b = 0
    for tokenids, nameids, tagid, inst in insts_numberize(batch, vocab):
        tinst.tags[b] = tagid
        cur_token_len, cur_name_len = len(inst.tokens), len(inst.names)
        for index in range(cur_token_len):
            tinst.tokens[b, index] = tokenids[index]
            tinst.token_masks[b, index] = 1

        for index in range(cur_name_len):
            tinst.names[b, index] = nameids[index]
            tinst.name_masks[b, index] = 1
        
        b += 1
    return tinst

def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.tokens, inst.names, pred_tag), pred_tag == inst.tag



