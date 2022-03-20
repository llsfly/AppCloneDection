from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import codecs

def read_corpus(clone_file):
    clone_data, non_clones = [], []
    with codecs.open(clone_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split(',')
            if len(items) != 3: continue
            src_id, tgt_id = int(items[0]), int(items[1])
            curInst = Instance(src_id, tgt_id, items[2])
            if items[2] == '0': non_clones.append(curInst)
            else: clone_data.append(curInst)

    print("Total num: #clones:%d,  #nonclones: %d" % (len(clone_data), len(non_clones)))
    return clone_data, non_clones

def creatVocab(alldatas):
    tag_counter = Counter()
    for inst in alldatas:
        tag_counter[inst.tag] += 1

    return Vocab(tag_counter)

def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)

def inst2id(inst, vocab):
    tagid = vocab.tag2id(inst.tag)
    return tagid, inst


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
    repr_dim = vocab.embedding_dim
    batch_size = len(batch)
    tinst = TensorInstances(batch_size, repr_dim)

    b = 0
    for tagid, inst in insts_numberize(batch, vocab):
        tinst.tags[b] = tagid
        src_repr = vocab.id2repr(inst.src_id)
        tinst.src_reprs[b].copy_(torch.from_numpy(src_repr))
        tgt_repr = vocab.id2repr(inst.tgt_id)
        tinst.tgt_reprs[b].copy_(torch.from_numpy(tgt_repr))
        
        b += 1
    return tinst

def batch_variable_inst(insts, tagids, probs, vocab):
    for inst, tagid, prob in zip(insts, tagids, probs):
        pred_tag = vocab.id2tag(tagid)
        str_probs = []
        for idx, p in enumerate(prob):
            cur_tag = vocab.id2tag(idx)
            str_probs.append(cur_tag + " " + str(p))
        yield Instance(inst.src_id,  inst.tgt_id, pred_tag), pred_tag == inst.tag, ' '.join(str_probs), inst.tag


import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bcb_file', default='clone4/bcb_map.txt')
    argparser.add_argument('--num', default=1000, type=int, help='clone num')
    argparser.add_argument('--train_out', default='clone4/bcb_map.train.txt')
    argparser.add_argument('--test_out', default='clone4/bcb_map.test.txt')

    args, extra_args = argparser.parse_known_args()
    clone_data, non_clones = read_corpus(args.bcb_file)

    np.random.shuffle(clone_data)
    np.random.shuffle(non_clones)

    train_out = open(args.train_out, 'w')
    test_out = open(args.test_out, 'w')

    for idx, cur_inst in enumerate(clone_data):
        if idx < args.num:
            printInstance(train_out, cur_inst)
        else:
            printInstance(test_out, cur_inst)

    for idx, cur_inst in enumerate(non_clones):
        if idx < 5 * args.num:
            printInstance(train_out, cur_inst)
        else:
            printInstance(test_out, cur_inst)

    train_out.close()
    test_out.close()


