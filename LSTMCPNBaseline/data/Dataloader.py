from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import codecs

def read_codes(code_file):
    codes = {}
    seq_codes = []
    with codecs.open(code_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split('\t')
            if len(items) != 2: continue
            index = int(items[0])
            if index in codes:
                print("Reduplicated key: " + items[0])
                continue

            tokens = items[1].split()
            tokens = tokens[:512]
            codes[index] = tokens

            cur_inst = CodeInstance(tokens, index)
            seq_codes.append(cur_inst)

    print("Total num: #codes: %d" % (len(codes)))
    return codes, seq_codes



def read_corpus(codes, clone_file):
    clone_data, non_clones = [], []
    with codecs.open(clone_file, encoding='utf8') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            items = line.split(',')
            if len(items) != 3: continue
            src_id, tgt_id = int(items[0]), int(items[1])
            if (src_id not in codes) or  (tgt_id not in codes): continue
            curInst = Instance(codes[src_id], src_id, codes[tgt_id], tgt_id, items[2])
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
    srcids = vocab.word2id(inst.src_words)
    tgtids = vocab.word2id(inst.tgt_words)
    tagid = vocab.tag2id(inst.tag)

    return srcids, tgtids, tagid, inst


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


def batch_data_variable_train(batch, vocab):
    slen, tlen = len(batch[0].src_words), len(batch[0].tgt_words)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_slen, cur_tlen = len(batch[b].src_words), len(batch[b].tgt_words)
        if cur_slen > slen: slen = cur_slen
        if cur_tlen > tlen: tlen = cur_tlen

    repr_dim = vocab.embedding_dim
    tinst = TrainTensorInstances(batch_size, slen, tlen, repr_dim)

    b = 0
    for srcids, tgtids, tagid, inst in insts_numberize(batch, vocab):
        tinst.tags[b] = tagid
        cur_slen, cur_tlen = len(inst.src_words), len(inst.tgt_words)

        src_repr = vocab.id2repr(inst.src_id)
        tinst.src_reprs[b].copy_(torch.from_numpy(src_repr))
        for index in range(cur_slen):
            tinst.src_words[b, index] = srcids[index]
            tinst.src_masks[b, index] = 1

        tgt_repr = vocab.id2repr(inst.tgt_id)
        tinst.tgt_reprs[b].copy_(torch.from_numpy(tgt_repr))
        for index in range(cur_tlen):
            tinst.tgt_words[b, index] = tgtids[index]
            tinst.tgt_masks[b, index] = 1
        
        b += 1
    return tinst


def batch_data_variable_test(batch, vocab, repr_dim, code_reprs):
    batch_size = len(batch)
    tinst = TestTensorInstances(batch_size, repr_dim)
    b = 0
    for inst in batch:
        tinst.tags[b] = vocab.tag2id(inst.tag)
        src_repr = code_reprs.get(inst.src_id)
        tinst.src_reprs[b].copy_(src_repr)
        tgt_repr = code_reprs.get(inst.tgt_id)
        tinst.tgt_reprs[b].copy_(tgt_repr)

        b += 1
    return tinst


def batch_data_variable_code(batch, vocab):
    max_len = len(batch[0].tokens)
    batch_size = len(batch)
    for b in range(1, batch_size):
        cur_len = len(batch[b].tokens)
        if cur_len > max_len: max_len = cur_len

    repr_dim = vocab.embedding_dim
    tinst = CodeTensorInstances(batch_size, max_len, repr_dim)

    b = 0
    for inst in batch:
        cur_len = len(inst.tokens)
        cur_repr = vocab.id2repr(inst.id)
        tinst.reprs[b].copy_(torch.from_numpy(cur_repr))
        token_ids = vocab.word2id(inst.tokens)
        for index in range(cur_len):
            tinst.tokens[b, index] = token_ids[index]
            tinst.masks[b, index] = 1
        b += 1
    return tinst


def batch_variable_inst(insts, tagids, probs, vocab):
    for inst, tagid, prob in zip(insts, tagids, probs):
        pred_tag = vocab.id2tag(tagid)
        str_probs = []
        for idx, p in enumerate(prob):
            cur_tag = vocab.id2tag(idx)
            str_probs.append(cur_tag + " " + str(p))
        yield Instance(inst.src_words,  inst.src_id,  inst.tgt_words, inst.tgt_id,  pred_tag), \
              pred_tag == inst.tag, ' '.join(str_probs), inst.tag
