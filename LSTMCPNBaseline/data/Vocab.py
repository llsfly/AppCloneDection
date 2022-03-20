import numpy as np
class Vocab(object):
    ##please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3
    def __init__(self, tag_counter):
        self._id2tag = []
        for tag, count in tag_counter.most_common():
            self._id2tag.append(tag)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: output tags dumplicated, please check!")

        print("Vocab info: #output tags %d" % (self.tag_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        self._id2word = []
        allwords = set()
        for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2word.append(special_word)

        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2word.append(curword)
                    embedding_dim = len(values) - 1
        word_num = len(self._id2word)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)

        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._word2id.get('<oov>')
        if self.UNK != oov_id:
            print("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    index = self._word2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector

        embeddings[self.UNK] = embeddings[self.UNK] / word_num

        return embeddings

    def load_pretrained_reprs(self, represent_file):
        self.embedding_dim = -1
        self.code_represents = {}
        with open(represent_file, encoding='utf-8') as input_file:
            for line in input_file.readlines():
                line = line.strip()
                items = line.split('\t')
                if len(items) != 2: continue
                index = int(items[0])
                if index in self.code_represents:
                    print("Reduplicated key: " + items[0])
                    continue
                vector = np.array(items[1].split(), dtype='float64')
                cur_dim = len(vector)
                if self.embedding_dim == -1:
                    self.embedding_dim = cur_dim
                    print("Represent dim: " + str(cur_dim))
                elif cur_dim != self.embedding_dim:
                    print("Error dim: " + str(cur_dim))
                    continue
                self.code_represents[index] = vector

        if self.embedding_dim > 0:
            self.pseudo = np.zeros(self.embedding_dim, dtype='float64')

        print("total codes: " + str(len(self.code_represents)))

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]


    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    def id2repr(self, xs):
        return self.code_represents.get(xs, self.pseudo)

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)


