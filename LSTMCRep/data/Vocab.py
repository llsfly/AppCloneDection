import numpy as np


class Vocab(object):

    # please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, name_counter):
        self._id2name = []
        self._id2strname = []
        for name, count in name_counter.most_common():
            repreat = int(np.power(count, 0.75))
            for idx in range(repreat):
                self._id2strname.append(name)
                self._id2name.append(name.split())

        print("Vocab info: #output names %d" % self.name_size)

    def create_token_embs(self, embfile):
        embedding_dim = -1
        alltokens = set()
        idx = 0
        for special_token in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_token not in alltokens:
                alltokens.add(special_token)
                if self._id2token[idx] != special_token:
                    print("error pair: (%s, %s)" %(self._id2token[idx], special_token))
                idx += 1

        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curtoken = values[0]
                    if curtoken not in alltokens:
                        alltokens.add(curtoken)
                        if self._id2token[idx] != curtoken:
                            print("error pair: (%s, %s)" % (self._id2token[idx], curtoken))
                        idx += 1
                    embedding_dim = len(values) - 1
        token_num = len(self._id2token)
        print('Total tokens: ' + str(token_num) + '\n')
        print('The dim of pretrained token embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._token2id = reverse(self._id2token)

        if len(self._token2id) != len(self._id2token):
            print("serious bug: tokens dumplicated, please check!")

        oov_id = self._token2id.get('<oov>')
        if self.UNK != oov_id:
            print("serious bug: oov token id is not correct, please check!")

        embeddings = np.zeros((token_num, embedding_dim))

        return embeddings

    def create_word_embs(self, embfile):
        embedding_dim = -1
        allwords = set()
        idx = 0
        for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_word not in allwords:
                allwords.add(special_word)
                if self._id2word[idx] != special_word:
                    print("error pair: (%s, %s)" %(self._id2word[idx], special_word))
                idx += 1

        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        if self._id2word[idx] != curword:
                            print("error pair: (%s, %s)" % (self._id2word[idx], curword))
                        idx += 1
                    embedding_dim = len(values) - 1
        word_num = len(self._id2word)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained word embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)

        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._word2id.get('<oov>')
        if self.UNK != oov_id:
            print("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))

        return embeddings

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._token2id.get(x, self.UNK) for x in xs]
        return self._token2id.get(xs, self.UNK)

    def id2token(self, xs):
        if isinstance(xs, list):
            return [self._id2token[x] for x in xs]
        return self._id2token[xs]

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def id2names(self, xs):
        if xs >= self.name_size or xs < 0:
            print('error: %d' % xs)
            xs = 0
        return self._id2strname[xs], self._id2name[xs]

    @property
    def vocab_size(self):
        return len(self._id2token)

    @property
    def word_size(self):
        return len(self._id2token)

    @property
    def name_size(self):
        return len(self._id2name)

