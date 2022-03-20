import numpy as np

class Vocab(object):
    ##please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    def __init__(self, tag_counter):
        self._id2tag = []
        for tag, count in tag_counter.most_common():
            self._id2tag.append(tag)
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: output tags dumplicated, please check!")

        print("Vocab info: #output tags %d" % (self.tag_size))

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
    def tag_size(self):
        return len(self._id2tag)





