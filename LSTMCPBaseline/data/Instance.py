class Instance:
    def __init__(self, src_words, src_id, tgt_words, tgt_id, tag):
        self.src_words = src_words
        self.src_id = src_id
        self.tgt_words = tgt_words
        self.tgt_id = tgt_id
        self.tag = tag

    def __str__(self):
        output = str(self.src_id) + ' ' + str(self.tgt_id) + ' ' + self.tag
        return output

def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')



class CodeInstance:
    def __init__(self, tokens, id):
        self.tokens = tokens
        self.id = id

    def __str__(self):
        output = str(self.id) + '\t' + ' '.join(self.tokens)
        return output

    @property
    def len(self):
        return len(self.tokens)
