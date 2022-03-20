class Instance:
    def __init__(self, tokens, names, tag):
        self.tokens = tokens
        self.names = names
        self.str_names = ' '.join(self.names)
        self.tag = tag

    def __str__(self):
        output = self.str_names + '\t' + ' '.join(self.tokens) + '\t' + str(self.tag)
        return output

    @property
    def token_len(self):
        return len(self.tokens)

    @property
    def name_len(self):
        return len(self.names)


def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
