class Instance:
    def __init__(self, tokens, id):
        self.tokens = tokens
        self.id = id

    def __str__(self):
        output = str(self.id) + '\t' + ' '.join(self.tokens)
        return output

    @property
    def len(self):
        return len(self.tokens)

def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
