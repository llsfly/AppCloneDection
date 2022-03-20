class Instance:
    def __init__(self, src_id, tgt_id, tag):
        self.src_id = src_id
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
