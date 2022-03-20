import torch
from torch.autograd import Variable


class TensorInstances:
    def __init__(self, batch_size, max_len):
        self.tokens = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
        self.masks = Variable(torch.Tensor(batch_size, max_len).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.tokens = self.tokens.cuda(device)
        self.masks = self.masks.cuda(device)

    @property
    def inputs(self):
        return self.tokens, self.masks

