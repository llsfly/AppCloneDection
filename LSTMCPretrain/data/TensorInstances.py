import torch
from torch.autograd import Variable


class TensorInstances:
    def __init__(self, batch_size, token_len, name_len):
        self.tokens = Variable(torch.LongTensor(batch_size, token_len).zero_(), requires_grad=False)
        self.token_masks = Variable(torch.Tensor(batch_size, token_len).zero_(), requires_grad=False)
        self.names = Variable(torch.LongTensor(batch_size, name_len).zero_(), requires_grad=False)
        self.name_masks = Variable(torch.Tensor(batch_size, name_len).zero_(), requires_grad=False)
        self.tags = Variable(torch.Tensor(batch_size).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.tokens = self.tokens.cuda(device)
        self.token_masks = self.token_masks.cuda(device)
        self.names = self.names.cuda(device)
        self.name_masks = self.name_masks.cuda(device)
        self.tags = self.tags.cuda(device)

    @property
    def inputs(self):
        return self.tokens, self.token_masks, self.names, self.name_masks

    @property
    def outputs(self):
        return self.tags
