import torch
from torch.autograd import Variable


class TensorInstances:
    def __init__(self, batch_size, repr_dim):
        self.src_reprs = Variable(torch.Tensor(batch_size, repr_dim).zero_(), requires_grad=False)
        self.tgt_reprs = Variable(torch.Tensor(batch_size, repr_dim).zero_(), requires_grad=False)
        self.tags = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.src_reprs = self.src_reprs.cuda(device)
        self.tgt_reprs = self.tgt_reprs.cuda(device)
        self.tags = self.tags.cuda(device)

    @property
    def inputs(self):
        return self.src_reprs, self.tgt_reprs

    @property
    def outputs(self):
        return self.tags
