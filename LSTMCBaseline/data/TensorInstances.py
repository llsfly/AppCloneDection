import torch
from torch.autograd import Variable


class TrainTensorInstances:
    def __init__(self, batch_size, slen, tlen):
        self.src_words = Variable(torch.LongTensor(batch_size, slen).zero_(), requires_grad=False)
        self.src_masks = Variable(torch.Tensor(batch_size, slen).zero_(), requires_grad=False)
        self.tgt_words = Variable(torch.LongTensor(batch_size, tlen).zero_(), requires_grad=False)
        self.tgt_masks = Variable(torch.Tensor(batch_size, tlen).zero_(), requires_grad=False)
        self.tags = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tgt_words = self.tgt_words.cuda(device)
        self.tgt_masks = self.tgt_masks.cuda(device)
        self.tags = self.tags.cuda(device)

    @property
    def inputs(self):
        return self.src_words, self.src_masks, self.tgt_words, self.tgt_masks

    @property
    def outputs(self):
        return self.tags


class TestTensorInstances:
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


class CodeTensorInstances:
    def __init__(self, batch_size, max_len):
        self.tokens = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
        self.masks = Variable(torch.Tensor(batch_size, max_len).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.tokens = self.tokens.cuda(device)
        self.masks = self.masks.cuda(device)

    @property
    def inputs(self):
        return self.tokens, self.masks