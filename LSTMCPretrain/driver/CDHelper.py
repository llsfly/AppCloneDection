import torch.nn as nn


class CloneDetection(object):
    def __init__(self, model, vocab, loss_margin):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.loss = nn.CosineEmbeddingLoss(margin=loss_margin)

    def forward(self, inputs):
        # if self.use_cuda:
        #     xlen = len(inputs)
        #     for idx in range(xlen):
        #         inputs[idx] = inputs[idx].cuda(self.device)
        token_represents, name_represents = self.model(inputs)
        # cache
        self.token_represents = token_represents
        self.name_represents = name_represents

    def compute_loss(self, true_tags):
        loss = self.loss(self.token_represents, self.name_represents, true_tags)
        return loss

    def classifier(self, inputs):
        if inputs[0] is not None:
            self.forward(inputs)
        return self.token_represents, self.name_represents
