class CloneDetection(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, inputs):
        tokens, masks = inputs
        self.model.eval()
        represents = self.model.code2vec(tokens, masks)
        # cache
        return represents
