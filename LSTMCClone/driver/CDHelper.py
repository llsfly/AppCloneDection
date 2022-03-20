import torch.nn.functional as F

class CloneDetection(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, inputs):
        src_reprs, tgt_reprs = inputs
        tag_logits = self.model(src_reprs, tgt_reprs)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        loss = F.cross_entropy(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, inputs):
        if inputs[0] is not None:
            self.forward(inputs)
        probs = F.softmax(self.tag_logits, dim=-1)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        probs = probs.detach().cpu().numpy()
        return pred_tags, probs
