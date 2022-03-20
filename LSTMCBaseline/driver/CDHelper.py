import torch.nn.functional as F
from torch.autograd import Variable

class CloneDetection(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, inputs):
        tag_logits = self.model(inputs)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        true_tags = Variable(true_tags, requires_grad=False)
        # if self.use_cuda: true_tags = true_tags.cuda()
        loss = F.cross_entropy(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def code2vec(self, inputs):
        tokens, masks = inputs
        represents = self.model.code2vec(tokens, masks)
        # cache
        return represents


    def classifier(self, inputs):
        inputs1, inputs2 = inputs
        tag_logits = self.model.classify(inputs1, inputs2)
        probs = F.softmax(tag_logits, dim=-1)
        pred_tags = tag_logits.detach().max(1)[1].cpu()
        probs = probs.detach().cpu().numpy()
        return pred_tags, probs
