from module.NewLSTM import *
from module.CPUEmbedding import *

class TagClassification(nn.Module):
    def __init__(self, code_dim, tag_size, drop_value):
        super(TagClassification, self).__init__()
        self.code_dim = code_dim
        self.drop = nn.Dropout(p=drop_value)
        self.proj = NonLinear(self.code_dim, tag_size)

    def forward(self, inputs1, inputs2):
        diff_vec = torch.abs(inputs1 - inputs2)
        hidden = self.drop(diff_vec * diff_vec)
        outputs = self.proj(hidden)
        return outputs

