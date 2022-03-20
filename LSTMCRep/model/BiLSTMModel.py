from module.NewLSTM import *
from module.Attention import *
from module.CPUEmbedding import *

class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, token_vec, word_vec):
        super(BiLSTMModel, self).__init__()
        self.config = config
        vocab_size, token_dims = token_vec.shape
        if vocab.vocab_size != vocab_size:
            print("token vocab size does not match, check token embedding file")
        self.token_embed = CPUEmbedding(vocab.vocab_size, token_dims, padding_idx=vocab.PAD)
        self.token_embed.weight.data.copy_(torch.from_numpy(token_vec))
        self.token_embed.weight.requires_grad = False

        vocab_size, word_dims = word_vec.shape
        if vocab.word_size != vocab_size:
            print("word vocab size does not match, check word embedding file")
        self.word_embed = CPUEmbedding(vocab.word_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_vec))
        self.word_embed.weight.requires_grad = False

        self.lstm = NewLSTM(
            input_size=token_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )
        self.sent_dim = word_dims
        self.trans = NonLinear(input_size=2*config.lstm_hiddens, hidden_size=self.sent_dim, activation=nn.Tanh())
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)


    def code2vec(self, tokens, token_masks):
        token_embed = self.token_embed(tokens)

        batch_size = token_embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)

        token_lstm_hiddens, token_state = self.lstm(token_embed, token_masks, None)
        token_hiddens = self.trans(token_lstm_hiddens.transpose(1, 0))
        token_sent_probs = self.atten(atten_guide, token_hiddens, token_masks)
        batch_size, srclen, dim = token_hiddens.size()
        token_sent_probs = token_sent_probs.view(batch_size, srclen, -1)
        token_represents = token_hiddens * token_sent_probs
        token_represents = token_represents.sum(dim=1)

        return token_represents

