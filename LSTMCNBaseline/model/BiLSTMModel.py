from module.NewLSTM import *
from module.Attention import *
from module.CPUEmbedding import *

class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(BiLSTMModel, self).__init__()
        self.config = config
        vocab_size, word_dims = pretrained_embedding.shape
        if vocab.vocab_size != vocab_size:
            print("word vocab size does not match, check word embedding file")
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

        self.lstm = NewLSTM(
            input_size=word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )
        self.sent_dim = word_dims
        self.trans = NonLinear(input_size=2 * config.lstm_hiddens, hidden_size=self.sent_dim, activation=nn.Tanh())
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.ext_dim = vocab.embedding_dim
        self.proj = NonLinear(self.sent_dim + self.ext_dim, vocab.tag_size)

    def forward(self, inputs):
        ##unpack inputs
        src_words, src_reprs, src_masks, tgt_words, tgt_reprs, tgt_masks = inputs
        src_embed = self.word_embed(src_words)
        tgt_embed = self.word_embed(tgt_words)

        if self.training:
            src_embed = drop_input_independent(src_embed, self.config.dropout_emb)
            tgt_embed = drop_input_independent(tgt_embed, self.config.dropout_emb)

        batch_size = src_embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)

        src_lstm_hiddens, src_state = self.lstm(src_embed, src_masks, None)
        src_hiddens = self.trans(src_lstm_hiddens.transpose(1, 0))
        if self.training:
            src_hiddens = drop_input_independent(src_hiddens, self.config.dropout_mlp)
        src_sent_probs = self.atten(atten_guide, src_hiddens, src_masks)
        batch_size, srclen, dim = src_hiddens.size()
        src_sent_probs = src_sent_probs.view(batch_size, srclen, -1)
        src_represents = src_hiddens * src_sent_probs
        src_represents = src_represents.sum(dim=1)
        src_final_represents = torch.cat([src_represents, src_reprs], 1)

        tgt_lstm_hiddens, tgt_state = self.lstm(tgt_embed, tgt_masks, None)
        tgt_hiddens = self.trans(tgt_lstm_hiddens.transpose(1, 0))
        if self.training:
            tgt_hiddens = drop_input_independent(tgt_hiddens, self.config.dropout_mlp)
        tgt_sent_probs = self.atten(atten_guide, tgt_hiddens, tgt_masks)
        batch_size, tgtlen, dim = tgt_hiddens.size()
        tgt_sent_probs = tgt_sent_probs.view(batch_size, tgtlen, -1)
        tgt_represents = tgt_hiddens * tgt_sent_probs
        tgt_represents = tgt_represents.sum(dim=1)
        tgt_final_represents = torch.cat([tgt_represents, tgt_reprs], 1)

        diff_hidden = torch.abs(src_final_represents - tgt_final_represents)
        outputs = self.proj(diff_hidden)
        return outputs

    def code2vec(self, tokens, reprs, token_masks):
        token_embed = self.word_embed(tokens)

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
        final_represents = torch.cat([token_represents, reprs], 1)

        return final_represents

    def classify(self, inputs1, inputs2):
        diff_hidden = torch.abs(inputs1 - inputs2)
        outputs = self.proj(diff_hidden)
        return outputs
