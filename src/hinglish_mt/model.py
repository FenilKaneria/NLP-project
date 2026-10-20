"""GRU encoder-decoder with additive attention (Bahdanau et al. 2015).

One embedding table is shared by the source, the target and the output layer.
With copy=True the decoder can also copy word pieces from the source
(pointer-generator, See et al. 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hinglish_mt.tokenise import PAD


class Attention(nn.Module):
    """score = v . tanh(W1 * encoder state + W2 * decoder state)"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, keys, mask):
        """`keys` is encoder_proj(encoder states). Returns one weight per source position."""
        query = self.decoder_proj(decoder_state).unsqueeze(1)
        scores = self.v(torch.tanh(keys + query)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=-1)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, layers=2, dropout=0.3, copy=False):
        super().__init__()
        self.args = {
            "vocab_size": vocab_size,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "dropout": dropout,
            "copy": copy,
        }
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.copy = copy

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(emb_dim, hidden_dim, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.decoder = nn.GRU(emb_dim + 2 * hidden_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.pre_output = nn.Linear(3 * hidden_dim + emb_dim, emb_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        if copy:
            self.copy_gate = nn.Linear(emb_dim, 1)

        # Small starting values, because the embedding table is also the output layer.
        nn.init.normal_(self.embedding.weight, std=emb_dim**-0.5)

    def encode(self, src, src_len):
        """Returns encoder states, attention keys, padding mask and the decoder's first hidden state."""
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.encoder(packed)
        memory, _ = pad_packed_sequence(output, batch_first=True, total_length=src.size(1))

        # hidden is (layers * 2, batch, hidden). Join the forward and backward state of each layer.
        hidden = hidden.view(self.layers, 2, -1, self.hidden_dim)
        forward_state = hidden[:, 0]
        backward_state = hidden[:, 1]
        both = torch.cat([forward_state, backward_state], dim=-1)
        decoder_hidden = torch.tanh(self.bridge(both))

        keys = self.attention.encoder_proj(memory)
        mask = src != PAD
        return memory, keys, mask, decoder_hidden

    def step(self, previous_word, hidden, memory, keys, mask):
        """One decoder step. Returns the output state, the new hidden state and the attention weights."""
        embedded = self.dropout(self.embedding(previous_word))
        weights = self.attention(hidden[-1], keys, mask)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)

        decoder_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, hidden = self.decoder(decoder_input, hidden)
        output = output.squeeze(1)

        state = self.pre_output(torch.cat([output, context, embedded], dim=-1))
        return state, hidden, weights

    def project(self, state):
        """Output state to a score for every vocabulary entry."""
        return F.linear(self.dropout(state), self.embedding.weight, self.output_bias)

    def word_scores(self, state, weights, src):
        """Log-probability of every vocabulary entry.

        With copy on, the model can also take a word piece straight from the
        source. The attention weights say which source piece, and a gate
        between 0 and 1 says how much to generate and how much to copy.
        """
        logits = self.project(state).float()
        if not self.copy:
            return F.log_softmax(logits, dim=-1)

        generated = F.softmax(logits, dim=-1)
        generate_share = torch.sigmoid(self.copy_gate(state).float())
        copied = torch.zeros_like(generated)
        copied.scatter_add_(-1, src, weights.float())
        mixed = generate_share * generated + (1 - generate_share) * copied
        return torch.log(mixed + 1e-9)

    def forward(self, src, src_len, tgt_in):
        """Log-probabilities for every target position, given the correct previous words."""
        memory, keys, mask, hidden = self.encode(src, src_len)
        states = []
        all_weights = []
        for t in range(tgt_in.size(1)):
            previous_word = tgt_in[:, t]
            state, hidden, weights = self.step(previous_word, hidden, memory, keys, mask)
            states.append(state)
            all_weights.append(weights)
        states = torch.stack(states, dim=1)
        all_weights = torch.stack(all_weights, dim=1)

        # One copy of the source ids for every target position.
        src_per_step = src.unsqueeze(1).expand(-1, tgt_in.size(1), -1)
        return self.word_scores(states, all_weights, src_per_step)

    def n_parameters(self):
        total = 0
        for parameter in self.parameters():
            total += parameter.numel()
        return total
