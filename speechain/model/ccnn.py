import torch
import torch.nn as nn


class CCNN(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, lengths, cand_ids):
        """
            x: (B,T)
            lengths: (B,)
            cand_ids: (B,T,K)
            returns lm_logits (B,T,V), cons_logits (B,T,K)
        """
        B, T = x.shape

        x_emb = self.dropout(self.emb(x))

        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )
        h = self.dropout(h)

        lm_logits = self.lm_head(h)

        if cand_ids is None:
            cons_logits = torch.matmul(h, self.emb.weight.t())  # (B,T,V)
        else:
            cand_emb = self.emb(cand_ids)      # (B,T,K,D)
            cons_logits = (h.unsqueeze(2) * cand_emb).sum(dim=-1)

        return lm_logits, cons_logits


        # # The constituency model is coded correctly as a neural scoring head
        # # h is learned prefix state from lstm
        # # cand_emb is learned embedding from each candidate token
        # # cons_logits is dot product score between prefix state and candidate embedding
        # cand_emb = self.emb(cand_ids)      # (B,T,K,D)
        # h_exp = h.unsqueeze(2)             # (B,T,1,D)
        # cons_logits = (h_exp * cand_emb).sum(dim=-1)  # (B,T,K)

        # return lm_logits, cons_logits
