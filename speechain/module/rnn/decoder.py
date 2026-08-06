"""
RNN-based decoder for Speechain
Author: Rupak Raj (with help of AI)
Affiliation: Speechain Project
Date: 2026.02
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from speechain.module.abs import Module


class RNNDecoderLayer(Module):
    """Single RNN decoder layer with attention mechanism"""

    def module_init(
        self,
        d_model: int = 512,
        rnn_type: str = "LSTM",
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = False,
        att_num_heads: int = 8,
        att_dropout: float = 0.1,
        res_dropout: float = 0.1,
        layernorm_first: bool = True,
    ):
        """
        Args:
            d_model: Feature dimension
            rnn_type: Type of RNN ('LSTM', 'GRU', 'RNN')
            rnn_hidden_size: Hidden size of RNN
            rnn_num_layers: Number of RNN layers
            rnn_dropout: Dropout between RNN layers
            rnn_bidirectional: Whether to use bidirectional RNN
            att_num_heads: Number of attention heads
            att_dropout: Attention dropout rate
            res_dropout: Residual connection dropout
            layernorm_first: Whether to apply layernorm before sublayers
        """
        self.d_model = d_model
        self.layernorm_first = layernorm_first

        # RNN layer
        rnn_class = getattr(nn, rnn_type)
        self.rnn = rnn_class(
            input_size=d_model,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
            batch_first=True,
        )

        # Calculate RNN output size
        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)

        # Project RNN output back to d_model if needed
        self.rnn_proj = nn.Linear(rnn_output_size, d_model) if rnn_output_size != d_model else nn.Identity()

        # Cross-attention layer (encoder-decoder attention)
        from speechain.module.transformer.attention import MultiHeadedAttention
        self.attention = MultiHeadedAttention(
            num_heads=att_num_heads,
            d_model=d_model,
            dropout=att_dropout,
        )

        # Feed-forward layer
        from speechain.module.transformer.feed_forward import PositionwiseFeedForward
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            fdfwd_dim=d_model * 4,
            fdfwd_activation="ReLU",
            dropout=0.1,
        )

        # Layer normalization
        self.rnn_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.att_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_ln = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout layers
        self.dropout = nn.Dropout(res_dropout)

    # dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(
    #             src=enc_feat, src_mask=enc_feat_mask, tgt=emb_text, tgt_mask=text_mask
    #         )
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        hidden_state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            src: (batch, src_len, d_model) Source features
            src_mask: (batch, 1, src_len) Source mask
            tgt: (batch, tgt_len, d_model) Target embeddings
            hidden_state: Previous hidden state for RNN

        Returns:
            output: (batch, tgt_len, d_model) Layer output
            attention_weights: (batch, tgt_len, src_len) Attention weights
            hidden_state: New hidden state for RNN
        """
        # RNN processing
        rnn_norm = self.rnn_ln(tgt) if self.layernorm_first else tgt
        rnn_output, new_hidden = self.rnn(rnn_norm, hidden_state)
        rnn_output = self.rnn_proj(rnn_output)
        rnn_output = self.dropout(rnn_output) + tgt
        rnn_output = self.rnn_ln(rnn_output) if not self.layernorm_first else rnn_output

        # Cross-attention
        att_norm = self.att_ln(rnn_output) if self.layernorm_first else rnn_output
        att_output, att_weights = self.attention(
            src, src, att_norm, mask=src_mask
        )
        att_output = self.dropout(att_output) + rnn_output
        att_output = self.att_ln(att_output) if not self.layernorm_first else att_output

        # Feed-forward
        ff_norm = self.ff_ln(att_output) if self.layernorm_first else att_output
        ff_output = self.feed_forward(ff_norm)
        ff_output = self.dropout(ff_output) + att_output
        # if tgt_mask is not None:
        #     ff_output = ff_output.masked_fill(tgt_mask, 0.0)
        ff_output = self.ff_ln(ff_output) if not self.layernorm_first else ff_output

        return ff_output, att_weights, new_hidden


class RNNDecoder(Module):
    """Multi-layer RNN decoder with attention"""

    def module_init(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        rnn_type: str = "LSTM",
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.1,
        rnn_bidirectional: bool = False,
        att_num_heads: int = 8,
        att_dropout: float = 0.1,
        res_dropout: float = 0.1,
        layernorm_first: bool = True,
    ):
        """
        Args:
            num_layers: Number of decoder layers
            d_model: Feature dimension
            rnn_type: Type of RNN ('LSTM', 'GRU', 'RNN')
            rnn_hidden_size: Hidden size of RNN
            rnn_num_layers: Number of RNN layers per decoder layer
            rnn_dropout: Dropout between RNN layers
            rnn_bidirectional: Whether to use bidirectional RNN
            att_num_heads: Number of attention heads
            att_dropout: Attention dropout rate
            res_dropout: Residual connection dropout
            layernorm_first: Whether to apply layernorm before sublayers
        """
        # Set input/output sizes
        if self.input_size is not None:
            d_model = self.input_size
        self.output_size = d_model

        self.num_layers = num_layers
        self.d_model = d_model

        # Create decoder layers
        self.layers = nn.ModuleList([
            RNNDecoderLayer(
                d_model=d_model,
                rnn_type=rnn_type,
                rnn_hidden_size=rnn_hidden_size,
                rnn_num_layers=rnn_num_layers,
                rnn_dropout=rnn_dropout,
                rnn_bidirectional=rnn_bidirectional,
                att_num_heads=att_num_heads,
                att_dropout=att_dropout,
                res_dropout=res_dropout,
                layernorm_first=layernorm_first,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        if layernorm_first:
            self.final_ln = nn.LayerNorm(d_model, eps=1e-6)

    # dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(
    #             src=enc_feat, src_mask=enc_feat_mask, tgt=emb_text, tgt_mask=text_mask
    #         )
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        return_att: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, list, list, list]:
        """
        Args:
            tgt: (batch, tgt_len, d_model) Target embeddings
            src: (batch, src_len, d_model) Source features
            src_mask: (batch, 1, src_len) Source mask
            return_att: Whether to return attention weights
            return_hidden: Whether to return hidden states

        Returns:
            output: (batch, tgt_len, d_model) Decoder output
            self_attmat: List of self-attention weights per layer
            encdec_attmat: List of encoder-decoder attention weights per layer
            hidden_states: List of hidden states per layer
        """
        batch_size = tgt.size(0)
        hidden_state = None

        attention_weights = []
        hidden_states = []

        # Pass through each layer
        for layer in self.layers:
            tgt, att_weight, hidden_state = layer(
                    tgt=tgt, src=src, src_mask=src_mask, hidden_state=hidden_state
            )

            if return_att:
                attention_weights.append(att_weight)
            if return_hidden:
                hidden_states.append(tgt.clone())

        # Final layer norm
        if hasattr(self, 'final_ln'):
            tgt = self.final_ln(tgt)

        if not attention_weights:
            dummy_att = torch.zeros(
                tgt.size(0),  # batch
                1,            # 1 head for dummy
                tgt.size(1),  # tgt_len
                src.size(1),  # src_len
                device=tgt.device,
                dtype=torch.bool
            )
            attention_weights = [dummy_att]

        # dec_feat, self_attmat, encdec_attmat, hidden = self.decoder(
        return tgt, None, attention_weights, hidden_states
