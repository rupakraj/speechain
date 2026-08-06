import math
import importlib
from typing import Dict, Optional

import numpy as np
import torch

from speechain.criterion.abs import Criterion
from speechain.utilbox.train_util import make_mask_from_len


def _import_from_path(path: str):
    """
    path format: "package.module:Symbol"
    Example: "myccnn.model:CCNN"
    """
    if ":" not in path:
        raise ValueError(f"Invalid import path '{path}'. Use 'module:Symbol'.")
    mod, sym = path.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, sym)


class CrossEntropy_CCNN(Criterion):
    """This criterion calculates the cross entropy between model predictions and target
    labels and smoothed by CCNN.
        Replaces uniform smoothing term:
            eps/V * sum_i log p(i)
        with CCNN-weighted expectation:
            eps * sum_i q_ccnn(i|prefix) * log p(i)

        CCNN is used in teacher forcing mode:
            ccnn_in[:, 0]  = BOS
            ccnn_in[:, 1:] = text[:, :-1]
    """

    def criterion_init(
        self,
        length_normalized: bool = False,
        label_smoothing: float = 0.0,
        temperature: float = 1.0,
        confid_threshold: float = 0.0,
        confid_level: str = "sentence",
        token_vocab: str = None,
        new_weights: Dict = None,

        # ---- CCNN params ----
        ccnn_import: Optional[str] = None,      # "myccnn.model:CCNN"
        ccnn_ckpt: Optional[str] = None,        # "/path/to/ccnn.pt" or state_dict
        ccnn_bos_id: Optional[int] = None,      # BOS id in shared tokenizer
        ccnn_pad_id: Optional[int] = None,      # PAD id in shared tokenizer
        ccnn_head: str = "lm",                  # "lm" or "cons" or "mix"
        ccnn_lm_temperature: float = 1.0,          # soften CCNN distribution
        ccnn_cons_temperature: float = 2.0,          # soften CCNN distribution
        ccnn_cons_beta: float = 0.5,             # weight for cons logits in fusion .3 ~ 0.8 (start at 0.5)
        ccnn_mix_alpha: float = 1.0,            # 1.0 = pure CCNN, 0.0 = uniform
        ccnn_d_model: int = 256,                # optional
        ccnn_n_layers: int = 2,                 # optional
        ccnn_dropout: float = 0.1,              # optional

    ):
        """

        Args:
            length_normalized: bool
                Controls whether the sentence normalization is performed.
            label_smoothing: float
                Controls the scale of label smoothing. 0 means no smoothing.
            temperature: float
                Controls the temperature of the Softmax operation.
            confid_threshold: float
                Controls whether to ignore the prediction lower than the threshold for loss calculation.
            confid_level: str
                The level of confidence calculation. Either 'token' (token_level confidence) or 'sent' (sentence-level
                confidence). Default to be 'sentence'.
            token_vocab: str
                The path of the given token vocabulary list. Necessary if new_weights is not None.
            new_weights: Dict
                The customized token weights to calculate the cross entropy. Must be given in the format below:
                'new_weights:
                    token1: weight1
                    token2: weight2
                    ...'
        """

        assert (
            0 <= label_smoothing < 1.0
        ), f"The value of label_smoothing should be a float number in [0, 1), but got {label_smoothing}!"
        assert (
            temperature >= 0.0
        ), f"The value of temperature should be a non-negative float number, but got {temperature}"
        assert (
            0 <= confid_threshold < 1.0
        ), f"The value of confid_threshold should be a float number in [0, 1), but got {label_smoothing}!"
        assert confid_level in [
            "sentence",
            "token",
        ], f"confid_level must be one of ['sentence', 'token'], but got {confid_level}"

        # para recording
        self.length_normalized = length_normalized
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.confid_threshold = confid_threshold
        self.confid_level = confid_level
        self.token_weights = None

        # update the token weights if new_weights is given
        if new_weights is not None:
            assert (
                token_vocab is not None
            ), "Please specify a token dictionary by 'token_vocab' if you want to customize the token weights."

            token_dict = np.loadtxt(token_vocab, dtype=str, delimiter="\n")
            token_dict = dict(zip(token_dict, np.arange(0, token_dict.shape[0])))
            self.token_weights = torch.ones(len(token_dict)).cuda().detach()

            for token, weight in new_weights.items():
                self.token_weights[token_dict[token]] = weight


        # --- CCNN wiring - Starts ---
        self.ccnn = None
        self.ccnn_bos_id = ccnn_bos_id
        self.ccnn_pad_id = None

        self.ccnn_head = ccnn_head
        self.ccnn_lm_temperature = float(ccnn_lm_temperature)
        self.ccnn_cons_temperature = float(ccnn_cons_temperature)
        self.ccnn_cons_beta = float(ccnn_cons_beta)
        self.ccnn_mix_alpha = float(ccnn_mix_alpha)

        self.ccnn_d_model = ccnn_d_model
        self.ccnn_n_layers = ccnn_n_layers
        self.ccnn_dropout = ccnn_dropout

        if self.label_smoothing > 0:
            # If you want CCNN smoothing, require import + ckpt + bos_id
            if ccnn_import is not None:
                assert ccnn_ckpt is not None, "ccnn_ckpt is required when ccnn_import is set."
                assert ccnn_bos_id is not None, "ccnn_bos_id is required when ccnn_import is set."
                assert ccnn_pad_id is not None, "ccnn_pad_id is required for CCNN padding_idx."
                assert self.ccnn_head in ["lm", "cons", "mix"], "ccnn_head must be 'lm', 'cons' or 'mix"

                # allow overriding CCNN ctor hyperparams from YAML if you pass them
                self.ccnn_d_model = int(getattr(self, "ccnn_d_model", 256))
                self.ccnn_n_layers = int(getattr(self, "ccnn_n_layers", 2))
                self.ccnn_dropout = float(getattr(self, "ccnn_dropout", 0.1))
                self.ccnn_pad_id = int(ccnn_pad_id)

                CCNNClass = _import_from_path(ccnn_import)
                ckpt = torch.load(ccnn_ckpt, map_location="cpu", weights_only=False)
                # allow either {"state_dict": ...} or direct state_dict
                state = ckpt["model"]

                self.ccnn = CCNNClass(
                    vocab_size=66, # TODO : automate this later
                    pad_id=self.ccnn_pad_id,
                    d_model=self.ccnn_d_model,
                    n_layers=self.ccnn_n_layers,
                    dropout=self.ccnn_dropout,
                )
                self.ccnn.load_state_dict(state)

                self.ccnn.eval()
                for p in self.ccnn.parameters():
                    p.requires_grad_(False)
        # --- CCNN wiring - Ends ---


    # --- CCNN wiring - Starts ---
    # @torch.no_grad()
    # def _ccnn_q(self, text: torch.Tensor, text_len: torch.Tensor, vocab_size: int) -> Optional[torch.Tensor]:
    #     """
    #     text: (B, T) targets aligned with logits
    #     returns q: (B, T, V) probability distribution for smoothing
    #     """
    #     if self.ccnn is None:
    #         return None

    #     device = text.device
    #     B, T = text.shape

    #     # teacher-forced CCNN input: BOS + shifted targets
    #     ccnn_in = torch.empty((B, T), dtype=torch.long, device=device)
    #     ccnn_in[:, 0] = int(self.ccnn_bos_id)
    #     if T > 1:
    #         ccnn_in[:, 1:] = text[:, :-1]

    #     lengths = text_len.to(device)

    #     # move CCNN to same device
    #     if next(self.ccnn.parameters()).device != device:
    #         self.ccnn.to(device)

    #     # dummy cand_ids to satisfy CCNN forward signature: (B,T,1)
    #     # Using pad_id is safe and cheap.
    #     cand_ids = torch.full((B, T, 1), int(self.ccnn_pad_id), dtype=torch.long, device=device)

    #     lm_logits, cons_logits = self.ccnn(ccnn_in, lengths, cand_ids)

    #     # choose head
    #     if self.ccnn_head == "lm":
    #         logits = lm_logits
    #     else:
    #         # cons_logits is (B,T,K) not (B,T,V) in your CCNN, so it cannot form q over vocab.
    #         # Therefore "cons" head cannot be used for smoothing distribution unless you redesign it.
    #         raise RuntimeError(
    #             "ccnn_head='cons' is not supported for label smoothing because cons_logits is (B,T,K), not (B,T,V). "
    #             "Use ccnn_head='lm'."
    #         )

    #     if logits.dim() != 3:
    #         raise RuntimeError(f"Expected CCNN lm_logits (B,T,V), got shape {tuple(logits.shape)}")

    #     if logits.size(-1) != vocab_size:
    #         raise RuntimeError(f"Vocab mismatch: CCNN V={logits.size(-1)} vs ASR V={vocab_size}")

    #     q = torch.softmax(logits / max(self.ccnn_temperature, 1e-6), dim=-1)

    #     # optional mixture with uniform
    #     if self.ccnn_mix_alpha < 1.0:
    #         uni = torch.full_like(q, 1.0 / vocab_size)
    #         q = self.ccnn_mix_alpha * q + (1.0 - self.ccnn_mix_alpha) * uni

    #     return q.detach()


    @torch.no_grad()
    def _ccnn_q(self, text: torch.Tensor, text_len: torch.Tensor, vocab_size: int) -> Optional[torch.Tensor]:
        """
        text: (B, T) targets aligned with logits
        returns q: (B, T, V) probability distribution for smoothing
        """
        if self.ccnn is None:
            return None

        device = text.device
        B, T = text.shape

        # teacher-forced CCNN input: BOS + shifted targets
        ccnn_in = torch.empty((B, T), dtype=torch.long, device=device)
        ccnn_in[:, 0] = int(self.ccnn_bos_id)
        if T > 1:
            ccnn_in[:, 1:] = text[:, :-1]

        lengths = text_len.to(device)

        # move CCNN to same device
        if next(self.ccnn.parameters()).device != device:
            self.ccnn.to(device)

        # all vocab as candidates -> cons_logits becomes (B,T,V)
        V = int(vocab_size)

        # calculated in the ccnn's forward itself
        # cand_ids = torch.arange(V, device=device, dtype=torch.long).view(1, 1, V).expand(B, T, V)

        lm_logits, cons_logits = self.ccnn(ccnn_in, lengths, None)  # (B,T,V), (B,T,V) <-- none will force

        # if lm_logits.dim() != 3 or cons_logits.dim() != 3:
        #     raise RuntimeError(
        #         f"Expected (B,T,V). Got lm={tuple(lm_logits.shape)}, cons={tuple(cons_logits.shape)}"
        #     )
        # if lm_logits.size(-1) != vocab_size or cons_logits.size(-1) != vocab_size:
        #     raise RuntimeError(
        #         f"Vocab mismatch: lmV={lm_logits.size(-1)}, consV={cons_logits.size(-1)} vs ASR V={vocab_size}"
        #     )

        # mask pad token (avoid giving smoothing mass to PAD)
        pad_id = int(self.ccnn_pad_id)
        lm_logits = lm_logits.clone()
        cons_logits = cons_logits.clone()
        lm_logits[..., pad_id] = -1e20
        cons_logits[..., pad_id] = -1e20

        # choose / fuse heads
        if self.ccnn_head == "lm":
            teacher_logits = lm_logits / max(self.ccnn_lm_temperature, 1e-6)

        elif self.ccnn_head == "cons":
            teacher_logits = cons_logits / max(self.ccnn_cons_temperature, 1e-6)

        elif self.ccnn_head == "mix":
            beta = float(self.ccnn_cons_beta)
            teacher_logits = (lm_logits / max(self.ccnn_lm_temperature, 1e-6)) + \
                            (beta * (cons_logits / max(self.ccnn_cons_temperature, 1e-6)))
        else:
            raise ValueError(f"Unknown ccnn_head={self.ccnn_head}")

        q = torch.softmax(teacher_logits, dim=-1)

        # optional mixture with uniform (keep your existing behavior)
        if self.ccnn_mix_alpha < 1.0:
            uni = torch.full_like(q, 1.0 / vocab_size)
            q = self.ccnn_mix_alpha * q + (1.0 - self.ccnn_mix_alpha) * uni

        return q.detach()
    # --- CCNN wiring - Ends ---


    def __call__(
        self, logits: torch.Tensor, text: torch.Tensor, text_len: torch.Tensor
    ):
        """

        Args:
            logits: (batch, text_maxlen, vocab_size)
                The model predictions for the text
            text: (batch, text_maxlen)
                The target text labels.
            text_len: (batch,)
                The text lengths

        Returns:
            The cross entropy between logits and text

        """
        # For the text attached by a <sos/eos> at the beginning
        if logits.size(1) == text.size(1) - 1:
            # text_len must match the sequence dimension of text
            assert text_len.max() == text.size(1), (
                f"There is a mismatch of the sentence length between text and text_len. "
                f"Expect text_len.max() is either equal to or 1 smaller than text.size(1), "
                f"but got text_len.max()={text_len.max()} and text.size(1)={text.size(1)}."
            )
            # # remove the <sos/eos> at the beginning
            # text = text[:, 1:].squeeze(dim=-1)
            # # don't use text_len -= 1 here because it will also change the text_len outside this function
            # text_len = text_len - 1
            # remove the <sos/eos> at the beginning
            text = text[:, 1:]
            # Only squeeze if there is an extra trailing singleton feature dim (B,T,1).
            if text.dim() == 3 and text.size(-1) == 1:
                text = text.squeeze(-1)

            text_len = text_len - 1
        # Otherwise, text must not have a <sos/eos> at the beginning (equal in length with logits)
        elif logits.size(1) != text.size(1):
            raise RuntimeError

        # reshape predictions and do log-softmax
        batch, seq_maxlen, vocab_size = logits.size()
        log_prob = torch.log_softmax(
            logits.contiguous().view(batch * seq_maxlen, vocab_size) / self.temperature,
            dim=-1,
        )

        # gather log p(y)
        flat_text = text.contiguous().view(-1)

        # reshape targets and calculate the loss
        log_prob_target = log_prob.gather(1, text.contiguous().view(-1, 1)).squeeze(
            dim=-1
        )

        # --- label smoothing ---
        if self.label_smoothing > 0:
            smooth_pos = 1.0 - self.label_smoothing

            # ----- ccnn integration -----
            # L = (1-ε) * log p(y_true) + ε * Σ_i q(i) * log p(i)
            q = self._ccnn_q(text=text, text_len=text_len, vocab_size=vocab_size)
            if q is None:
                # fallback to uniform smoothing
                smooth_neg = self.label_smoothing / vocab_size
                loss = (log_prob_target * smooth_pos) + (log_prob * smooth_neg).sum(dim=1)
            else:
                # compute from ccnn
                q_flat = q.contiguous().view(batch * seq_maxlen, vocab_size)
                # loss = (log_prob_target * smooth_pos) + (log_prob * (self.label_smoothing * q_flat)).sum(dim=1)
                # Formulae for loss computation: L = (1-ε) * log p(y_true) + ε * Σ_i q(i) * log p(i)
                loss = (log_prob_target * smooth_pos) + (log_prob * q_flat).sum(dim=1) * self.label_smoothing

            # ----- ccnn integration ends -----
        else:
            loss = log_prob_target

        # reweight each token in the calculated loss
        if self.token_weights is not None:
            loss = loss * self.token_weights.index_select(0, text.reshape(-1))

        # convert the text length into the bool masks
        text_mask = make_mask_from_len(text_len, return_3d=False)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # padding the extra part of each sentence by zeros
        loss_mask = ~text_mask.reshape(-1)
        if loss.is_cuda:
            loss_mask = loss_mask.cuda(loss.device)
        if self.confid_threshold > 0:
            # padding the token predictions whose token-level confidences are lower than the threshold
            if self.confid_level == "token":
                # confid_mask: (batch * seq_maxlen,)
                confid_mask = log_prob_target <= math.log(self.confid_threshold)
                loss_mask = torch.logical_or(loss_mask, confid_mask)
                # update text_len by confid_mask for normalization (mask the extra part of each sentence)
                # (batch * seq_maxlen,) -> (batch, seq_maxlen) -> (batch,)
                text_len = (
                    (~confid_mask.reshape(batch, seq_maxlen))
                    .masked_fill(~text_mask, False)
                    .sum(dim=-1)
                )
                # whether a sentence is valid (i.e. contains unmasked token prediction): (batch,)
                valid_sent = text_len > 0
            # padding the whole sentence predictions whose sentence-level confidences are lower than the threshold
            elif self.confid_level == "sentence":
                # sent_confid: (batch * seq_maxlen,) -> (batch, seq_maxlen) -> (batch, 1)
                sent_confid = (
                    log_prob_target.reshape(batch, seq_maxlen)
                    .masked_fill(~text_mask, 0.0)
                    .sum(dim=-1, keepdim=True)
                )
                # confid_mask: (batch, 1)
                confid_mask = sent_confid <= (
                    text_len * math.log(self.confid_threshold)
                ).unsqueeze(-1)
                # confid_mask: (batch, 1) -> (batch, seq_maxlen) -> (batch * seq_maxlen,)
                loss_mask = torch.logical_or(
                    loss_mask, confid_mask.expand(-1, seq_maxlen).reshape(-1)
                )
                # whether a sentence is valid (i.e. unmasked sentence): (batch, 1) -> (batch,)
                valid_sent = ~confid_mask.squeeze(-1)
            else:
                raise RuntimeError(
                    f"confid_level must be one of ['sentence', 'token'], but got {self.confid_level}"
                )
        else:
            valid_sent = None

        loss = loss.masked_fill(loss_mask, 0.0).reshape(batch, seq_maxlen).sum(dim=-1)

        # normalize the loss by the token sequence length if specified
        if self.length_normalized:
            loss /= text_len + 1e-10

        # valid_sent is used to calculate the number of valid sentence included in the loss
        return (
            -loss.mean()
            if self.confid_threshold == 0
            else -loss.sum() / (torch.sum(valid_sent) + 1e-10)
        )
