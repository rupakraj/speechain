import torch

from speechain.infer_func.ctc_decoding import CTCPrefixScorer  # :contentReference[oaicite:2]{index=2}
from speechain.utilbox.train_util import make_len_from_mask

eps = 1e-20
minus_inf = -1e20


class BeamHypothesesSingle:
    """N-best container for a single utterance."""

    def __init__(self, beam_size: int, max_length: int, length_penalty: float):
        self.max_length = max_length - 1  # ignoring <sos>
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        score = sum_logprobs / ((len(hyp) + eps) ** self.length_penalty)
        if len(self) < self.beam_size or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self.beams) > self.beam_size:
                sorted_scores = sorted((s, i) for i, (s, _) in enumerate(self.beams))
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(self.worst_score, score)

    def is_done(self, best_sum_logprobs: float, curr_len: int):
        if len(self) < self.beam_size:
            return False
        curr_score = best_sum_logprobs / ((curr_len + eps) ** self.length_penalty)
        return curr_score < self.worst_score


def beam_search_single(
    enc_feat: torch.Tensor,          # The encoder output features, shape: (T, D) or (1, T, D)
    enc_feat_mask: torch.Tensor,     # The mask for encoder features, shape: (1, T) or (1, 1, T)
    asr_decode_fn,                   # The ASR decoder function (autoregressive), returns tuple where logits is [0]
    vocab_size: int,                 # The size of the vocabulary
    sos_eos: int = None,             # The index of the Start-Of-Sentence / End-Of-Sentence token
    padding_idx: int = 0,            # The index of the padding token
    beam_size: int = 5,              # The beam width for beam search
    min_f2t_ratio: float | int = 3.0, # The minimum ratio of feature length to text length (or absolute max length if negative)
    length_penalty: float = 1.0,     # The length penalty factor (alpha) for score normalization
    temperature: float = 1.0,        # The temperature for the ASR decoder softmax
    eos_filtering: bool = False,     # Whether to apply EOS filtering heuristic
    eos_threshold: float = 1.5,      # The threshold multiplier for EOS filtering
    ctc_weight: float = 0.0,         # The weight for CTC score interpolation (0.0 to disable)
    ctc_decode_fn=None,              # The CTC decoder function (e.g. ctc_layer), takes enc_feat
    ctc_temperature: float = 1.0,    # The temperature for the CTC softmax
    lm_weight: float = 0.0,          # The weight for LM score interpolation (0.0 to disable)
    lm_temperature: float = 1.0,     # The temperature for the LM softmax
    lm_decode_fn=None,               # The LM decoder function, returns tuple where logits is [0]
    lm_window_size: int | None = None # The context window size for the LM (None for full context)
):
    """
    Beam Search for single file inference.
    """

    assert beam_size >= 1 , "beam_size must be >= 1."
    assert temperature > 0 and ctc_temperature > 0 and lm_temperature > 0
    assert ctc_weight >= 0 and lm_weight >= 0
    assert beam_size + 1 <= vocab_size, "beam_size too large for vocab_size."

    device = enc_feat.device
    if sos_eos is None:
        sos_eos = vocab_size - 1

    # ---- shape normalize ----
    # enc_feat: (1, T, D)
    if enc_feat.dim() == 2:
        enc_feat = enc_feat.unsqueeze(0)
    # enc_feat_mask: (1, 1, T)
    if enc_feat_mask.dim() == 2:
        enc_feat_mask = enc_feat_mask.unsqueeze(1)
    elif enc_feat_mask.dim() != 3:
        raise ValueError("enc_feat_mask must be (1,T) or (1,1,T).")

    feat_maxlen = enc_feat.size(1)
    hypo_maxlen = int(feat_maxlen / min_f2t_ratio) if min_f2t_ratio > 0 else int(-min_f2t_ratio)
    hypo_maxlen = max(hypo_maxlen, 1)

    # repeat encoder outputs for beams: (beam, T, D) and (beam, 1, T)
    enc_feat = enc_feat.repeat(beam_size, 1, 1).contiguous()
    enc_feat_mask = enc_feat_mask.repeat(beam_size, 1, 1).contiguous()

    # ---- optional CTC scorer init ----
    ctc_scorer = None
    ctc_memory = None
    if ctc_decode_fn is not None and ctc_weight > 0:
        ctc_logits = ctc_decode_fn(enc_feat)  # (beam, T, V)
        ctc_logits[:, :, sos_eos] = minus_inf  # no sos/eos in CTC
        ctc_scorer = CTCPrefixScorer(
            x=torch.log_softmax(ctc_logits / ctc_temperature, dim=-1),
            enc_lens=make_len_from_mask(enc_feat_mask),  # (beam,)
            batch_size=1,
            beam_size=beam_size,
            blank_index=padding_idx,
            eos_index=sos_eos,
        )

    # ---- init beams ----
    finished = BeamHypothesesSingle(beam_size=beam_size, max_length=hypo_maxlen, length_penalty=length_penalty)

    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0

    hypo_text = torch.full((beam_size, 1), sos_eos, dtype=torch.long, device=device)  # includes leading <sos>
    hypo_text_len = torch.ones((beam_size,), dtype=torch.long, device=device)

    done = False

    # autoregressive decoding loop terminates when all beams are finished or max length is reached
    #        (not done) is convergence check
    #        (hypo_text_len.max().item() < hypo_maxlen) is max length check, it prevents the infinite loop
    while (not done) and (hypo_text_len.max().item() < hypo_maxlen):
        # ---- ASR decoder step ----
        dec_out = asr_decode_fn(
            enc_feat=enc_feat,
            enc_feat_mask=enc_feat_mask,
            text=hypo_text,
            text_len=hypo_text_len,
        )[0].detach()  # (beam, L, V)

        next_token_scores = torch.log_softmax(dec_out[:, -1, :] / temperature, dim=-1)  # (beam, V)

        # ---- CTC fusion ----
        if ctc_scorer is not None:
            next_token_scores[:, padding_idx] = minus_inf  # block blank in fusion
            ctc_token_scores, ctc_memory = ctc_scorer.forward_step(g=hypo_text[:, 1:], state=ctc_memory)
            next_token_scores = (1 - ctc_weight) * next_token_scores + ctc_weight * ctc_token_scores

        # ---- LM fusion ----
        if lm_decode_fn is not None and lm_weight > 0:
            # --- LM Integration ---
            lm_text = hypo_text if lm_window_size is None else hypo_text[:, -lm_window_size:]
            lm_len = hypo_text_len if lm_window_size is None else torch.clamp(hypo_text_len, max=lm_window_size)

            lm_out = lm_decode_fn(text=lm_text, text_len=lm_len)[0].detach()  # (beam, L, V)
            lm_scores = torch.log_softmax(lm_out[:, -1, :] / lm_temperature, dim=-1)
            next_token_scores = next_token_scores + lm_weight * lm_scores

        # total scores for all expansions: (beam, V)
        total_scores = next_token_scores + beam_scores.unsqueeze(-1)

        # pick top 2*beam among beam*V candidates
        flat_scores = total_scores.view(-1)  # (beam*V,)
        topk_scores, topk_ids = torch.topk(flat_scores, k=2 * beam_size, largest=True, sorted=True)

        next_beam = []
        for rank in range(topk_ids.numel()):
            flat_id = topk_ids[rank]
            src_beam = torch.div(flat_id, vocab_size, rounding_mode="floor")
            tok = flat_id % vocab_size
            score = topk_scores[rank]

            src_beam_i = int(src_beam.item())
            tok_i = int(tok.item())

            # EOS handling
            if tok_i == sos_eos:
                if rank >= beam_size:
                    continue

                if eos_filtering:
                    eos_score = next_token_scores[src_beam_i, sos_eos]
                    ref_score = next_token_scores[src_beam_i, torch.arange(vocab_size, device=device) != sos_eos].max()
                    if eos_score <= eos_threshold * ref_score:
                        continue

                finished.add(hyp=hypo_text[src_beam_i, 1:].detach().cpu(), sum_logprobs=float(score.item()))
            else:
                next_beam.append((score, tok, src_beam))

            if len(next_beam) == beam_size:
                break

        if len(next_beam) == 0:
            break

        # update done
        best_sum_logprobs = float(topk_scores.max().item())
        curr_len = int(hypo_text_len.max().item() - 1)
        done = finished.is_done(best_sum_logprobs=best_sum_logprobs, curr_len=curr_len)

        # commit next beams
        new_scores = torch.stack([x[0] for x in next_beam]).to(device)
        new_toks = torch.stack([x[1] for x in next_beam]).to(device)
        new_src = torch.stack([x[2] for x in next_beam]).to(device)

        hypo_text = torch.cat([hypo_text[new_src], new_toks.unsqueeze(1)], dim=1)
        hypo_text_len = torch.sum(hypo_text != padding_idx, dim=-1)
        beam_scores = new_scores

        # align encoder and ctc memory with new beam order
        enc_feat = enc_feat[new_src]
        enc_feat_mask = enc_feat_mask[new_src]
        if ctc_memory is not None:
            # token_idx shape must be (batch=1, beam)
            token_idx = new_toks.view(1, beam_size)
            ctc_memory = ctc_scorer.permute_mem(ctc_memory, new_src, token_idx)

    # add unfinished beams if needed
    for b in range(beam_size):
        finished.add(hyp=hypo_text[b, 1:].detach().cpu(), sum_logprobs=float(beam_scores[b].item()))

    # pick best
    best_score, best_hyp = sorted(finished.beams, key=lambda x: x[0])[-1]

    best_hyp = best_hyp.to(device)
    best_len = torch.tensor([len(best_hyp)], dtype=torch.long, device=device)

    # encoder length for ratio
    enc_len = enc_feat_mask[0].squeeze(0).sum().to(best_len.dtype)  # (T,) -> scalar
    ratio = enc_len / (best_len.to(enc_len.dtype) + 1e-10)

    return {
        "hypo_text": best_hyp.unsqueeze(0),          # (1, L)
        "hypo_text_len": best_len,                   # (1,)
        "feat_token_len_ratio": ratio,               # (1,)
        "hypo_text_confid": torch.tensor([best_score], dtype=torch.float),
    }
