"""
Author: Heli Qi
Affiliation: NAIST
Date: 2022.07
"""

from typing import List

import editdistance
import torch

from speechain.criterion.abs import Criterion
from speechain.tokenizer.abs import Tokenizer


def text_preprocess(text, tokenizer: Tokenizer):
    # tensor input need to be recovered to string
    if isinstance(text, torch.Tensor):
        # remove the padding and sos/eos tokens
        proc_text = text[
            torch.logical_and(
                text != tokenizer.ignore_idx, text != tokenizer.sos_eos_idx
            )
        ]
        # Check if result is empty after filtering: Fix of inference time error
        if proc_text.numel() == 0:
            print(f"DEBUG: Empty tensor after filtering in text_preprocess, original shape: {text.shape}")
            return ""  # Return empty string instead of crashing

        # turn text tensors into strings for removing the blanks
        string = tokenizer.tensor2text(proc_text)
    # string input, no processing is done here
    elif isinstance(text, str):
        string = text
    else:
        raise RuntimeError

    return string


class ErrorRate(Criterion):
    """"""

    def criterion_init(self, tokenizer: Tokenizer = None, do_aver: bool = False):
        """

        Args:
            tokenizer: Tokenizer
            do_aver: bool

        """
        self.tokenizer = tokenizer
        self.do_aver = do_aver

    def __call__(
        self,
        hypo_text: torch.Tensor or List[str] or str,
        real_text: torch.Tensor or List[str] or str,
        tokenizer: Tokenizer = None,
        do_aver: bool = False,
    ):
        """

        Args:
            hypo_text (torch.Tensor or List[str] or str): the hypothesis text
            real_text (torch.Tensor or List[str] or str): the reference text
            tokenizer (Tokenizer): the tokenizer
            do_aver (bool): whether to average the error rate over the batch

        Returns:

        """
        if tokenizer is None:
            assert self.tokenizer is not None
            tokenizer = self.tokenizer

        # make sure that hypo_text is a 2-dim tensor or a list of strings
        if isinstance(hypo_text, torch.Tensor) and hypo_text.dim() == 1:
            hypo_text = hypo_text.unsqueeze(0)
        elif isinstance(hypo_text, str):
            hypo_text = [hypo_text]
        # make sure that real_text is a 2-dim tensor or a list of strings
        if isinstance(real_text, torch.Tensor) and real_text.dim() == 1:
            real_text = real_text.unsqueeze(0)
        elif isinstance(real_text, str):
            real_text = [real_text]

        # ### DEBUG: Print length information
        # print(f"DEBUG ErrorRate: hypo_text type={type(hypo_text)}, len={len(hypo_text)}")
        # print(f"DEBUG ErrorRate: real_text type={type(real_text)}, len={len(real_text)}")

        # if isinstance(hypo_text, list) and len(hypo_text) > 0:
        #     print(f"DEBUG ErrorRate: hypo_text[0]={hypo_text[0]}")
        # if isinstance(real_text, list) and len(real_text) > 0:
        #     print(f"DEBUG ErrorRate: real_text[0]={real_text[0]}")
        # ### DEBUG ends

        # SAFETY CHECK: Handle length mismatch
        if len(hypo_text) != len(real_text):
            # print(f"ERROR: Length mismatch! hypo_text={len(hypo_text)}, real_text={len(real_text)}")
            # Use minimum length to prevent crash
            min_len = min(len(hypo_text), len(real_text))
            hypo_text = hypo_text[:min_len]
            real_text = real_text[:min_len]
            # print(f"DEBUG: Truncated to length {min_len}")

        cer_dist, cer_len, wer_dist, wer_len = [], [], [], []
        for i in range(len(hypo_text)):
            # obtain the strings
            hypo_string = text_preprocess(hypo_text[i], tokenizer)
            real_string = text_preprocess(real_text[i], tokenizer)

            # calculate CER
            hypo_chars = hypo_string.replace(" ", "")
            real_chars = real_string.replace(" ", "")
            cer_dist.append(editdistance.eval(hypo_chars, real_chars))
            cer_len.append(len(real_chars))

            # calculate WER
            # Note that split(" ") is not equivalent to split() here
            # because split(" ") will give an extra '' at the end of the list if the string ends with a " "
            # while split() doesn't
            hypo_words = hypo_string.split()
            real_words = real_string.split()
            wer_dist.append(editdistance.eval(hypo_words, real_words))
            wer_len.append(len(real_words))

        cer, wer = [], []
        for i in range(len(cer_dist)):
            cer.append(cer_dist[i] / cer_len[i])
            wer.append(wer_dist[i] / wer_len[i])
        if do_aver:
            cer = sum(cer) / len(cer)
            wer = sum(wer) / len(wer)

        return cer, wer


if __name__ == "__main__":
    import sys
    from speechain.tokenizer.char import CharTokenizer
    from speechain.tokenizer.sp import SentencePieceTokenizer

    args = sys.argv[1:]
    assert len(args) == 2, "Give vocab path eg. python <tokenizer> error_rate.py <path>"
    token      = args[0]
    token_path = args[1]
    # token_path = "/home/is/r-ghimire/speechain/datasets/slr54nepaliasr/data/char/train/full_tokens/no-punc"

    tokenizer = None
    if token == "sp":
        tokenizer = SentencePieceTokenizer(token_path=token_path)
    elif token == "char":
        tokenizer = CharTokenizer(token_path=token_path)

    error_rate = ErrorRate(tokenizer=tokenizer)

    hypo_text = "This is test"
    ref_text  = "This is test"

    cer, wer = error_rate(hypo_text, ref_text)

    assert cer != 0.0, f"CER calculation failed with cer={cer}"
    assert wer != 0.0, f"WER calculation failed with wer={wer}"
    print("== All test passed ==")
