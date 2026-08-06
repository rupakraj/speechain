import copy
from typing import Dict, List

import torch
import torch.nn as nn

from speechain.criterion.accuracy import Accuracy
from speechain.criterion.cross_entropy import CrossEntropy
from speechain.model.abs import Model
from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.sp import SentencePieceTokenizer
from speechain.utilbox.tensor_util import to_cpu
from speechain.utilbox.train_util import make_mask_from_len


class CCNNLanguageModel(nn.Module):
    """CCNN-based Language Model that combines LSTM with constituency constraints."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
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

    def forward(self, text: torch.Tensor, text_len: torch.Tensor, cand_ids: torch.Tensor = None):
        """
        Args:
            text: (batch, text_maxlen) Input tokens
            text_len: (batch,) Sequence lengths
            cand_ids: (batch, text_maxlen, k) Candidate token IDs (optional)

        Returns:
            logits: (batch, text_maxlen, vocab_size) Language model logits
            cons_logits: (batch, text_maxlen, k) Constituency logits
            enc_attmat: None (for compatibility with LM interface)
        """
        B, T = text.shape

        # Embedding
        x_emb = self.dropout(self.emb(text))

        # LSTM processing with packing
        lengths_cpu = text_len.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )
        h = self.dropout(h)

        # Language model logits
        logits = self.lm_head(h)

        # Constituency logits
        if cand_ids is None:
            cons_logits = torch.matmul(h, self.emb.weight.t())  # (B,T,V)
        else:
            cand_emb = self.emb(cand_ids)      # (B,T,K,D)
            cons_logits = (h.unsqueeze(2) * cand_emb).sum(dim=-1)  # (B,T,K)

        return logits, cons_logits, None


class LM_CCNN(Model):
    """Auto-Regressive CCNN-based Language Model compatible with LM interface."""

    def module_init(
        self,
        token_type: str,
        token_path: str,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        return_att_head_num: int = 2,
        return_att_layer_num: int = 2,
    ):
        # --- 1. Module-independent Initialization --- #
        # Initialize tokenizer (same as LM)
        if token_type.lower() == "char":
            self.tokenizer = CharTokenizer(token_path, copy_path=self.result_path)
        elif token_type.lower() == "sentencepiece":
            self.tokenizer = SentencePieceTokenizer(
                token_path, copy_path=self.result_path
            )
        else:
            raise ValueError(
                f"Unknown token_type {token_type}. "
                f"Currently, {self.__class__.__name__} supports one of ['char', 'sentencepiece']."
            )

        self.return_att_head_num = return_att_head_num
        self.return_att_layer_num = return_att_layer_num

        # --- 2. Module Initialization --- #
        self.lm_ccnn = CCNNLanguageModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout
        )

    def criterion_init(self, **criterion_conf):
        # Initialize cross-entropy loss
        self.ce_loss = CrossEntropy(**criterion_conf)

        # Initialize teacher-forcing accuracy for validation
        self.accuracy = Accuracy()

    @staticmethod
    def bad_cases_selection_init_fn() -> List[List[str or int]] or None:
        return [
            ["text_ppl", "max", 30],
            ["text_ppl", "min", 30],
            ["text_confid", "max", 30],
            ["text_confid", "min", 30],
        ]

    def module_forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        epoch: int = None,
        domain: str = None,
        return_att: bool = False,
        cand_ids: torch.Tensor = None,
        **kwargs,
    ) -> Dict:
        """Forward pass compatible with LM interface."""
        assert text_len.size(0) == text.size(0)

        # Remove the <sos/eos> at the end of each sentence (same as LM)
        for i in range(text_len.size(0)):
            text[i, text_len[i] - 1] = self.tokenizer.ignore_idx
        text, text_len = text[:, :-1], text_len - 1

        # CCNN forward pass
        logits, cons_logits, enc_attmat = self.lm_ccnn(text, text_len, cand_ids)

        # Initialize outputs (compatible with LM)
        outputs = dict(logits=logits, cons_logits=cons_logits)

        # Return attention if specified (for compatibility)
        if return_att and enc_attmat is not None:
            outputs.update(att=enc_attmat)

        return outputs

    def criterion_forward(
        self, logits: torch.Tensor, text: torch.Tensor, text_len: torch.Tensor, cons_logits: torch.Tensor = None
    ) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]) or Dict[str, torch.Tensor]:
        """Criterion calculation compatible with LM interface."""
        accuracy = self.accuracy(logits=logits, text=text, text_len=text_len)

        # Mask generation for the input text
        text_mask = make_mask_from_len(text_len - 1, return_3d=False)
        if text.is_cuda:
            text_mask = text_mask.cuda(text.device)

        # Perplexity calculation (same as LM)
        log_prob = torch.log_softmax(logits, dim=-1)
        text_prob = log_prob.gather(-1, text[:, 1:].view(text.size(0), -1, 1)).squeeze(dim=-1)
        text_prob = text_prob.masked_fill(~text_mask, 0.0)
        text_ppl = torch.exp(
            torch.sum(text_prob, dim=-1) * (-1 / (text_len - 1))
        ).mean()

        metrics = dict(accuracy=accuracy.detach(), text_ppl=text_ppl.clone().detach())

        loss = self.ce_loss(logits=logits, text=text, text_len=text_len)
        losses = dict(loss=loss)
        metrics.update(loss=loss.clone().detach())

        # Add constituency loss if available
        if cons_logits is not None:
            # You can add constituency-specific loss calculation here
            # For now, we'll just track it as a metric
            metrics.update(cons_logits_mean=cons_logits.mean().detach())

        if self.training:
            return losses, metrics
        else:
            return metrics

    def inference(
        self,
        infer_conf: Dict,
        text: torch.Tensor = None,
        text_len: torch.Tensor = None,
        domain: str = None,
        return_att: bool = False,
    ) -> Dict[str, Dict[str, str or List]]:
        """Inference compatible with LM interface."""
        assert text is not None and text_len is not None

        # Copy input data
        model_input = copy.deepcopy(dict(text=text, text_len=text_len))

        # LM Decoding by Teacher Forcing
        infer_results = self.module_forward(return_att=return_att, **model_input)
        outputs = dict()

        # Add attention matrix if requested
        if return_att and "att" in infer_results:
            outputs.update(att=infer_results["att"])

        # Perplexity Calculation (same as LM)
        log_prob = torch.log_softmax(infer_results["logits"], dim=-1)
        hypo_text_prob = log_prob.gather(
            -1, text[:, 1:].view(text.size(0), -1, 1)
        ).squeeze(dim=-1)
        hypo_text_ppl = torch.exp(
            torch.sum(hypo_text_prob, dim=-1) * (-1 / (text_len - 1))
        )

        # Confidence Calculation (same as LM)
        log_prob = log_prob[:, :-1]
        hypo_text_prob, hypo_text = torch.max(log_prob, dim=-1)
        length_penalty = (
            infer_conf["length_penalty"]
            if "length_penalty" in infer_conf.keys()
            else 1.0
        )
        hypo_text_confid = torch.sum(hypo_text_prob, dim=-1) / (
            (text_len - 2) ** length_penalty
        )

        # Convert to CPU
        hypo_text_confid, hypo_text_ppl = to_cpu(hypo_text_confid), to_cpu(hypo_text_ppl)

        # Recover text tensors back to strings
        hypo_text = [
            self.tokenizer.tensor2text(
                hypo[
                    (hypo != self.tokenizer.ignore_idx)
                    & (hypo != self.tokenizer.sos_eos_idx)
                ]
            )
            for hypo in hypo_text
        ]

        # Update outputs
        outputs.update(
            text=dict(format="txt", content=hypo_text),
            text_confid=dict(format="txt", content=hypo_text_confid),
            text_ppl=dict(format="txt", content=hypo_text_ppl),
        )

        # Add constituency scores if available
        # Add constituency scores if available
        if "cons_logits" in infer_results:
            cons_scores = torch.max(infer_results["cons_logits"], dim=-1)[0]
            # Take mean across sequence length for each batch item BEFORE converting to CPU
            cons_scores_mean = to_cpu(cons_scores.mean(dim=1))  # (batch,)
            outputs.update(cons_scores=dict(format="txt", content=cons_scores_mean))

        # Instance reports
        instance_report_dict = {}
        for i in range(len(text)):
            if "Text Confidence" not in instance_report_dict.keys():
                instance_report_dict["Text Confidence"] = []
            instance_report_dict["Text Confidence"].append(f"{hypo_text_confid[i]:.6f}")

            if "Text Perplexity" not in instance_report_dict.keys():
                instance_report_dict["Text Perplexity"] = []
            instance_report_dict["Text Perplexity"].append(f"{hypo_text_ppl[i]:.4f}")

            if "cons_scores" in outputs:
                if "Constituency Scores" not in instance_report_dict.keys():
                    instance_report_dict["Constituency Scores"] = []
                # instance_report_dict["Constituency Scores"].append(f"{cons_scores_mean[i].item():.4f}")
                instance_report_dict["Constituency Scores"].append(f"{cons_scores_mean[i]:.4f}")


        self.register_instance_reports(md_list_dict=instance_report_dict)

        return outputs

    def visualize(
        self,
        epoch: int,
        sample_index: str,
        snapshot_interval: int = 1,
        epoch_records: Dict = None,
        domain: str = None,
        text: torch.Tensor = None,
        text_len: torch.Tensor = None,
    ):
        """Visualization compatible with LM interface."""
        # Default visualization inference
        if len(self.visual_infer_conf) == 0:
            self.visual_infer_conf = dict()

        # Get inference results
        infer_results = self.inference(
            infer_conf=self.visual_infer_conf,
            return_att=True,
            text=text,
            text_len=text_len,
        )

        # Snapshot objective metrics
        vis_logs = []
        materials = dict()

        # Track standard LM metrics
        for metric in ["text_confid", "text_ppl"]:
            if metric not in epoch_records[sample_index].keys():
                epoch_records[sample_index][metric] = []
            epoch_records[sample_index][metric].append(
                infer_results[metric]["content"][0]
            )
            materials[metric] = epoch_records[sample_index][metric]

        # Track constituency scores if available
        if "cons_scores" in infer_results:
            if "cons_scores" not in epoch_records[sample_index].keys():
                epoch_records[sample_index]["cons_scores"] = []
            epoch_records[sample_index]["cons_scores"].append(
                infer_results["cons_scores"]["content"][0]
            )
            materials["cons_scores"] = epoch_records[sample_index]["cons_scores"]

        # Save visualization log
        vis_logs.append(
            dict(
                plot_type="curve",
                materials=copy.deepcopy(materials),
                epoch=epoch,
                xlabel="epoch",
                x_stride=snapshot_interval,
                sep_save=False,
                subfolder_names=sample_index,
            )
        )

        # Record input text at first snapshot
        if epoch // snapshot_interval == 1:
            vis_logs.append(
                dict(
                    materials=dict(
                        real_text=[
                            copy.deepcopy(self.tokenizer.tensor2text(text[0][1:-1]))
                        ]
                    ),
                    plot_type="text",
                    subfolder_names=sample_index,
                )
            )

        # Handle attention matrix visualization if available
        if "att" in infer_results:
            infer_results["att"] = self.attention_reshape(infer_results["att"])
            self.matrix_snapshot(
                vis_logs=vis_logs,
                hypo_attention=copy.deepcopy(infer_results["att"]),
                subfolder_names=sample_index,
                epoch=epoch,
            )

        return vis_logs
