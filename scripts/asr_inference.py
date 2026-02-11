import os
import random
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.data_loading_util import read_data_by_path
from speechain.runner import Runner

from speechain.infer_func.beam_search_single import beam_search_single


tqdm.pandas()

SPEECHAIN_ROOT = "/home/is/r-ghimire/speechain"
EXP_DIR    = f"{SPEECHAIN_ROOT}/recipes/asr/slr54nepaliasr/exp/ne_char_conformer-tuned_lr2e-3"
CHECKPOINT = f"{EXP_DIR}/models/10_valid_accuracy_average.pth"
DEVICE = "cuda"


def load_asr_model():
    os.environ.setdefault("SPEECHAIN_ROOT", SPEECHAIN_ROOT)

    exp_cfg = load_yaml(f"{EXP_DIR}/exp_cfg.yaml")
    model_cfg = exp_cfg["train_cfg"]["model"]

    args = SimpleNamespace(
        train_result_path=EXP_DIR,
        non_blocking=True,
        distributed=False,
    )
    device = torch.device(DEVICE)
    model = Runner.build_model(model_cfg, args=args, device=device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    state = ckpt["latest_model"] if "latest_model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def transcribe(audio_file_name, model, decoding_tech = 'builtin', infer_conf={}):
    """
        decoding_tech: builtin (as is of speechain)
                 ccnn_guided
    """
    wav, sr = read_data_by_path(audio_file_name, return_tensor=True, return_sample_rate=True)
    # frontend in your config expects 16k
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav.squeeze(-1)).unsqueeze(-1)

    feat = wav.unsqueeze(0).to(DEVICE)  # (B,T,1)
    feat_len = torch.tensor([wav.shape[0]], device=DEVICE)

    if decoding_tech == 'builtin':
        infer_conf = {
            "beam_size": 16,
            "ctc_weight": 0.2,
            "decode_only": True,
        }
        with torch.inference_mode():
            out = model.inference(
                infer_conf=infer_conf,
                feat=feat,
                feat_len=feat_len,
                decode_only=True,
            )
        return out["text"]["content"][0]

    elif decoding_tech == 'ccnn_guided':
        enc_feat, enc_feat_mask, _, _ = model.encoder(feat=feat, feat_len=feat_len)
        out = beam_search_single(
            enc_feat=enc_feat,
            enc_feat_mask=enc_feat_mask,
            asr_decode_fn=model.decoder,
            ctc_decode_fn=model.ctc_layer if hasattr(model, "ctc_layer") else None,
            lm_decode_fn=model.lm if hasattr(model, "lm") else None,
            vocab_size=model.tokenizer.vocab_size,
            sos_eos=model.tokenizer.sos_eos_idx,
            padding_idx=model.tokenizer.ignore_idx,
            **infer_conf,
        )
        hypo_ids = out["hypo_text"]  # torch.LongTensor
        texts = [model.tokenizer.tensor2text(h) for h in hypo_ids]

        return {
            "sample_rate": sr,
            "text": texts[0],
            "confidence": float(out["hypo_text_confid"][0].item()),
            "token_ids": hypo_ids[0].detach().cpu(),
            "token_len": int(out["hypo_text_len"][0].item()),
        }

if __name__ == "__main__":
    # load data from test
    id2text = { }
    id2wav  = { }
    test_folder = "/home/is/r-ghimire/speechain/datasets/slr54nepaliasr/data/wav16000/test"
    #todo
    with open(os.path.join(test_folder, "idx2text"), encoding = "utf-8") as f:
        id2text = dict(line.rstrip("\n").split(" ", 1) for line in f)

    with open(os.path.join(test_folder, "idx2wav"), encoding = "utf-8") as f:
        id2wav = dict(line.rstrip("\n").split(" ", 1) for line in f)

    model = load_asr_model()

    # df = (
    #     pd.DataFrame({
    #         "text": pd.Series(id2text),
    #         "wav_path": pd.Series(id2wav),
    #     })
    #     .rename_axis("id")
    #     .reset_index()
    # )
    # print(f"Data Loaded: {len(df)}")
    # print(f"Model Loaded")

    # def trans(path):
    #     return transcribe(path, model, decoder='builtin')

    # #only do for 1000 samples after suffle
    # df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    # df["transcript"] = df["wav_path"].progress_map(trans)
    # print("Transcription Complete")
    # df.to_csv("test_transcripts.csv", index=False, sep='\t')

    ### Test
    fileid = random.choice(list(id2text))
    text   = id2text[fileid]
    path   = id2wav[fileid]

    print(f"FileID: {fileid}   Reference Text: {text}  \nPath: {path}")

    infer_conf = {
        "beam_size": 5,
        "temperature": 1.0,
        "length_penalty": 1.0,
        "min_f2t_ratio": 3.0,

        # optional fusions
        "ctc_weight": 0.3,
        "ctc_temperature": 1.0,

        "lm_weight": 0.2,
        "lm_temperature": 1.0,

        # optional eos filtering
        "eos_filtering": False,
        "eos_threshold": 1.5,
    }

    # transcript = transcribe(path, model, decoder='builtin')
    transcript = transcribe(path, model, decoding_tech='ccnn_guided', infer_conf=infer_conf)
    print(f"{"- "*10}\nRef. Text: {text}\nGen. Text: {transcript}\n{"- "*10}")
