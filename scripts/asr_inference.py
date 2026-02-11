import os
import sys
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

# Add path for CCNN
sys.path.append("/home/is/r-ghimire/decoding")
from CCNN.model import CCNN
from CCNN.vocab import Vocab as CCNNVocab

tqdm.pandas()

SPEECHAIN_ROOT = "/home/is/r-ghimire/speechain"
EXP_DIR    = f"{SPEECHAIN_ROOT}/recipes/asr/slr54nepaliasr/exp/ne_char_conformer-tuned_lr2e-3"
CHECKPOINT = f"{EXP_DIR}/models/10_valid_accuracy_average.pth"
DEVICE = "cuda:3"

CCNN_VOCAB_PATH = "/home/is/r-ghimire/decoding/CCNN/ne.vocab"
CCNN_CHECKPOINT = "/home/is/r-ghimire/decoding/checkpoint/ccnn_lstm_slr54_ep20.pt"


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


def load_ccnn_model(device):
    with open(CCNN_VOCAB_PATH, "r", encoding="utf-8") as f:
        tokens = [v.rstrip('\n') for v in f]
    vocab = CCNNVocab(tokens)

    ckpt = torch.load(CCNN_CHECKPOINT, map_location=device, weights_only=False)
    args = ckpt['config']

    model = CCNN(
        vocab_size=len(vocab.tokens),
        pad_id=vocab.pad_id,
        d_model=args.d_model,
        n_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocab


def get_asr2ccnn_map(asr_tokenizer, ccnn_vocab, device):
    mapping = torch.zeros(asr_tokenizer.vocab_size, dtype=torch.long)

    for i in range(asr_tokenizer.vocab_size):
        if i == asr_tokenizer.ignore_idx:
            mapping[i] = ccnn_vocab.pad_id
        elif i == asr_tokenizer.sos_eos_idx:
            mapping[i] = ccnn_vocab.eos_id
        else:
            # Assuming idx2token exists or similar method
            token = asr_tokenizer.idx2token.get(i, "")
            mapping[i] = ccnn_vocab.token_to_id.get(token, ccnn_vocab.pad_id)

    return mapping.to(device)


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
        # infer_conf = {
        #     "beam_size": 20,
        #     "ctc_weight": 0.2,
        #     "decode_only": True,
        # }
        with torch.inference_mode():
            out = model.inference(
                infer_conf=infer_conf,
                feat=feat,
                feat_len=feat_len,
                # decode_only=True,
            )
        return out

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

    # Load CCNN and create map
    device = torch.device(DEVICE)
    ccnn_model, ccnn_vocab = load_ccnn_model(device)
    asr2ccnn_map = get_asr2ccnn_map(model.tokenizer, ccnn_vocab, device)

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
    # fileid = random.choice(list(id2text))
    fileid = "948d1cd7a7"
    text   = id2text[fileid]
    path   = id2wav[fileid]

    print(f"FileID: {fileid}   Reference Text: {text}  \nPath: {path}")

    infer_conf = {
        "beam_size": 20,
        "temperature": 1.0,
        "length_penalty": 1.0,
        "min_f2t_ratio": 1.0,

        # optional fusions
        "ctc_weight": 0.2,
        "ctc_temperature": 1.0,

        "lm_weight": 0.2,
        "lm_temperature": 1.0,

        # optional eos filtering
        "eos_filtering": True,
        "eos_threshold": 1.5,

        # CCNN config
        "ccnn_model": ccnn_model,
        "asr2ccnn_map": asr2ccnn_map,
        "ccnn_bos_id": ccnn_vocab.bos_id,
        "ccnn_pad_id": ccnn_vocab.pad_id,
        "ccnn_lm_weight": 0.8,
        "ccnn_cons_weight": 1.0,
    }

    print(f"{"- "*10}\nRef. Text: {text}")
    infer_conf["ccnn_model"] = None
    transcript = transcribe(path, model, decoding_tech='ccnn_guided', infer_conf=infer_conf)
    gen = transcript['text'].replace("*", " ")
    print(f"Gen. Text (No CCNN):      {gen}")
    infer_conf["ccnn_model"] = ccnn_model
    transcript = transcribe(path, model, decoding_tech='ccnn_guided', infer_conf=infer_conf)
    gen = transcript['text'].replace("*", " ")
    print(f"Gen. Text (CCNN):  {gen}")
    print(f"{"- "*10}")