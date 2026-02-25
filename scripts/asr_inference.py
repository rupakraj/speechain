import os
import sys
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

from speechain.utilbox.yaml_util import load_yaml
from speechain.utilbox.data_loading_util import read_data_by_path
from speechain.runner import Runner

tqdm.pandas()

SPEECHAIN_ROOT = "/home/is/r-ghimire/speechain"
EXP_DIR    = f"{SPEECHAIN_ROOT}/recipes/asr/slr54nepaliasr/exp/ne_char_conformer-large-v2_lr2e-3"
CHECKPOINT = f"{EXP_DIR}/models/10_valid_accuracy_average.pth"
DEVICE = "cuda:1"


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
                       ccnn_guided #TODO
    """
    wav, sr = read_data_by_path(audio_file_name, return_tensor=True, return_sample_rate=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav.squeeze(-1)).unsqueeze(-1)

    feat = wav.unsqueeze(0).to(DEVICE)  # (B,T,1)
    feat_len = torch.tensor([wav.shape[0]], device=DEVICE)

    if decoding_tech == 'builtin':
        infer_conf = {
            "beam_size": 20,
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
        return out

    elif decoding_tech == 'ccnn_guided':
        raise NotImplementedError


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
    # fileid = random.choice(list(id2text))
    fileid = "6d41af7ee6" # "948d1cd7a7"
    text   = id2text[fileid]
    path   = id2wav[fileid]
    # path   = "/home/is/r-ghimire/decoding/ds/arctic_1203000909.wav"

    print(f"FileID: {fileid}   Reference Text: {text}  \nPath: {path}")

    # text = "बढि सामाग्री जम्मा गर र पुन सोच"
    #  6d41af7ee6: गर्नुको महान् उद्देश्य
    print(f"{"- "*10}\nRef. Text: {text}")
    transcript = transcribe(path, model)
    # gen = transcript['text'].replace("*", " ")
    print(f"Gen. Text (Default):      {transcript['text']['content'][0]}")
    print(f"{"- "*10}")
