import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from speechain.utilbox.data_loading_util import read_data_by_path


def transcribe(audio_file_name, model, infer_conf={}):
    wav, sr = read_data_by_path(audio_file_name, return_tensor=True, return_sample_rate=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav.squeeze(-1)).unsqueeze(-1)

    feat = wav.unsqueeze(0).to(model.device)  # (B,T,1)
    feat_len = torch.tensor([wav.shape[0]], device=model.device)
    with torch.inference_mode():
        out = model.inference(
            infer_conf=infer_conf,
            feat=feat,
            feat_len=feat_len,
            decode_only = True,
        )
    return out


def bulk_transcribe(df, model, infer_cfg):
    # individual transcibe
    def trans(path):
        result =  transcribe(path, model, infer_cfg)
        token_ids = result['token_ids']['content'][0][0]
        return {
            "pred_text": result['text']['content'][0][0],
            "pred_confid": result['text_confid']['content'][0],
            "pred_tokens": "##".join([model.tokenizer.tensor2text([id]) for id in token_ids]),
        }
    
    def text_to_tokens(text):
        token_ids = model.tokenizer.text2tensor(text).squeeze(0).tolist()
        # remove the bos and eos tokens
        token_ids = token_ids[1:-1]
        return "##".join([model.tokenizer.tensor2text([id]) for id in token_ids])
    
    results = df["wav_path"].progress_map(trans)
    result_df = pd.DataFrame(results.tolist(), index=df.index)
    result_df["ref_tokens"] = df["text"].progress_map(text_to_tokens)
    
    return pd.concat([df, result_df], axis=1)


def decode(audio_file_name, ref_text,  model, infer_conf={}):
    wav, sr = read_data_by_path(audio_file_name, return_tensor=True, return_sample_rate=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav.squeeze(-1)).unsqueeze(-1)

    feat = wav.unsqueeze(0).to(model.device)  # (B,T,1)
    feat_len = torch.tensor([wav.shape[0]], device=model.device)
    
    ref_text_tensor = model.tokenizer.text2tensor(ref_text).unsqueeze(0).to(model.device)  # (B, T_ref)
    ref_text_len = torch.tensor([ref_text_tensor.shape[1]], device=model.device)

    with torch.inference_mode():
        out = model.inference(
            infer_conf=infer_conf,
            feat=feat,
            feat_len=feat_len,
            text=ref_text_tensor,
            text_len=ref_text_len,
            decode_only = False,
        )
    return out


def decode_bulk(df, model, infer_cfg):
    # inner func used for mapping
    def to_decode(path, text):
        return decode(path, text, model, infer_cfg)
    
    df["transcript"] = df.apply(lambda row: to_decode(row["wav_path"], row["text"]), axis=1  )
    return df
