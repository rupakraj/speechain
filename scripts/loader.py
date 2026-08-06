import os
import torch
import pandas as pd
from types import SimpleNamespace

from speechain.runner import Runner


def load_asr_model(exp_dir, checkpoint, device, exp_cfg):
    model_cfg = exp_cfg["train_cfg"]["model"]
    args = SimpleNamespace(
        train_result_path=exp_dir,
        non_blocking=True,
        distributed=False,
    )

    device = torch.device(device)
    model = Runner.build_model(model_cfg, args=args, device=device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["latest_model"] if "latest_model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_dataset(ds_folder):
    id2text = { }
    id2wav  = { }
    
    with open(os.path.join(ds_folder, "idx2text"), encoding = "utf-8") as f:
        id2text = dict(line.rstrip("\n").split(" ", 1) for line in f)

    with open(os.path.join(ds_folder, "idx2wav"), encoding = "utf-8") as f:
        id2wav = dict(line.rstrip("\n").split(" ", 1) for line in f)

    df = (
        pd.DataFrame({
            "text": pd.Series(id2text),
            "wav_path": pd.Series(id2wav),
        })
        .rename_axis("id")
        .reset_index()
    )
    return df
