#%% Script for evaluating the model
# %load_ext autoreload
# %autoreload 2

#%% Imports
import os, sys
import editdistance
from tqdm import tqdm

from speechain.utilbox.yaml_util import load_yaml
from loader import load_asr_model, load_dataset
from inference_asr import transcribe, bulk_transcribe

# os.environ.setdefault("SPEECHAIN_ROOT", "/home/is/r-ghimire/workspace/speechain")
SPEECHAIN_ROOT = os.environ.get("SPEECHAIN_ROOT", None)
if SPEECHAIN_ROOT is None:
    print("Error: SPEECHAIN_ROOT environment variable is not set.")
    sys.exit(1)


def calculate_er(ref_str, hyp_str, sep=''):
    ref = ref_str.split(sep) if sep else list(ref_str)
    hyp = hyp_str.split(sep) if sep else list(hyp_str)
    error_rate = editdistance.eval(ref, hyp) / len(ref)
    error_rate = round(error_rate, 4)*100
    return error_rate


#%% Main Method
if __name__ == "__main__":
    #%% Defaults
    dataset_name = "slr54nepaliasr"
    model_name   = "ne_ccnn_bpe1k5h_conformer-large-v2_lr2e-3"
    #%% Command Line Arguments
    if len(sys.argv) != 3:
        print("Usage: python asr_inference.py <dataset> <model_name>")
        sys.exit(1)
    
    dataset_name  = sys.argv[1]
    model_name    = sys.argv[2]

    #%% set Paths, Device and models
    exp_dir         = f"{SPEECHAIN_ROOT}/recipes/asr/{dataset_name}/exp/{model_name}"
    test_dir        = f"{SPEECHAIN_ROOT}/datasets/{dataset_name}/data/wav16000/test"
    checkpoint_dir  = f"{exp_dir}/models/10_valid_accuracy_average.pth"
    device          = "cuda:0"
    
    exp_cfg = load_yaml(f"{exp_dir}/exp_cfg.yaml")
    infer_conf = {
        "beam_size": 20,
        "ctc_weight": 0.2,
        "decode_only": True,
    }
    
    df = load_dataset(test_dir)
    # downsample for development
    # df = df.sample(n=2000, random_state=42).reset_index(drop=True)    
    model = load_asr_model(exp_dir, checkpoint_dir, device, exp_cfg)

    # %% Single test
    # Transcribe a few samples and print the results
    # infer_conf["decode_only"] = True
    # for row in df.sample(n=3, random_state=42).itertuples(): 
    #     fileid = row.id
    #     hyp_text = row.text
    #     path = row.wav_path
        
    #     transcript = transcribe(path, model, infer_conf)
    #     pred_text = transcript['text']['content'][0]
        
    #     print(f"{"- "*10}\nRef. Text: {hyp_text}")
    #     print(f"Gen. Text (Default):      {pred_text}")
    #     print(f"Raw:      {transcript}")
    #     print(f"{"- "*10}")

    #%% Run inference on dataframe
    # df = df.sample(n=20, random_state=42).reset_index(drop=True)    
    df_inf = bulk_transcribe(df, model, infer_conf)
    
    df_inf['cer'] = df_inf.apply(lambda row: calculate_er(row['text'], row['pred_text'], sep=''), axis=1)
    df_inf['wer'] = df_inf.apply(lambda row: calculate_er(row['text'], row['pred_text'], sep=' '), axis=1)
    df_inf['ter'] = df_inf.apply(lambda row: calculate_er(row['ref_tokens'], row['pred_tokens'], sep='##'), axis=1)
    
    df_inf.to_csv(f"{model_name}.csv", index=False, sep='\t')    
    # df_inf.head(10)
