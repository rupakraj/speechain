"""
    Author: Rupak Raj (rughimire)
    Affiliation: NAIST
    Date: 2026.02
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Add parent directory and SPEECHAIN_ROOT to path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
speechain_root = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if speechain_root not in sys.path:
    sys.path.insert(0, speechain_root)

from meta_generator import SpeechTextMetaGenerator


def ne_text_process(text:str, txt_format: str = "plain")->str:
    text = text.strip()
    return text

# remove <unk> tokens if any from the dataset
def filter_df_by_vocab(df, vocab_path, text_col ='text'):
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab_chars = set()
        vocab_chars.add(' ')

        for line in vocab_file:
            token = line.strip('\n')
            if token:
                vocab_chars.update(token)

    def is_valid_text(text):
        if pd.isna(text):
            return False
        if not all(ch in vocab_chars for ch in text):
            return False
        # check the total number of words if it is 1 then return false
        # check the total number of character if it is less than 5 then return false
        if len(text.split()) == 1 or len(text) < 5:
            return False
        return True


    mask = df[text_col].apply(is_valid_text)
    return df[mask].reset_index(drop=True)


class SLR54MetaGenerator(SpeechTextMetaGenerator):
    # no additional argument needed as the --src_path is enough

    def generate_meta_dict(self, src_path: str, txt_format: str, **kwargs) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:  # type: ignore

        src_path = os.path.abspath(src_path)
        file_mapping = {
            'train': 'utt_train.tsv',
            'test':  'utt_test.tsv',
            'valid': 'utt_valid.tsv'
        }
        meta_dict = dict()

        for subset, filename in file_mapping.items():
            tsv_path = os.path.join(src_path, "data", filename)

            if not os.path.exists(tsv_path):
                print(f"Warning: {filename} not found in {src_path}/data. Skipping...")
                continue

            df_subset = pd.read_csv(tsv_path, sep="\t", header=None, names=["fileid", "speakerid", "text"])
            df_subset = filter_df_by_vocab(df_subset, "ne_speechain.vocab")

            meta_dict[subset] = {
                'idx2wav': dict(),
                'idx2spk': dict(),
                'idx2text': dict(),
                'idx2no-punc_text': dict()
            }

            for _, row in df_subset.iterrows():
                fileid = row['fileid']
                # Path construction logic
                wav_path = os.path.join(src_path, "data", "wav", fileid[:2], f"{fileid}.wav")
                meta_dict[subset]['idx2wav'][fileid] = wav_path
                meta_dict[subset]['idx2spk'][fileid] = row['speakerid']
                meta_dict[subset]['idx2text'][fileid] = row['text']
                meta_dict[subset]['idx2no-punc_text'][fileid] = row['text']

        return meta_dict


if __name__ == '__main__':
    SLR54MetaGenerator().main()

