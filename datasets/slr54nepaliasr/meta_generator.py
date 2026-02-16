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
        return all(ch in vocab_chars for ch in text)

    mask = df[text_col].apply(is_valid_text)
    return df[mask].reset_index(drop=True)


class SLR54MetaGenerator(SpeechTextMetaGenerator):
    # no additional argument needed as the --src_path is enough

    def generate_meta_dict(self, src_path: str, txt_format: str, **kwargs) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]: # type: ignore
        """
            - Read tsv file
            - split the train/dev/test as 80:10:10
            - return speechain style dict
        """

        src_path = os.path.abspath(src_path)
        tsv_path = os.path.join(src_path, "data", "utt_spk_text.tsv")
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Tsv file not found in {tsv_path}")

        # read tsv file
        df_all = pd.read_csv(tsv_path, sep="\t", header=None, names=["fileid","speaker", "text"])
        df = filter_df_by_vocab(df_all, "ne.vocab")

        def populate_file_path(id):
            return os.path.join(src_path,"data", "wav", id[:2], f"{id}.wav")
        def populate_npz_path(id):
            return os.path.join(src_path,"data", "npz", id[:2], f"{id}.npz")

        df['path'] = df["fileid"].map(populate_file_path)
        # df['npz_path'] = df["fileid"].map(populate_npz_path)

        # Define the labels and their corresponding ratios
        labels = ['train', 'test', 'valid']
        ratios = [0.8, 0.1, 0.1]
        df['split'] = np.random.choice(labels, size=len(df), p=ratios)

        meta_dict = dict()
        for subset in labels:
            meta_dict[subset] = dict(
                                    idx2wav  = dict(),
                                    idx2spk  = dict(),
                                    idx2gen  = dict(),
                                    idx2text = dict(),
                                    # idx2feat = dict(),
                                )
            meta_dict[subset][f'idx2no-punc_text'] = dict()

        for _, row in df.iterrows():
            split = row['split']
            fileid = row['fileid']
            meta_dict[split]['idx2wav'][fileid] = row['path']
            # meta_dict[split]['idx2feat'][fileid] = row['npz_path']
            meta_dict[split]['idx2spk'][fileid] = row['speaker']
            meta_dict[split]['idx2text'][fileid] = row['text']
            meta_dict[split]['idx2no-punc_text'][fileid] = row['text']

        return meta_dict

        # ret_dict = {
        #     "idx2wav": [],
        #     "idx2wav_len": [],
        #     "idx2feat": [],
        #     "idx2feat_len": [],
        #     "idx2text": [],
        #     "idx2spk": [],
        #     "spk_list": []
        # }


if __name__ == '__main__':
    SLR54MetaGenerator().main()

