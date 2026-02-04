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


class SLR54MetaGenerator(SpeechTextMetaGenerator):
    # no additional argument needed as the --src_path is enough

    def generate_meta_dict(self, src_path: str, txt_format: str, **kwargs) \
            -> Dict[str, Dict[str, Dict[str, str] or List[str]]]:
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
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["fileid","speaker", "text"])

        def populate_file_path(id):
            return os.path.join(src_path,"data", "wav", id[:2], f"{id}.wav")
        def populate_npz_path(id):
            return os.path.join(src_path,"data", "npz", id[:2], f"{id}.npz")

        df['path'] = df["fileid"].map(populate_file_path)
        df['npz_path'] = df["fileid"].map(populate_file_path)

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
                                    idx2feat = dict(),
                                )
            meta_dict[subset][f'idx2no-punc_text'] = dict()

        for _, row in df.iterrows():
            split = row['split']
            fileid = row['fileid']
            meta_dict[split]['idx2wav'][fileid] = row['path']
            meta_dict[split]['idx2feat'][fileid] = row['npz_path']
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
