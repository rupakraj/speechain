import os
import sys
import torch

from speechain.tokenizer.char import CharTokenizer
from speechain.tokenizer.modified_bpe import ModifiedBPE
from speechain.tokenizer.sp import SentencePieceTokenizer

def test_ccnn_tokenizer(lang, text):
    # Initialize your custom tokenizer
    token_path = f'datasets/indicvoices/data/{lang}/modified_bpe/train/bpe5h_ccnn/no-punc/vocab'
    copy_path = f'datasets/indicvoices/data/{lang}/modified_bpe/train/bpe5h_ccnn/no-punc'
    tokenizer = ModifiedBPE(token_path=token_path, copy_path=copy_path)

    # Encode text
    tokens = tokenizer.text2tensor_new(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    # Decode back to text
    decoded_text = tokenizer.tensor2text(tokens)
    print(f"Decoded: {decoded_text}")
    print(f"Matched: {text == decoded_text}")
    print(f"Total number of tokens: {len(tokens)}")
    # for token in tokens:
    #     print(f"Token: {token}, String: '{tokenizer.tensor2text(token)}'")


def test_sp_tokenizer(lang, text):
    # Initialize your custom tokenizer
    token_path = f'datasets/indicvoices/data/{lang}/sentencepiece/train/bpe5h/no-punc/vocab'
    copy_path = f'datasets/indicvoices/data/{lang}/sentencepiece/train/bpe5h/no-punc'
    tokenizer = SentencePieceTokenizer(token_path=token_path, copy_path=copy_path)

    # Encode text
    tokens = tokenizer.text2tensor(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    # Decode back to text
    decoded_text = tokenizer.tensor2text(tokens)
    print(f"Decoded: {decoded_text}")
    print(f"Matched: {text == decoded_text}")
    print(f"Total number of tokens: {len(tokens)}")

    # for token_id in tokens.tolist():
    #     print(f"Token: {token_id}, String: '{tokenizer.tensor2text([int(token_id)])}'")


def test_char_tokenizer():
    # Path to the vocabulary file provided
    vocab_file_path = "/home/is/r-ghimire/speechain/datasets/slr54nepaliasr/data/char/train/full_tokens/no-punc/vocab"

    # The Tokenizer class expects the directory containing the 'vocab' file as token_path
    token_path = os.path.dirname(vocab_file_path)

    print(f"Initializing CharTokenizer from: {token_path}")

    if not os.path.exists(vocab_file_path):
        print(f"Error: Vocab file not found at {vocab_file_path}")
        return

    try:
        # Initialize the tokenizer
        tokenizer = CharTokenizer(token_path=token_path)
        print("Tokenizer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        return

    # Display basic info
    print(f"\nVocabulary Size: {tokenizer.vocab_size}")
    print(f"Special Token Indices:")
    print(f"  SOS/EOS: {tokenizer.sos_eos_idx}")
    print(f"  UNK:     {tokenizer.unk_idx}")
    print(f"  BLANK:   {tokenizer.ignore_idx}")
    if tokenizer.space_idx is not None:
        print(f"  SPACE:   {tokenizer.space_idx}")

    # Test strings (Nepali and English/UNK)
    test_cases = [
        "नेपालमा",        # Nepali
        "Hello World"   # English
    ]

    for text in test_cases:
        print(f"\n--- Testing text: '{text}' ---")

        # 1. Text to Tensor
        tensor = tokenizer.text2tensor(text)
        print(f"Encoded Tensor: {tensor}")
        print(f"Tensor Shape: {tensor.shape}")

        # 2. Tensor to Text
        decoded_text = tokenizer.tensor2text(tensor)
        print(f"Decoded Text: '{decoded_text}'")

        # 3. Check reconstruction (ignoring UNK effects for exact match check)
        if text == decoded_text:
            print("Status: Perfect Reconstruction")
        else:
            print("Status: Reconstruction differs (likely due to UNK tokens or special tokens)")


if __name__ == "__main__":
    text = "पुनश्च वार्तामाद्यमाः तेषां कृते ये येषां कृते प्रामुख्यं यच्छन्ति इत्युक्तौ"
    lang = "Sanskrit"
    # test_char_tokenizer()
    print("\nTesting SentencePiece Tokenizer:")
    test_sp_tokenizer(lang, text)

    print("\nTesting CCNN Tokenizer:")
    test_ccnn_tokenizer(lang, text)