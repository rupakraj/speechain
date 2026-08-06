"""
Modified BPE Tokenizer for Speechain
Similar to CharTokenizer but uses BPE vocabulary instead of characters
"""

import torch
from typing import List

from speechain.tokenizer.abs import Tokenizer


class ModifiedBPE(Tokenizer):
    """
    Tokenizer implementation that converts input text using BPE vocabulary.
    Similar to CharTokenizer but operates on BPE subword units instead of individual characters.

    This tokenizer handles:
    - BPE subword tokens from the vocabulary
    - Word boundary tokens (▁ prefix)
    - Nepali script tokens
    - Special tokens (<blank>, <unk>, <sos/eos>)
    """

    def tokenizer_init_fn(self, **tokenizer_conf):
        """
        Initialize any custom components for the ModifiedBPE tokenizer.

        Args:
            **tokenizer_conf: Additional configuration parameters
        """
        # You can add any custom initialization here
        self.lowercase = tokenizer_conf.get('lowercase', False)
        self.handle_word_boundaries = tokenizer_conf.get('handle_word_boundaries', True)

    def text2tensor(
        self,
        text: str,
        no_sos: bool = False,
        no_eos: bool = False,
        return_tensor: bool = True,
    ) -> torch.LongTensor or List:
        """
        Encode text string into token indices using BPE vocabulary.

        This implementation is similar to CharTokenizer but processes text at the BPE level.
        It attempts to match the longest possible BPE tokens from the vocabulary.

        Args:
            text: Input text string to be encoded
            no_sos: Whether to remove the <sos/eos> at the beginning
            no_eos: Whether to remove the <sos/eos> at the end
            return_tensor: Whether to return as tensor (vs list)

        Returns:
            Token indices as torch.LongTensor or List
        """
        # Preprocess text
        if self.lowercase:
            text = text.lower()

        # Initialize token list
        tokens = []

        # Add SOS token if not disabled
        if not no_sos:
            tokens.append(self.sos_eos_idx)

        # BPE tokenization - greedy longest-match algorithm
        remaining_text = text.strip()

        while remaining_text:
            # Try to find the longest matching token from the beginning
            matched = False

            # Try from longest to shortest (greedy approach)
            for length in range(min(len(remaining_text), 20), 0, -1):
                candidate = remaining_text[:length]

                # Check if candidate exists in vocabulary
                if candidate in self.token2idx:
                    tokens.append(self.token2idx[candidate])
                    remaining_text = remaining_text[length:]
                    matched = True
                    break

            # If no match found, try with word boundary prefix
            if not matched and self.handle_word_boundaries:
                for length in range(min(len(remaining_text), 20), 0, -1):
                    candidate_with_boundary = '▁' + remaining_text[:length]

                    if candidate_with_boundary in self.token2idx:
                        tokens.append(self.token2idx[candidate_with_boundary])
                        remaining_text = remaining_text[length:]
                        matched = True
                        break

            # If still no match, use UNK token and move one character forward
            if not matched:
                tokens.append(self.unk_idx)
                remaining_text = remaining_text[1:]

        # Add EOS token if not disabled
        if not no_eos:
            tokens.append(self.sos_eos_idx)

        # Return as tensor or list
        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens


    def text2tensor_new(self, text: str, no_sos: bool = False, no_eos: bool = False, return_tensor: bool = True):
        """Greedy longest-match-first BPE tokenization."""

        if self.lowercase:
            text = text.lower()

        tokens = []

        if not no_sos:
            tokens.append(self.sos_eos_idx)

        remaining_text = text.strip()

        while remaining_text:
            matched = False
            max_length = len(remaining_text)  # Remove the 20-char limit

            # TRY WITHOUT WORD BOUNDARY FIRST (longest match)
            for length in range(max_length, 0, -1):
                candidate = remaining_text[:length]
                if candidate in self.token2idx:
                    tokens.append(self.token2idx[candidate])
                    remaining_text = remaining_text[length:]
                    matched = True
                    break

            # FALLBACK: Try with word boundary prefix (if enabled)
            if not matched and self.handle_word_boundaries:
                for length in range(max_length, 0, -1):
                    candidate_with_boundary = '▁' + remaining_text[:length]
                    if candidate_with_boundary in self.token2idx:
                        tokens.append(self.token2idx[candidate_with_boundary])
                        remaining_text = remaining_text[length:]
                        matched = True
                        break

            # FALLBACK: Single character (last resort)
            if not matched:
                single_char = remaining_text[0]
                if single_char in self.token2idx:
                    tokens.append(self.token2idx[single_char])
                else:
                    tokens.append(self.unk_idx)  # Use UNK only if char not in vocab
                remaining_text = remaining_text[1:]

        if not no_eos:
            tokens.append(self.sos_eos_idx)

        if return_tensor:
            return torch.LongTensor(tokens)
        else:
            return tokens