"""
Test `Vocabulary` class and its main components.
"""

import unittest

from moleculegen.base import Token
from moleculegen.data import SMILESDataset
from moleculegen.vocab import Vocabulary
from moleculegen.tests.utils import TempSMILESFile


class VocabTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile()
        self.fh = self.temp_file.open()

        # See `test_data.py` for data set test cases.
        self.dataset = SMILESDataset(self.fh.name)
        self.vocab = Vocabulary(self.dataset, need_corpus=True)

    def test_tokens_and_idx(self):
        self.assertSequenceEqual(
            # Tokenize the entire dataset to get a set of unique tokens.
            sorted(
                set(
                    Token.tokenize(
                        self.temp_file.smiles_strings.replace('\n', '')
                    )
                )
            ),
            # The temporary file is not augmented by the special tokens.
            sorted(set(self.vocab.token_to_idx) - Token.SPECIAL),
        )
        self.assertSequenceEqual(
            sorted(
                set(self.vocab.token_to_idx)
                # Pad and unknown tokens does not appear in the original set.
                - {Token.PAD, Token.UNK}
            ),
            sorted(set(self.vocab.token_freqs)),
        )

    def test_corpus(self):
        # Original SMILES list without padded special tokens.
        smiles_list = self.temp_file.smiles_strings.split('\n')

        self.assertEqual(len(self.vocab.corpus), len(smiles_list))

        for idx, tokens in zip(self.vocab.corpus, smiles_list):
            # Add special tokens in order to correspond to the loaded corpus
            # for data sampling and model fitting.
            tokens = Token.augment(tokens)
            # Test id-to-token mapping.
            self.assertEqual(
                ''.join(self.vocab.get_tokens(idx)),
                tokens,
            )
            # Test token-to-id mapping.
            self.assertListEqual(idx, self.vocab[Token.tokenize(tokens)])

    def test_contains(self):
        self.assertNotIn(Token.UNK, self.vocab)

        all_tokens = Token.get_all_tokens()

        for token in self.vocab:
            if len(token) == 1 and token.islower():
                token = token.upper()
            self.assertIn(token, all_tokens)

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
