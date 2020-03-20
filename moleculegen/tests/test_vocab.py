"""
Test `Vocabulary` class and its main components.
"""

import unittest

from moleculegen.data import SMILESDataset
from moleculegen.utils import SpecialTokens
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
        special_tokens = set(
            token.value for token in SpecialTokens.__members__.values()
        )

        self.assertSequenceEqual(
            sorted(set(self.temp_file.smiles_strings.replace('\n', ''))),
            sorted(set(self.vocab.token_to_idx) - special_tokens),
        )
        self.assertSequenceEqual(
            sorted(
                set(self.vocab.token_to_idx)
                # Pad and unknown tokens does not appear in the original set.
                - {SpecialTokens.PAD.value, SpecialTokens.UNK.value}
            ),
            sorted(set(self.vocab.token_freqs)),
        )

    def test_all_tokens(self):
        self.assertEqual(
            len(self.vocab.all_tokens),
            len(self.temp_file.smiles_strings.split('\n')),
        )

    def test_corpus(self):
        # Original SMILES list without padded special tokens.
        smiles_list = self.temp_file.smiles_strings.split('\n')

        self.assertEqual(len(self.vocab.corpus), len(smiles_list))

        for idx, tokens in zip(self.vocab.corpus, smiles_list):
            # Add special tokens in order to correspond to the loaded corpus
            # for data sampling and model fitting.
            tokens = SpecialTokens.add_tokens_to(tokens)
            # Test id-to-token mapping.
            self.assertListEqual(self.vocab.get_tokens(idx), list(tokens))
            # Test token-to-id mapping.
            self.assertListEqual(idx, self.vocab[tokens])

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
