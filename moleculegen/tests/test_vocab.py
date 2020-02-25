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
        self.assertSequenceEqual(
            sorted(set(self.temp_file.smiles_strings)),
            sorted(set(self.vocab.token_to_idx)
                   - {SpecialTokens.UNK.value, SpecialTokens.PAD.value}),
        )
        self.assertSequenceEqual(
            sorted(set(self.vocab.token_to_idx)
                   - {SpecialTokens.UNK.value, SpecialTokens.PAD.value}),
            sorted(set(self.vocab.token_freqs)),
        )

    def test_all_tokens(self):
        self.assertEqual(
            len(self.vocab.all_tokens),
            len(self.temp_file.smiles_strings.split('\n')),
        )

    def test_corpus(self):
        smiles_list = self.temp_file.smiles_strings.split('\n')

        self.assertEqual(len(self.vocab.corpus), len(smiles_list))

        for idx, tokens in zip(self.vocab.corpus, smiles_list):
            tokens += SpecialTokens.EOS.value
            self.assertListEqual(self.vocab.get_tokens(idx), list(tokens))

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
