import os
import tempfile
import unittest

from moleculegen.data import SMILESDataset
from moleculegen.utils import UNK
from moleculegen.vocab import Vocabulary


class VocabTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles_strings = (
            'CC(=O)NCCC1=CNc2c1cc(OC)cc2\n'
            'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
            'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
        )

        self.fh = tempfile.NamedTemporaryFile(mode='w+', encoding='ascii')
        self.fh.write(self.smiles_strings)
        self.fh.seek(os.SEEK_SET)

        # See `test_data.py` for data set test cases.
        self.dataset = SMILESDataset(self.fh.name)
        self.vocab = Vocabulary(self.dataset, need_corpus=True)

    def test_tokens_and_idx(self):
        self.assertSequenceEqual(
            sorted(set(self.smiles_strings)),
            sorted(set(self.vocab.token_to_idx) - set(UNK)),
        )
        self.assertSequenceEqual(
            sorted(set(self.vocab.token_to_idx) - set(UNK)),
            sorted(set(self.vocab.token_freqs)),
        )

    def test_all_tokens(self):
        self.assertEqual(
            len(self.vocab.all_tokens),
            len(self.smiles_strings.split('\n')),
        )

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
