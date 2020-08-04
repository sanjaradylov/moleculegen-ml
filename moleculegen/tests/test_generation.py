"""
Test SMILES samplers.
"""

import unittest

import mxnet as mx

from moleculegen import Token
from moleculegen.data import (
    SMILESDataset,
    SMILESVocabulary,
)
from moleculegen.estimation import SMILESEncoderDecoder
from moleculegen.generation import GreedySearch
from moleculegen.tests.utils import TempSMILESFile


class GreedySearchTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(
            tempfile_kwargs={'prefix': 'greedy_search'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)

        self.model = SMILESEncoderDecoder(len(self.vocabulary))

        self.predictor = GreedySearch()

    def test_call(self):
        self.model.initialize()
        states = self.model.begin_state(batch_size=1)

        valid_tokens = Token.get_all_tokens() - frozenset(Token.UNK)

        for _ in range(100):

            smiles = self.predictor(self.model, states, self.vocabulary)

            for token in Token.tokenize(smiles):
                if len(token) == 1 and token.islower():
                    token = token.upper()

                self.assertIn(token, valid_tokens)

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    mx.npx.set_np()
    unittest.main()
