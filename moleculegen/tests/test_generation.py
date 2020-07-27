"""
Test SMILES samplers.
"""

import unittest

import mxnet as mx
from mxnet import gluon

from moleculegen import (
    SMILESRNNModel,
    Token,
)
from moleculegen.data import (
    SMILESDataset,
    SMILESVocabulary,
)
from moleculegen.generation import GreedySearch
from moleculegen.tests.utils import TempSMILESFile


class GreedySearchTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(
            tempfile_kwargs={'prefix': 'greedy_search'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)

        embedding_layer = gluon.nn.Embedding(len(self.vocabulary), 4)
        rnn_layer = gluon.rnn.LSTM(hidden_size=16, num_layers=1)
        dense_layer = gluon.nn.Dense(len(self.vocabulary), flatten=True)
        self.model = SMILESRNNModel(
            embedding_layer=embedding_layer,
            rnn_layer=rnn_layer,
            dense_layer=dense_layer,
        )

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
