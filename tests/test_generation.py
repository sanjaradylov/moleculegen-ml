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
from moleculegen.estimation import SMILESRNN
from moleculegen.generation.search import SoftmaxSearch, ArgmaxSearch
from .utils import TempSMILESFile


class ArgmaxSearchTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'argmax_sampler'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)

        self.model = SMILESRNN(len(self.vocabulary))

        self.predictor = ArgmaxSearch(self.model, self.vocabulary)

    def test_1_distribution(self):
        logits = mx.np.array([-10.5, -10.7, 0.5, 20.3, 30.1])
        next_token_id = self.predictor.distribution(self.predictor.normalizer(logits))

        self.assertEqual(next_token_id, logits.argmax().item())

    def test_2_tokens_are_valid(self):
        valid_tokens = Token.get_all_tokens() - frozenset(Token.UNK)

        for _ in range(100):
            smiles = self.predictor()

            for token in Token.tokenize(smiles):
                if len(token) == 1 and token.islower():
                    token = token.upper()

                self.assertIn(token, valid_tokens)

    def tearDown(self):
        self.fh.close()


class SoftmaxSearchTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'softmax_sampler'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)

        self.model = SMILESRNN(len(self.vocabulary))

        self.predictor = SoftmaxSearch(self.model, self.vocabulary)

    def test_1_normalizer(self):
        # Check if probabilities from `self.predictor` sum to 1.
        logits = mx.np.random.uniform(low=-10, high=20, size=len(self.vocabulary))
        probabilities = self.predictor.normalizer(logits)

        self.assertAlmostEqual(probabilities.sum().item(), 1.0, 3)

        # Check if high temperature gives almost equal probabilities.
        self.predictor.temperature = 100 * logits.max().item()
        probabilities = self.predictor.normalizer(logits)

        expected_prob = 1 / len(self.vocabulary)
        for prob in probabilities:
            self.assertAlmostEqual(prob, expected_prob, 2)

    def test_2_strategy_p(self):
        with self.assertRaises(ValueError):
            self.predictor.strategy = 2.5  # must be in (0., 1.]

        y = mx.np.array([0.2, 0.22, 0.08, 0.1, 0.15, 0.02, 0.015, 0.004, 0.18, 0.031])
        self.predictor.strategy = 0.9
        top_p_idx = (0, 1, 8, 4, 3, 2)

        for _ in range(1_000):
            self.assertIn(self.predictor.distribution(y), top_p_idx)

    def test_3_strategy_k(self):
        with self.assertRaises(ValueError):
            self.predictor.strategy = 0

        y = mx.np.array([0.01, 0.5, 0.3, 0.1, 0.02, 0.04, 0.03])
        self.predictor.strategy = 3
        top_k_idx = (1, 2, 3)

        for _ in range(1_000):
            self.assertIn(self.predictor.distribution(y), top_k_idx)

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    mx.npx.set_np()
    unittest.main()
