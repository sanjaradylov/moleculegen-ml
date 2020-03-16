"""
Test moleculegen.utils.
"""

import unittest

from mxnet import np, npx

from moleculegen.utils import Perplexity


class PerplexityTestCase(unittest.TestCase):
    def setUp(self):
        self.predictions = np.array([
            [0.1, 0.15, 0.75],
            [0.9, 0.01, 0.09],
            [0.0, 0.35, 0.65],
            [0.05, 0.45, 0.5],
        ])
        self.labels = np.array([2, 0, 2, 1])

    def test_without_ignored_labels(self):
        perplexity = Perplexity(ignore_label=None)
        perplexity.update(labels=self.labels, predictions=self.predictions)

        self.assertAlmostEqual(perplexity.get()[1], 1.5, 3)

    def test_with_ignored_labels(self):
        perplexity = Perplexity(ignore_label=0)
        perplexity.update(labels=self.labels, predictions=self.predictions)

        self.assertAlmostEqual(perplexity.get()[1], 1.658, 3)


if __name__ == '__main__':
    npx.set_np()
    unittest.main()
