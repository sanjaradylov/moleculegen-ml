"""
Test losses and metrics.
"""

import unittest

from mxnet import np, npx

from moleculegen.evaluation import (
    get_mask_for_loss,
    MaskedSoftmaxCELoss,
    Perplexity,
)


class MaskedSoftmaxCELossTestCase(unittest.TestCase):
    def test_forward(self):
        batch_size, n_steps, vocab_dim = 4, 8, 16
        loss_fn = MaskedSoftmaxCELoss()
        # noinspection PyUnresolvedReferences
        predictions = np.repeat(
            np.random.uniform(size=(vocab_dim, n_steps)),
            batch_size,
        ).reshape(vocab_dim, n_steps, batch_size).T
        # noinspection PyUnresolvedReferences
        labels = np.repeat(
            np.random.randint(0, vocab_dim, size=n_steps),
            batch_size,
        ).reshape(n_steps, batch_size).T
        valid_lengths = np.array([4, 3, 2, 1])

        # noinspection PyArgumentList
        loss = loss_fn(predictions, labels, valid_lengths)

        self.assertTrue(all(x > y for x, y in zip(loss[:-1], loss[1:])))

    def test_get_mask_for_loss(self):
        output_shape = (4, 8)
        valid_lengths = np.array([8, 7, 6, 4])
        label_mask = get_mask_for_loss(output_shape, valid_lengths)

        self.assertTupleEqual(label_mask.shape, output_shape + (1,))
        self.assertTrue(all(
            a == b for a, b in zip(
                label_mask.sum(axis=1).squeeze(),
                valid_lengths
            )
        ))


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
