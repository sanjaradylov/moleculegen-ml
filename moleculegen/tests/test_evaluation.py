"""
Test losses and metrics.
"""

import unittest

from mxnet import np, npx
from rdkit.RDLogger import DisableLog

from moleculegen.evaluation import (
    get_mask_for_loss,
    MaskedSoftmaxCELoss,
    Perplexity,
    RAC,
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


class RUACTestCase(unittest.TestCase):
    def setUp(self):
        self.rac = RAC(count_unique=False)
        self.ruac = RAC(name='RUAC', count_unique=True)

        self.predictions = [
            # Invalid:
            '(((((',
            # Unique but duplicates:
            'N#N',
            'N#N',
            # Presented in `self.labels`:
            'CN=C=O',
            # Unique:
            '[Cu+2].[O-]S(=O)(=O)[O-]',
        ]
        self.labels = [
            'CN=C=O',
            'CN1CCC[C@H]1c2cccnc2',
            'CC[C@H](O1)CC[C@@]12CCCO2',
            'CN1CCC[C@H]1c2cccnc2',
            'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2',
            'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N',
        ]

    def test_1_rac_without_labels(self):
        self.rac.update(predictions=self.predictions, labels=None)
        name, result = self.rac.get()
        self.rac.reset()

        self.assertEqual('RAC', name)
        self.assertEqual(result, 4/5)

    def test_2_rac_with_labels(self):
        self.rac.update(predictions=self.predictions, labels=self.labels)
        _, result = self.rac.get()
        self.rac.reset()

        self.assertEqual(result, 3/5)

    def test_3_ruac_without_labels(self):
        self.ruac.update(predictions=self.predictions, labels=None)
        name, result = self.ruac.get()
        self.ruac.reset()

        self.assertEqual('RUAC', name)
        self.assertEqual(result, 3/5)

    def test_4_ruac_with_labels(self):
        self.ruac.update(predictions=self.predictions, labels=self.labels)
        _, result = self.ruac.get()
        self.ruac.reset()

        self.assertEqual(result, 2/5)


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
    DisableLog('rdApp.*')
    npx.set_np()
    unittest.main()
