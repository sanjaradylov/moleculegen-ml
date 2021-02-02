"""
Test losses and metrics.
"""

import unittest

import mxnet as mx
from rdkit.RDLogger import DisableLog

from moleculegen.evaluation import (
    get_mask_for_loss,
    KLDivergence,
    MaskedSoftmaxCELoss,
    Novelty,
    Perplexity,
    RAC,
    Uniqueness,
    Validity,
)


class MaskedSoftmaxCELossTestCase(unittest.TestCase):
    def test_forward(self):
        batch_size, n_steps, vocab_dim = 4, 8, 16
        loss_fn = MaskedSoftmaxCELoss()
        # noinspection PyUnresolvedReferences
        predictions = mx.np.repeat(
            mx.np.random.uniform(size=(vocab_dim, n_steps)),
            batch_size,
        ).reshape(vocab_dim, n_steps, batch_size).T
        # noinspection PyUnresolvedReferences
        labels = mx.np.repeat(
            mx.np.random.randint(0, vocab_dim, size=n_steps),
            batch_size,
        ).reshape(n_steps, batch_size).T
        valid_lengths = mx.np.array([4, 3, 2, 1])

        # noinspection PyArgumentList
        loss = loss_fn(predictions, labels, valid_lengths)

        self.assertTrue(all(x > y for x, y in zip(loss[:-1], loss[1:])))

    def test_get_mask_for_loss(self):
        output_shape = (4, 8)
        valid_lengths = mx.np.array([8, 7, 6, 4])
        label_mask = get_mask_for_loss(output_shape, valid_lengths)

        self.assertTupleEqual(label_mask.shape, output_shape + (1,))
        self.assertTrue(all(
            a == b for a, b in zip(
                label_mask.sum(axis=1).squeeze(),
                valid_lengths
            )
        ))


class NoveltyTestCase(unittest.TestCase):
    def setUp(self):
        self.novelty = Novelty()

        self.predictions = [
            '(((((',
            'N#N',
            'N#N',
            'CN=C=O',
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

    def test_1(self):
        self.novelty.update(predictions=self.predictions, labels=self.labels)
        name, result = self.novelty.get()

        self.assertEqual(name, 'Novelty')
        self.assertEqual(result, 0.8)

    def test_2_perfect_score(self):
        self.predictions.remove('CN=C=O')
        self.novelty.reset()
        self.novelty.update(predictions=set(self.predictions), labels=self.labels)
        _, result = self.novelty.get()

        self.assertEqual(result, 1.0)


class UniquenessTestCase(unittest.TestCase):
    def setUp(self):
        self.uniqueness = Uniqueness()

        self.predictions = [
            '(((((',
            'N#N',
            'N#N',
            'CN=C=O',
            'CN=C=O',
            '[Cu+2].[O-]S(=O)(=O)[O-]',
        ]

    def test_1(self):
        self.uniqueness.update(predictions=self.predictions)
        name, result = self.uniqueness.get()

        self.assertEqual(name, 'Uniqueness')
        self.assertAlmostEqual(float(result), 0.67, 2)

    def test_2_perfect_score(self):
        self.uniqueness.reset()
        self.uniqueness.update(predictions=set(self.predictions))
        _, result = self.uniqueness.get()

        self.assertEqual(result, 1.0)


class ValidityTestCase(unittest.TestCase):
    def setUp(self):
        self.validity = Validity()

        self.predictions = [
            # Valid
            'CN=C=O',
            'CN1CCC[C@H]1c2cccnc2',
            'CC[C@H](O1)CC[C@@]12CCCO2',
            'CN1CCC[C@H]1c2cccnc2',
            'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2',
            'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N',
            # Invalid
            '(((((',
        ]

    def test_1(self):
        self.validity.update(predictions=self.predictions)
        name, result = self.validity.get()

        self.assertEqual(name, 'Validity')
        self.assertAlmostEqual(float(result), 0.86, 2)

    def test_2_perfect_score(self):
        self.predictions.pop()  # Remove invalid SMILES.
        self.validity.reset()
        self.validity.update(predictions=self.predictions)
        _, result = self.validity.get()

        self.assertEqual(result, 1.0)


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


class KLDivergenceTestCase(unittest.TestCase):
    def setUp(self):
        self.metric = KLDivergence()

    def test_1_same_inputs(self):
        train = (
            'CN=C=O',
            'CN1CCC[C@H]1c2cccnc2',
            'CC[C@H](O1)CC[C@@]12CCCO2',
            'CN1CCC[C@H]1c2cccnc2',
            'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2',
            'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N',
        )
        valid = ('CCO', 'N#N', 'CN=C=O')
        self.metric.update(predictions=valid, labels=train)

        self.assertLess(self.metric.get()[1], 0.1)


class PerplexityTestCase(unittest.TestCase):
    def setUp(self):
        self.predictions = mx.np.array([
            [[0.1, 0.15, 0.75],
             [0.9, 0.01, 0.09],
             [0.0, 0.35, 0.65],
             [0.05, 0.45, 0.5]],
            [[0.2, 0.35, 0.45],
             [0.5, 0.11, 0.39],
             [0.1, 0.85, 0.05],
             [0.25, 0.45, 0.3]],
        ])
        self.labels = mx.np.array([
            [2, 0, 2, 1],
            [2, 2, 1, 1],
        ])

    def test_1_without_ignored_labels(self):
        perplexity = Perplexity(from_probabilities=True)
        perplexity.update(labels=self.labels, preds=self.predictions)

        self.assertAlmostEqual(perplexity.get()[1], 1.717, 3)

    def test_2_with_ignored_labels(self):
        perplexity = Perplexity(from_probabilities=True, ignore_label=0)
        perplexity.update(labels=self.labels, preds=self.predictions)

        self.assertAlmostEqual(perplexity.get()[1], 1.827, 3)

    def test_3_not_from_probabilities(self):
        log_predictions = mx.np.log(self.predictions)

        perplexity = Perplexity()
        perplexity.update(labels=self.labels, preds=log_predictions)

        self.assertAlmostEqual(perplexity.get()[1], 1.717, 3)


if __name__ == '__main__':
    DisableLog('rdApp.*')
    mx.npx.set_np()
    unittest.main()
