"""
Test models and trainers.
"""

import unittest

import mxnet as mx
from mxnet import gluon

from moleculegen.callback import ProgressBar
from moleculegen.data import (
    SMILESDataset,
    SMILESBatchColumnSampler,
    SMILESVocabulary,
)
from moleculegen.estimation import SMILESEncoderDecoder
from moleculegen.tests.utils import TempSMILESFile


class SMILESEncoderDecoderTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'model'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        vocabulary = SMILESVocabulary(dataset, need_corpus=True)
        self.batch_sampler = SMILESBatchColumnSampler(
            vocabulary=vocabulary,
            batch_size=3,
            n_steps=8,
        )

        self.n_rnn_layers = 1  # Used in output/state shape testing.
        self.n_rnn_units = 32  # Used in output/state shape testing.

        self.model = SMILESEncoderDecoder(
            len(vocabulary),
            use_one_hot=False,
            embedding_dim=4,
            embedding_init=mx.init.Orthogonal(),
            rnn='lstm',
            n_rnn_layers=self.n_rnn_layers,
            n_rnn_units=self.n_rnn_units,
            rnn_dropout=0.0,
            rnn_init=mx.init.Normal(),
            n_dense_layers=1,
            n_dense_units=64,
            dense_activation='relu',
            dense_dropout=0.0,
            dense_init=mx.init.Xavier(),
            dtype='float32',
        )

    def test_1_params(self):
        param_names = (
            'embedding0_weight',

            'lstm0_l0_i2h_weight', 'lstm0_l0_h2h_weight',
            'lstm0_l0_i2h_bias', 'lstm0_l0_h2h_bias',

            'dense0_weight', 'dense0_bias',
        )

        for actual_p, test_p in zip(param_names, self.model.collect_params()):
            self.assertEqual(actual_p, test_p)

    def test_2_outputs(self):
        inputs = next(iter(self.batch_sampler)).inputs
        states = self.model.begin_state(self.batch_sampler.batch_size)
        outputs, states = self.model(inputs, states)

        self.assertTupleEqual(
            (
                self.batch_sampler.batch_size,
                self.batch_sampler.n_steps,
                len(self.batch_sampler.vocabulary),
            ),
            outputs.shape,
        )
        self.assertTupleEqual(
            (
                self.n_rnn_layers,
                self.batch_sampler.batch_size,
                self.n_rnn_units,
            ),
            states[0].shape,
        )

    def test_3_fit(self):
        """Test fit method 'visually', without assertions. See bars and logs to ensure
        that training progresses.
        """
        callbacks = [ProgressBar()]
        self.model.fit(
            batch_sampler=self.batch_sampler,
            optimizer=mx.optimizer.Adam(learning_rate=0.005),
            loss_fn=gluon.loss.SoftmaxCELoss(),
            n_epochs=20,
            callbacks=callbacks,
        )

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    mx.npx.set_np()
    unittest.main()
