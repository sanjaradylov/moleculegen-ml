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
from moleculegen.estimation import SMILESRNN
from .utils import TempSMILESFile


class SMILESRNNTestCase(unittest.TestCase):
    def setUp(self):
        temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'model'})
        self.fh = temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)
        self.batch_sampler = SMILESBatchColumnSampler(
            corpus=self.vocabulary.corpus,
            batch_size=3,
            n_steps=8,
        )

        self.n_rnn_layers = 1  # Used in output/state shape testing.
        self.n_rnn_units = 32  # Used in output/state shape testing.

        self.model = SMILESRNN(
            len(self.vocabulary),

            use_one_hot=False,
            embedding_dim=4,
            embedding_dropout=0.25,
            embedding_dropout_axes=0,
            embedding_init=mx.init.Uniform(),
            embedding_prefix='embedding_',

            rnn='lstm',
            rnn_n_layers=self.n_rnn_layers,
            rnn_n_units=self.n_rnn_units,
            rnn_i2h_init='xavier_normal',
            rnn_h2h_init='orthogonal_normal',
            rnn_reinit_state=True,
            rnn_detach_state=False,
            rnn_state_init=mx.nd.random.uniform,
            rnn_dropout=0.0,
            rnn_prefix='encoder_',

            dense_n_layers=2,
            dense_n_units=32,
            dense_activation='relu',
            dense_dropout=0.5,
            dense_init=mx.init.Xavier(),
            dense_prefix='decoder_',

            dtype='float32',
            prefix='model_'
        )

    def test_1_params(self):
        param_names = (
            'model_embedding_weight',

            'model_encoder_l0_i2h_weight', 'model_encoder_l0_h2h_weight',
            'model_encoder_l0_i2h_bias', 'model_encoder_l0_h2h_bias',

            'model_decoder_l0_weight', 'model_decoder_l0_bias',
            'model_decoder_out_weight', 'model_decoder_out_bias',
        )

        for actual_p, test_p in zip(param_names, self.model.collect_params()):
            self.assertEqual(actual_p, test_p)

    def test_2_outputs(self):
        batch = next(iter(self.batch_sampler))
        self.model.begin_state(self.batch_sampler.batch_size)
        outputs = self.model(batch)

        self.assertTupleEqual(
            (
                self.batch_sampler.batch_size,
                self.batch_sampler.n_steps,
                len(self.vocabulary),
            ),
            outputs.shape,
        )
        self.assertTupleEqual(
            (
                self.n_rnn_layers,
                self.batch_sampler.batch_size,
                self.n_rnn_units,
            ),
            self.model.state[0].shape,
        )

    def test_3_fit(self):
        """Test fit method 'visually', without assertions. See bars and logs to ensure
        that training progresses.
        """
        callbacks = [ProgressBar()]
        # noinspection PyTypeChecker
        self.model.fit(
            batch_sampler=self.batch_sampler,
            optimizer=mx.optimizer.Adam(learning_rate=0.005),
            loss_fn=gluon.loss.SoftmaxCELoss(),
            n_epochs=10,
            callbacks=callbacks,
            verbose=True,
        )

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    mx.npx.set_np()
    unittest.main()
