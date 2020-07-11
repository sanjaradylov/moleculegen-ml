"""
Test `SMILESRNNModel` class and its main components.
"""

import unittest

from mxnet import gluon, np, npx

from moleculegen.description import OneHotEncoder
from moleculegen.model import SMILESRNNModel


class SMILESRNNModelTestCase(unittest.TestCase):
    def setUp(self):
        # OneHotEncoder.
        self.depth = 4
        self.embedding_layer = OneHotEncoder(self.depth)

        # LSTM.
        self.n_units = 16
        self.n_hidden = 2
        self.rnn_layer = gluon.rnn.LSTM(self.n_units, self.n_hidden)

        # Dense.
        self.dense_layer = gluon.nn.Dense(self.depth)

        # Model.
        self.model = SMILESRNNModel(
            self.embedding_layer, self.rnn_layer, self.dense_layer)
        self.model.initialize()

        # Initialize categories.
        self.batch_shape = (64, 8)  # Number of samples and time steps.
        self.samples = np.random.randint(0, 6, size=self.batch_shape)

    def test_intermediate_steps(self):
        # One-hot encoding.
        embedding = self.embedding_layer(self.samples.T)

        self.assertTupleEqual(
            # Time steps, batch size, depth.
            embedding.shape,
            (self.batch_shape[1], self.batch_shape[0], self.depth),
        )
        self.assertEqual(embedding.asnumpy().dtype.kind, 'f')

        # Memory cell and hidden state.
        states = self.model.begin_state(self.samples.shape[0])

        self.assertEqual(len(states), 2)
        self.assertTupleEqual(states[0].shape, states[1].shape)
        true_state_shape = (self.n_hidden, self.batch_shape[0], self.n_units)
        self.assertTupleEqual(states[0].shape, true_state_shape)

        # RNN outputs.
        outputs, states = self.rnn_layer(embedding, states)
        # Time steps, batch size, hidden layer units.
        true_outputs_shape = embedding.shape[:2] + (self.n_units,)
        self.assertTupleEqual(outputs.shape, true_outputs_shape)
        self.assertTupleEqual(states[0].shape, true_state_shape)
        self.assertTupleEqual(states[1].shape, true_state_shape)

    def test_model(self):
        self.model.initialize(force_reinit=True)
        states = self.model.begin_state(self.samples.shape[0])
        outputs, states = self.model(self.samples, states)

        self.assertTupleEqual(
            outputs.shape,
            # For every time step, we obtain batches with the corresponding
            # output scores of size equalling the number of unique categories.
            (self.batch_shape[1] * self.batch_shape[0], self.depth),
        )

        weights = self.model.collect_params()
        self.assertEqual(
            # (W_i2h, b_i2h, W_h2h, b_h2h) * n_hidden, W_h2o, b_h2o.
            len(weights.items()),
            self.n_hidden * 4 + 2,
        )


if __name__ == '__main__':
    npx.set_np()
    unittest.main()
