"""
Test callbacks.
"""

import time
import unittest

import mxnet as mx

from moleculegen.callback import ProgressBar


class ProgressBarTestCase(unittest.TestCase):
    """Test ProgressBar visually, without assertions.
    """

    def test_1_visually(self):
        """Imitate model training.
        """
        callback = ProgressBar()

        n_epochs = 2
        n_batches = 40
        batch_sampler = list(range(n_batches))

        callback.on_train_begin(batch_sampler=batch_sampler)

        for epoch in range(1, n_epochs + 1):
            callback.on_epoch_begin(
                batch_sampler=batch_sampler, n_epochs=n_epochs, epoch=epoch)
            losses = mx.np.random.uniform(low=-100, high=200, size=n_batches)

            for (batch_no, batch), loss in zip(
                    enumerate(batch_sampler, start=1), losses):
                callback.on_batch_begin()
                time.sleep(0.15 + mx.np.random.uniform(0, 0.1).item())
                callback.on_batch_end(loss=loss, batch_no=batch_no)

            callback.on_epoch_end()

        callback.on_train_end()


if __name__ == '__main__':
    mx.npx.set_np()
    unittest.main()
