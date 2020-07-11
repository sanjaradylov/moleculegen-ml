"""
Additional loss functions (classes) for training neural networks.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.
"""

__all__ = (
    'MaskedSoftmaxCELoss',
)


from mxnet import gluon, np, npx


class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """Softmax cross-entropy loss that ignores padded tokens.
    """

    # noinspection PyMethodOverriding
    def forward(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            valid_lengths: np.ndarray,
    ) -> np.ndarray:
        """Compute softmax cross-entropy loss creating masks for token ids.

        Parameters
        ----------
        predictions : np.ndarray,
                shape = (batch size, time steps, vocabulary dimension)
            Predicted probabilities.
        labels : np.ndarray, shape = (batch size, time steps)
            True labels.
        valid_lengths : np.ndarray, shape = (batch size,) or
                ??? shape = (batch size, time steps)
            For every entry in labels, specified valid token length.

        Returns
        -------
        loss : np.ndarray, shape = (batch size,)
            Computed loss.
        """
        # noinspection PyUnresolvedReferences
        weight_mask = np.expand_dims(np.ones_like(labels), axis=-1)
        # noinspection PyUnresolvedReferences
        weight_mask = npx.sequence_mask(
            weight_mask,
            valid_lengths,
            use_sequence_length=True,
            value=0,
            axis=1,
        )
        # noinspection PyTypeChecker
        return super().forward(predictions, labels, weight_mask)
