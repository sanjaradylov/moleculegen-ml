"""
Additional loss functions (classes) for training neural networks.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.

Functions
---------
get_mask_for_loss
    Return the mask for valid tokens.
"""

__all__ = (
    'get_mask_for_loss',
    'MaskedSoftmaxCELoss',
)


from typing import Tuple
from mxnet import gluon, np, npx


def get_mask_for_loss(
        output_shape: Tuple[int, int],
        valid_lengths: np.ndarray,
) -> np.ndarray:
    """Get a boolean mask for the output tokens. Labels 1 for valid tokens
    and 0 for Token.PAD. Use as a weight mask for SoftmaxCELoss or similar
    losses.

    Parameters
    ----------
    output_shape : tuple of int
        The shape of output sequences, i.e. (batch size, time steps).
    valid_lengths : mxnet.np.ndarray, shape = (batch size,)
        The number of valid tokens for every entry in the output sequences.

    Returns
    -------
    label_mask : mxnet.np.ndarray, shape = (batch size, time steps, 1)

    See also
    --------
    MaskedSoftmaxCELoss
    """
    label_mask = np.expand_dims(
        np.ones(output_shape, ctx=valid_lengths.ctx), axis=-1)
    # noinspection PyUnresolvedReferences
    label_mask = npx.sequence_mask(
        label_mask,
        valid_lengths,
        use_sequence_length=True,
        value=0,
        axis=1,
    )
    return label_mask


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
