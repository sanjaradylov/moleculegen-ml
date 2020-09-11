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

import mxnet as mx
import mxnet.gluon as gluon


def get_mask_for_loss(
        output_shape: Tuple[int, int],
        valid_lengths: mx.np.ndarray,
) -> mx.np.ndarray:
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
    label_mask = mx.np.expand_dims(
        mx.np.ones(output_shape, ctx=valid_lengths.ctx), axis=-1)
    # noinspection PyUnresolvedReferences
    label_mask = mx.npx.sequence_mask(
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
            predictions: mx.np.ndarray,
            labels: mx.np.ndarray,
            valid_lengths: mx.np.ndarray,
    ) -> mx.np.ndarray:
        """Compute softmax cross-entropy loss creating masks for token ids.

        Parameters
        ----------
        predictions : mxnet.np.ndarray,
                shape = (batch size, time steps, vocabulary dimension)
            Predicted probabilities.
        labels : mxnet.np.ndarray, shape = (batch size, time steps)
            True labels.
        valid_lengths : mxnet.np.ndarray, shape = (batch size,) or
                ??? shape = (batch size, time steps)
            For every entry in labels, specified valid token length.

        Returns
        -------
        loss : mxnet.np.ndarray, shape = (batch size,)
            Computed loss.
        """
        weight_mask = mx.np.expand_dims(mx.np.ones_like(labels), axis=-1)
        # noinspection PyUnresolvedReferences
        weight_mask = mx.npx.sequence_mask(
            weight_mask,
            valid_lengths,
            use_sequence_length=True,
            value=0,
            axis=1,
        )
        # noinspection PyTypeChecker
        return super().forward(predictions, labels, weight_mask)
