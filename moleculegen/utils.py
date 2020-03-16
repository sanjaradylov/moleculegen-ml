"""
Utilities.
"""

import enum
from typing import NamedTuple, Sequence, Tuple, Union

from mxnet import metric, np, npx


@enum.unique
class SpecialTokens(enum.Enum):
    """Enumeration of special tokens.
    """
    BOS = '^'  # beginning of string
    EOS = '\n'  # end of string
    UNK = '_'  # unknown token
    PAD = '*'  # padding


class Batch(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    v_x: np.ndarray
    v_y: np.ndarray
    s: bool


Batch.__doc__ += ": Named tuple that stores mini-batch items"
Batch.x.__doc__ += "\nInput samples."
Batch.y.__doc__ += "\nOutput samples."
Batch.v_x.__doc__ += "\nValid lengths for input samples."
Batch.v_y.__doc__ += "\nValid lengths for output samples."
Batch.s.__doc__ += "\nWhether to initialize state or not."


class Perplexity(metric.Perplexity):
    """Re-implementation of mxnet.metrics.Perplexity that supports Numpy
    ndarrays. See the documentation for more information.
    """

    def update(
            self,
            labels: Union[np.ndarray, Sequence[np.ndarray]],
            predictions: Union[np.ndarray, Sequence[np.ndarray]],
    ):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : sequence of mxnet.np.ndarray
            The labels of the data.

        predictions : sequence of mxnet.np.ndarray
            Predicted values.

        See the documentation for more information.
        """
        if len(labels) != len(predictions):
            raise ValueError(
                f"Labels vs. Predictions size mismatch: "
                f"{len(labels)} vs. {len(predictions)}"
            )

        loss = 0.0
        num = 0

        for label, prediction in zip(labels, predictions):
            probability = prediction[label.astype(np.int32).item()]
            if self.ignore_label is not None:
                ignore = (label == self.ignore_label).astype(prediction.dtype)
                num -= ignore.astype(np.int32).item()
                probability = probability * (1 - ignore) + ignore
            loss -= np.sum(np.log(np.maximum(1e-10, probability))).item()
            num += 1

        self.sum_metric += loss
        self.global_sum_metric += loss
        self.num_inst += num
        self.global_num_inst += num


def get_mask_for_loss(
        label_shape: Tuple[int, ...],
        valid_lengths: np.ndarray,
) -> np.ndarray:
    """Get mask of valid labels, i.e. SpecialTokens.PAD.value is invalid,
    therefore filled with zeros, and other tokens retain their weights = 1.

    Parameters
    ----------
    label_shape : tuple of int
        Label shape.
    valid_lengths : np.ndarray, shape = label_shape[0]
        For every entry in labels, specified valid token length.

    Returns
    -------
    label_mask : np.ndarray, shape = label_shape
        Mask of valid labels.
    """
    label_weights = np.expand_dims(np.ones(label_shape), axis=-1)
    label_mask = npx.sequence_mask(label_weights, valid_lengths, True, axis=1)
    return label_mask
