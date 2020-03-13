"""
Utilities.
"""

import enum
from typing import NamedTuple, Tuple

from mxnet import np, npx


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
