"""
Utilities.
"""

import enum
from typing import Tuple

from mxnet import nd


@enum.unique
class SpecialTokens(enum.Enum):
    """Enumeration of special tokens.
    """
    BOS = '^'  # beginning of string
    EOS = '\n'  # end of string
    UNK = '_'  # unknown token
    PAD = '*'  # padding


def get_mask_for_loss(
        label_shape: Tuple[int, ...],
        valid_lengths: nd.NDArray,
) -> nd.NDArray:
    """Get mask of valid labels, i.e. SpecialTokens.PAD.value is invalid,
    therefore filled with zeros, and other tokens retain their weights = 1.

    Parameters
    ----------
    label_shape : tuple of int
        Label shape.
    valid_lengths : nd.NDArray, shape = label_shape[0]
        For every entry in labels, specified valid token length.

    Returns
    -------
    label_mask : nd.NDArray, shape = label_shape
        Mask of valid labels.
    """
    label_weights = nd.ones(label_shape).expand_dims(axis=-1)
    label_mask = nd.SequenceMask(label_weights, valid_lengths, True, axis=1)
    return label_mask
