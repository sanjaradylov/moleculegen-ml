"""
Compute losses, evaluate models.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.

Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.

Functions
---------
get_mask_for_loss
    Return the mask for valid tokens.
"""

__all__ = (
    'get_mask_for_loss',
    'MaskedSoftmaxCELoss',
    'Perplexity',
)


from .loss import (
    get_mask_for_loss,
    MaskedSoftmaxCELoss,
)
from .metric import (
    Perplexity,
)
