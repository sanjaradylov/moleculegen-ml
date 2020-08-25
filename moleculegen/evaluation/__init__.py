"""
Compute losses, evaluate models.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.

Metric
    A metric ABC for the comparison of training/validation and generated compounds.
Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
RAC
    Calculate the rate of unique acceptable compounds.

Functions
---------
get_mask_for_loss
    Return the mask for valid tokens.
"""

__all__ = (
    'get_mask_for_loss',
    'MaskedSoftmaxCELoss',
    'Metric',
    'Perplexity',
    'RAC',
)


from .loss import (
    get_mask_for_loss,
    MaskedSoftmaxCELoss,
)
from .metric import (
    Metric,
    Perplexity,
    RAC,
)
