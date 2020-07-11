"""
Compute losses, evaluate models.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.

Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
"""

__all__ = (
    'MaskedSoftmaxCELoss',
    'Perplexity',
)


from .loss import (
    MaskedSoftmaxCELoss,
)
from .metric import (
    Perplexity,
)
