"""
Compute losses, evaluate models.

Classes
-------
MaskedSoftmaxCELoss
    Softmax cross-entropy loss that ignores padded tokens.

CompositeMetric
    Calculate a series of metrics on the same predictions and labels.
Metric
    A metric ABC for the comparison of training/validation and generated compounds.
Novelty
    Calculate the rate of compounds not presented in the training set.
Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
RAC
    Calculate the rate of unique acceptable compounds.
Uniqueness
    Calculate the rate of unique generated compounds.
Validity
    Calculate the rate of valid (theoretically realistic) molecules.

Functions
---------
get_mask_for_loss
    Return the mask for valid tokens.
"""

__all__ = (
    'get_mask_for_loss',
    'CompositeMetric',
    'KLDivergence',
    'MaskedSoftmaxCELoss',
    'Metric',
    'Novelty',
    'Perplexity',
    'RAC',
    'Uniqueness',
    'Validity',
)


from .loss import (
    get_mask_for_loss,
    MaskedSoftmaxCELoss,
)
from .metric import (
    CompositeMetric,
    KLDivergence,
    Metric,
    Novelty,
    Perplexity,
    RAC,
    Uniqueness,
    Validity,
)
