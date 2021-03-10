"""
Compute losses, evaluate models.

Classes:
    MaskedSoftmaxCELoss
        Softmax cross-entropy loss that ignores padded tokens.

    Metric:
        A metric ABC for the comparison of training/validation and generated compounds.
    CompositeMetric:
        Calculate a series of metrics on the same predictions and labels.

    Novelty:
        Calculate the rate of compounds not presented in the training set.
    Uniqueness:
        Calculate the rate of unique generated compounds.
    Validity:
        Calculate the rate of valid (theoretically realistic) molecules.
    RAC:
        Calculate the rate of unique acceptable compounds.
    KLDivergence:
        Calculate the Kullback-Leibler divergence of physicochemical descriptors.
    InternalDiversity:
        Calculate internal diversity using Tanimoto similarity.

    Perplexity:
        Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
        ndarray.

Functions:
    get_mask_for_loss: Return the mask for valid tokens.
"""

__all__ = (
    'get_mask_for_loss',
    'CompositeMetric',
    'InternalDiversity',
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
    InternalDiversity,
    KLDivergence,
    Metric,
    Novelty,
    Perplexity,
    RAC,
    Uniqueness,
    Validity,
)
