"""
Generate SMILES sequences using various search methods.

Classes
-------
BaseSearch
    The base SMILES sampler class.

ArgmaxSearch
    Generate new SMILES strings using greedy search (argmax) method.

SoftmaxSearch
    Softmax with a sensitivity parameter followed by sampling from multinomial
    distribution.
GumbelSoftmaxSearch
    Gumbel-Softmax with a sensitivity parameter followed by sampling from multinomial
    distribution.

GreedySearch
    Generate new SMILES strings using greedy search methods.
"""

__all__ = (
    'ArgmaxSearch',
    'BaseSearch',
    'GreedySearch',
    'GumbelSoftmaxSearch',
    'SoftmaxSearch',
)


from .greedy_search import GreedySearch
from .search import (
    ArgmaxSearch,
    BaseSearch,
    GumbelSoftmaxSearch,
    SoftmaxSearch,
)
