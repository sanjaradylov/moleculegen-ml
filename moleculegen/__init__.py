"""
Package for generation of novel molecules using machine learning methods.
"""

__author__ = 'Sanjar Ad[iy]lov'
__maintainer__ = 'Sanjar Ad[iy]lov'
__status__ = 'Development'
__version__ = 'beta'


from .data import (
    SMILESDataset,
    SMILESDataLoader,
)
from .model import (
    OneHotEncoder,
    SMILESRNNModel,
)
from .utils import (
    SpecialTokens,
    get_mask_for_loss,
)
from .vocab import (
    Vocabulary,
    count_tokens,
    tokenize,
)


__all__ = [
    'SpecialTokens',
    'get_mask_for_loss',
    'count_tokens',
    'tokenize',
    'OneHotEncoder',
    'SMILESDataLoader',
    'SMILESDataset',
    'SMILESRNNModel',
    'Vocabulary',
]
