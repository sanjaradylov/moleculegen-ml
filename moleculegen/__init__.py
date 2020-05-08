"""
Package for generation of novel molecules using machine learning methods.
"""

__author__ = 'Sanjar Ad[iy]lov'
__maintainer__ = 'Sanjar Ad[iy]lov'
__status__ = 'Development'
__version__ = 'beta'

from . import description

from .base import Token
from .data import (
    SMILESDataset,
    SMILESDataLoader,
)
from .model import (
    OneHotEncoder,
    SMILESRNNModel,
)
from .utils import (
    Perplexity,
    get_mask_for_loss,
)
from .vocab import (
    Vocabulary,
    count_tokens,
    tokenize,
)


__all__ = [
    # Modules.
    'description',

    # Functions.
    'count_tokens',
    'get_mask_for_loss',
    'tokenize',

    # Classes.
    'OneHotEncoder',
    'Perplexity',
    'SMILESDataLoader',
    'SMILESDataset',
    'SMILESRNNModel',
    'Token',
    'Vocabulary',
]
