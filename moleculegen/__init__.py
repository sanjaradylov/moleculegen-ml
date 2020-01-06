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
    SMILESRNNModel,
)
from .utils import (
    EOF,
)
from .vocab import (
    Vocabulary,
    count_tokens,
    tokenize,
)


__all__ = [
    'EOF',
    'count_tokens',
    'tokenize',
    'SMILESDataLoader',
    'SMILESDataset',
    'SMILESRNNModel',
    'Vocabulary',
]
