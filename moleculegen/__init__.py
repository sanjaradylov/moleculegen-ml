"""
Package for generation of novel molecules using machine learning methods.
"""

__author__ = 'Sanjar Ad[iy]lov'
__maintainer__ = 'Sanjar Ad[iy]lov'
__status__ = 'Development'
__version__ = 'beta'

__all__ = (
    # Modules.
    'data',
    'description',
    'evaluation',

    # Classes.
    'SMILESRNNModel',
    'StateInitializerMixin',
    'Token',
)


from . import data
from . import description
from . import evaluation

from .base import (
    StateInitializerMixin,
    Token,
)
from .model import (
    SMILESRNNModel,
)
