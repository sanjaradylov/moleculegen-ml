"""
Create, train, evaluate, and fine-tune generative models.

Classes
-------
SMILESRNNModel
    A generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESRNNModel',
)


from .model import SMILESRNNModel
