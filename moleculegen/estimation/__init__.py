"""
Create, train, evaluate, and fine-tune generative models.

Classes
-------
SMILESEncoderDecoder
    A generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESEncoderDecoder',
)


from .model import SMILESEncoderDecoder
