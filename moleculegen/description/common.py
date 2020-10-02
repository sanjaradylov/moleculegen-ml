"""
Common feature transformers.

Classes
-------
OneHotEncoder
    One-hot encoder functor.
MoleculeTransformer
    Convert SMILES compounds into RDKit molecules.
"""

__all__ = (
    'OneHotEncoder',
    'MoleculeTransformer',
)

from typing import Iterable, List

import mxnet as mx
from rdkit.Chem import Mol
from sklearn.base import BaseEstimator, TransformerMixin

from .base import check_compounds_valid


class OneHotEncoder:
    """One-hot encoder class. It is implemented as a functor for more
    convenience, to pass it as a detached embedding layer.

    Parameters
    ----------
    depth : int
        The depth of one-hot encoding.
    """

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(self, indices: mx.np.ndarray, *args, **kwargs) -> mx.np.ndarray:
        """Return one-hot encoded tensor.

        Parameters
        ----------
        indices : nx.np.ndarray or mx.sym.Symbol
            The indices (categories) to encode.
        *args, **kwargs
            Additional arguments for `nd.one_hot`.
        """
        if isinstance(indices, mx.sym.Symbol):
            return mx.sym.one_hot(
                indices.as_nd_ndarray(), self.depth, *args, **kwargs).as_np_ndarray()
        # noinspection PyUnresolvedReferences
        return mx.npx.one_hot(indices, self.depth, *args, **kwargs)


class MoleculeTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES compounds into RDKit molecules.

    Parameters
    ----------
    converter_kwargs : dict, default {}
        Optional key-word arguments for `rdkit.Chem.MolFromSmiles`.
    """

    def __init__(self, **converter_kwargs):
        self.converter_kwargs = converter_kwargs

    # noinspection PyUnusedLocal
    def fit(self, compounds: Iterable[str], y_ignored=None) -> 'MoleculeTransformer':
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, compounds: Iterable[str]) -> List[Mol]:
        """Convert SMILES compounds into RDKit molecules. Raise TypeError if there is an
        invalid compound.

        Parameters
        ----------
        compounds : iterable of str
            SMILES compounds.

        Returns
        -------
        molecules : list of rdkit.Chem.Mol

        Raises
        ------
        TypeError
            If any compound is invalid.
        """
        return check_compounds_valid(
            compounds, raise_exception=True, **self.converter_kwargs)
