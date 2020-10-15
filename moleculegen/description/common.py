"""
Common feature transformers.

Classes
-------
OneHotEncoder
    One-hot encoder functor.
MoleculeTransformer
    Convert SMILES compounds into RDKit molecules.
RDKitDescriptorTransformer
    Calculate RDKit descriptors.
"""

__all__ = (
    'OneHotEncoder',
    'MoleculeTransformer',
    'RDKitDescriptorTransformer',
)


from typing import Iterable, List

import mxnet as mx
from rdkit.Chem import Descriptors, Mol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from .base import check_compounds_valid, get_descriptor_df_from_mol


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
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1

        return self

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
        check_array(
            compounds,
            accept_large_sparse=False,
            dtype='object',
            ensure_2d=False,
        )

        return check_compounds_valid(
            compounds, raise_exception=True, **self.converter_kwargs)


class RDKitDescriptorTransformer(BaseEstimator, TransformerMixin):
    """Calculate RDKit descriptors.

    Examples
    --------
    >>> import moleculegen as mg
    >>> molecule_tr = mg.description.MoleculeTransformer()
    >>> rdkit_descriptor_tr = mg.description.RDKitDescriptorTransformer()
    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(molecule_tr, rdkit_descriptor_tr)
    """

    # noinspection PyUnusedLocal
    def fit(self, compounds: Iterable[str], y_ignored=None):
        # noinspection PyAttributeOutsideInit
        # noinspection PyProtectedMember
        self.descriptor_names_ = tuple(name for name, func in Descriptors._descList)
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1

        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, molecules: List[Mol]):
        """Return a numpy array of calculated RDKit descriptors.

        Parameters
        ----------
        molecules : list of rdkit.Chem.Mol
            A list of RDKit molecules.

        Returns
        -------
        descriptors : pandas.DataFrame,
                shape = (`len(molecules)`, `len(self.descriptor_names_)`)
        """
        # noinspection PyProtectedMember
        descriptor_df = get_descriptor_df_from_mol(
            molecules=molecules,
            function_map=dict(Descriptors._descList),
        )

        return descriptor_df
