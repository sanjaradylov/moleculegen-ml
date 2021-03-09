"""
Common feature preprocessing methods.

Classes:
    OneHotEncoder: One-hot encoder functor.
    MoleculeTransformer: Convert SMILES compounds into RDKit molecules.
    RDKitDescriptorTransformer: Calculate RDKit descriptors.
"""

__all__ = (
    'get_descriptor_df_from_mol',
    'OneHotEncoder',
    'MoleculeTransformer',
    'RDKitDescriptorTransformer',
)


from typing import Collection, Iterable, List

import mxnet as mx
from rdkit.Chem import Descriptors, Mol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from .base import check_compounds_valid, get_descriptor_df_from_mol


class OneHotEncoder(mx.gluon.HybridBlock):
    """One-hot encoder class. It is implemented as a functor for more
    convenience, to pass it as a detached embedding layer.

    Parameters
    ----------
    depth : int
        The depth of one-hot encoding.
    """

    def __init__(self, depth: int, **kwargs):
        super().__init__(prefix=kwargs.get('prefix'), params=None)

        self.depth = depth

    def hybrid_forward(self, module, indices: mx.np.ndarray, *args, **kwargs) \
            -> mx.np.ndarray:
        """Return one-hot encoded tensor.

        Parameters
        ----------
        module : mxnet.symbol or mxnet.nd
        indices : nx.np.ndarray or mx.sym.Symbol
            The indices (categories) to encode.
        *args, **kwargs
            Additional arguments for `mxnet.nd.one_hot`.
        """
        # FIXME `one_hot` doesn't support numpy arrays...
        return module.one_hot(
            indices.as_nd_ndarray(), self.depth, *args, **kwargs).as_np_ndarray()


class MoleculeTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES compounds into RDKit molecules.

    Parameters
    ----------
    invalid : {'nan', 'raise', 'skip'}, default='skip'
        Whether to a) replace invalid SMILES with `numpy.NaN`s, b) raise
        `InvalidSMILESError`, or c) ignore them.
    converter_kwargs : dict, default={}
        Optional key-word arguments for `rdkit.Chem.MolFromSmiles`.

    Examples
    --------
    >>> from rdkit.Chem import Mol
    >>> mt = MoleculeTransformer(invalid='skip')
    >>> smiles = ('CCO', 'C#N', 'N#N', 'invalid_smiles')
    >>> molecules = mt.fit_transform(smiles)
    >>> len(molecules)
    3
    """

    def __init__(self, invalid: str = 'skip', **converter_kwargs):
        self.invalid = invalid
        self.converter_kwargs = converter_kwargs

    # noinspection PyUnusedLocal
    def fit(self, compounds: Collection[str], unused_y=None) -> 'MoleculeTransformer':
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1
        return self

    def transform(self, compounds: Collection[str]) -> List[Mol]:
        """Convert SMILES compounds into RDKit molecules.

        Parameters
        ----------
        compounds : collection of str
            SMILES strings.

        Returns
        -------
        molecules : list of rdkit.Chem.Mol

        Raises
        ------
        InvalidSMILESError
            If `self.invalid='raise'` and at least one compound has invalid SMILES.
        """
        check_array(
            compounds, accept_large_sparse=False, dtype='object', ensure_2d=False)
        return check_compounds_valid(
            compounds, invalid=self.invalid, **self.converter_kwargs)


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
    def fit(self, compounds: Iterable[Mol], unused_y=None):
        # noinspection PyAttributeOutsideInit
        # noinspection PyProtectedMember
        self.descriptor_names_ = tuple(name for name, func in Descriptors._descList)
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1

        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, molecules: Iterable[Mol]):
        """Return calculated RDKit descriptors.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.

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
