"""
Fingerprint transformers.

Classes
-------
MorganFingerprint
    Apply the Morgan algorithm to a set of molecules to get circular fingerprints.
"""

__all__ = (
    'MorganFingerprint',
)


from typing import Iterable, Union

import numpy as np
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_scalar
from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


class MorganFingerprint(BaseEstimator, TransformerMixin):
    """Apply the Morgan algorithm to a set of compounds to get circular
    fingerprints.

    Parameters
    ----------
    radius : int, default 4
        The radius of fingerprint.
    n_bits : int, default 2048
        The number of bits.
    return_sparse : bool, default False
        Whether to return csr-sparse matrix or numpy array.

    Examples
    --------
    >>> import moleculegen as mg
    >>> molecule_tr = mg.description.MoleculeTransformer()
    >>> ecfp_tr = mg.description.MorganFingerprint(n_bits=1024)
    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(molecule_tr, ecfp_tr)
    """

    def __init__(
            self,
            *,
            radius: int = 4,
            n_bits: int = 2048,
            return_sparse: bool = False,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.return_sparse = return_sparse

    # noinspection PyUnusedLocal
    def fit(
            self,
            molecules: Iterable[Mol],
            y_ignored=None,
    ):
        """Check the instance parameters and return the instance.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.
        y_ignored : None
            This formal parameter will be ignored.
        """
        check_scalar(self.radius, 'radius', int, min_val=1)
        check_scalar(self.n_bits, 'number of bits', int, min_val=1)
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1

        return self

    def transform(self, molecules: Iterable[Mol]) -> Union[np.array, sparse.csr_matrix]:
        """Return circular fingerprints as bit vectors.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.

        Returns
        -------
        fingerprints : np.array or scipy.sparse.csr_matrix
            ECFP.
        """
        fingerprints = [
            GetMorganFingerprintAsBitVect(molecule, self.radius, self.n_bits)
            for molecule in molecules
        ]

        if self.return_sparse:
            matrix_format = sparse.csr_matrix
        else:
            matrix_format = np.array

        return matrix_format(fingerprints, dtype=np.uint8)
