"""
Fingerprint transformers.

Classes
-------
MorganFingerprint
    Apply the Morgan algorithm to a set of compounds to get circular
    fingerprints.
"""

__all__ = (
    'MorganFingerprint',
)


from typing import MutableSequence, Union

import numpy as np
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_scalar
from rdkit.Chem import MolFromSmiles
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
    return_sparse : bool, default True
        Whether to return csr-sparse matrix or numpy array.
    """

    def __init__(
            self,
            radius: int = 4,
            n_bits: int = 2048,
            return_sparse: bool = True,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.return_sparse = return_sparse

    # noinspection PyUnusedLocal
    def fit(
            self,
            smiles_strings: MutableSequence[str],
            y_ignored=None,
    ):
        """Check the instance parameters and return the instance.

        Parameters
        ----------
        smiles_strings : sequence of str
            SMILES strings.
        y_ignored : None
            This formal parameter will be ignored.
        """
        check_scalar(self.radius, 'radius', int, min_val=1)
        check_scalar(self.n_bits, 'number of bits', int, min_val=1)

        return self

    def transform(
            self,
            smiles_strings: MutableSequence[str],
    ) -> Union[np.array, sparse.csr_matrix]:
        """Return circular fingerprints as bit vectors.

        Parameters
        ----------
        smiles_strings : sequence of str
            SMILES strings.

        Returns
        -------
        fingerprints : np.array or scipy.sparse.csr_matrix
            ECFP.
        """
        check_array(
            smiles_strings,
            accept_large_sparse=False,
            dtype='object',
            ensure_2d=False,
        )

        molecules = [MolFromSmiles(smiles) for smiles in smiles_strings]
        molecules = filter(lambda m: m is not None, molecules)
        fingerprints = [
            GetMorganFingerprintAsBitVect(molecule, self.radius, self.n_bits)
            for molecule in molecules
        ]

        if self.return_sparse:
            matrix_format = sparse.csr_matrix
        else:
            matrix_format = np.array

        return matrix_format(fingerprints, dtype=np.uint8)
