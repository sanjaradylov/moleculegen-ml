"""
Fingerprint transformers.

Classes
-------
InternalTanimoto
    Build internal Tanimoto similarity matrix.
MorganFingerprint
    Apply the Morgan algorithm to a set of molecules to get circular fingerprints.
"""

__all__ = (
    'InternalTanimoto',
    'MorganFingerprint',
)


from typing import Iterable, Union

import numpy as np
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_scalar
from rdkit.Chem import Mol
from rdkit.DataStructs import BulkTanimotoSimilarity
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
    return_type : {'ndarray', 'csr_sparse', 'bitvect_list'}, default 'ndarray'
        Whether to return csr-sparse matrix, numpy array, or list of rdkit bit vectors.

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
            return_type: str = 'ndarray',
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.return_type = return_type

    # noinspection PyUnusedLocal
    def fit(self, molecules: Iterable[Mol], y_ignored=None) -> 'MorganFingerprint':
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

        valid_return_types = {'ndarray', 'csr_sparse', 'bitvect_list'}
        if self.return_type not in valid_return_types:
            raise ValueError(
                f'`return_type` must be in {valid_return_types}, '
                f'not {self.return_type!r}'
            )

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

        if self.return_type == 'ndarray':
            return np.array(fingerprints, dtype=np.uint8)
        elif self.return_type == 'csr_sparse':
            return sparse.csr_matrix(fingerprints, dtype=np.uint8)
        elif self.return_type == 'bitvect_list':
            return fingerprints


class InternalTanimoto(BaseEstimator):
    """Build internal Tanimoto similarity matrix. Compare ECFPs of molecules.

    Parameters
    ----------
    radius : int, default=2
        The radius of fingerprint.
    n_bits : int, default=4096
        The number of bits.
    dtype : str or numpy dtype, default=numpy.float16
    """

    def __init__(
            self,
            radius: int = 2,
            n_bits: int = 4096,
            dtype=np.float16,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.dtype = dtype

    # noinspection PyUnusedLocal
    def fit(self, molecules: Iterable[Mol], y_ignored=None) -> 'InternalTanimoto':
        self.fit_transform(molecules)
        return self

    # noinspection PyUnusedLocal
    def fit_transform(self, molecules: Iterable[Mol], y_ignored=None) -> np.array:
        """Return Tanimoto similarity matrix.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.
        y_ignored : None
            This formal parameter will be ignored.

        Returns
        -------
        numpy.ndarray, shape = (len(molecules), len(molecules))
        """
        ecfp = MorganFingerprint(
            radius=self.radius, n_bits=self.n_bits, return_type='bitvect_list')
        fingerprints = ecfp.fit_transform(molecules)

        # noinspection PyAttributeOutsideInit
        self.ecfp_ = ecfp
        # noinspection PyAttributeOutsideInit
        self.n_features_in_ = 1

        n_fingerprints = len(fingerprints)
        sim_matrix = np.ones((n_fingerprints, n_fingerprints), dtype=self.dtype)
        for i in range(1, n_fingerprints):
            sim_index = BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
            sim_matrix[i, :i] = sim_index
            sim_matrix[:i, i] = sim_index

        return sim_matrix
