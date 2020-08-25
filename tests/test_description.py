"""
Test fingerprint transformers.
"""

import unittest

from numpy import ndarray
from scipy.sparse import csr_matrix

from moleculegen.description import MorganFingerprint
from .utils import SMILES_STRINGS


class ECFPTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles_strings = SMILES_STRINGS.split('\n')

    def test_1_invalid_parameters(self):
        mp = MorganFingerprint(radius=0, n_bits=0)
        self.assertRaises(ValueError, mp.fit, self.smiles_strings)

        mp.radius = 2
        self.assertRaises(ValueError, mp.fit, self.smiles_strings)

        mp.n_bits = '1024'
        self.assertRaises(TypeError, mp.fit, self.smiles_strings)

        mp.radius = 2.0
        self.assertRaises(TypeError, mp.fit, self.smiles_strings)

        mp.radius = 2
        mp.n_bits = 2048
        mp.fit(SMILES_STRINGS)

    def test_2_transform(self):
        mp = MorganFingerprint(return_sparse=True)
        ecfp = mp.fit_transform(self.smiles_strings)
        self.assertIsInstance(ecfp, csr_matrix)

        mp.return_sparse = False
        ecfp = mp.transform(self.smiles_strings)
        self.assertIsInstance(ecfp, ndarray)


if __name__ == '__main__':
    unittest.main()
