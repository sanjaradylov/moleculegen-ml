"""
Test sklearn-compatible transformers.
"""

import unittest

from numpy import ndarray
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline

from moleculegen.description import (
    check_compounds_valid,
    MoleculeTransformer,
    MorganFingerprint,
    RDKitDescriptorTransformer,
)
from .utils import SMILES_STRINGS


class MorganFingerprintTestCase(unittest.TestCase):
    def setUp(self):
        smiles_strings = SMILES_STRINGS.split('\n')
        self.molecules = check_compounds_valid(smiles_strings, raise_exception=True)

    def test_1_invalid_parameters(self):
        mp = MorganFingerprint(radius=0, n_bits=0)
        self.assertRaises(ValueError, mp.fit, self.molecules)

        mp.radius = 2
        self.assertRaises(ValueError, mp.fit, self.molecules)

        mp.n_bits = '1024'
        self.assertRaises(TypeError, mp.fit, self.molecules)

        mp.radius = 2.0
        self.assertRaises(TypeError, mp.fit, self.molecules)

        mp.radius = 2
        mp.n_bits = 2048
        mp.fit(self.molecules)

    def test_2_transform(self):
        mp = MorganFingerprint(return_sparse=True)
        ecfp = mp.fit_transform(self.molecules)
        self.assertIsInstance(ecfp, csr_matrix)

        mp.return_sparse = False
        ecfp = mp.transform(self.molecules)
        self.assertIsInstance(ecfp, ndarray)


class MoleculeTransformerTestCase(unittest.TestCase):
    def setUp(self):
        self.transformer = MoleculeTransformer()

    def test_1_invalid(self):
        compounds = SMILES_STRINGS.split('\n')
        compounds.extend(['invalid', '#C%'])

        with self.assertRaises(TypeError):
            self.transformer.fit_transform(compounds)


class RDKitDescriptorTransformerTestCase(unittest.TestCase):
    def setUp(self):
        self.compounds = SMILES_STRINGS.split('\n')
        self.pipe = make_pipeline(MoleculeTransformer(), RDKitDescriptorTransformer())

    def test_1_descriptors(self):
        descriptors = self.pipe.fit_transform(self.compounds)

        self.assertTupleEqual(
            descriptors.shape,
            (len(self.compounds), len(self.pipe[1].descriptor_names_)),
        )


if __name__ == '__main__':
    unittest.main()
