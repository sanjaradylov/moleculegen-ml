"""
Test sklearn-compatible transformers.
"""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline

from moleculegen.description import (
    check_compounds_valid,
    InternalTanimoto,
    InvalidSMILESError,
    MoleculeTransformer,
    MorganFingerprint,
    RDKitDescriptorTransformer,
)
from .utils import SMILES_STRINGS


class MorganFingerprintTestCase(unittest.TestCase):
    def setUp(self):
        smiles_strings = SMILES_STRINGS.split('\n')
        self.molecules = check_compounds_valid(smiles_strings, invalid='raise')

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
        mp = MorganFingerprint(return_type='csr_sparse')
        ecfp = mp.fit_transform(self.molecules)
        self.assertIsInstance(ecfp, csr_matrix)

        mp.return_type = 'ndarray'
        ecfp = mp.transform(self.molecules)
        self.assertIsInstance(ecfp, np.ndarray)


class MoleculeTransformerTestCase(unittest.TestCase):
    def setUp(self):
        self.transformer = MoleculeTransformer(invalid='raise')

    def test_1_invalid(self):
        compounds = SMILES_STRINGS.split('\n')
        compounds.extend(['invalid', '#C%'])

        with self.assertRaises(InvalidSMILESError):
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


class InternalTanimotoTestCase(unittest.TestCase):
    def test_1_sim_matrix(self):
        molecules = ['N#N', 'CN=C=O', 'C#N', 'C#C']
        mt = MoleculeTransformer()
        it = InternalTanimoto(radius=2, n_bits=4096)
        pipe = make_pipeline(mt, it)
        sim_matrix = pipe.fit_transform(molecules)

        self.assertTrue(np.allclose(sim_matrix, sim_matrix.T))


if __name__ == '__main__':
    unittest.main()
