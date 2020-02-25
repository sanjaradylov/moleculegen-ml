"""
Test `SMILESDataset` and `SMILESDataLoader` classes and their components.
"""

import unittest

from moleculegen import SMILESDataset, SMILESDataLoader
from moleculegen.tests.utils import TempSMILESFile


class DataTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'dataset'})
        self.fh = self.temp_file.open()

    def test_read(self):
        dataset = SMILESDataset(self.fh.name)
        self.assertEqual(
            len(self.temp_file.smiles_strings.split('\n')),
            len([sample for sample in dataset]),
            len(dataset),
        )

    def tearDown(self):
        self.fh.close()


class DataLoaderTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile(
            tempfile_kwargs={'prefix': 'dataloader'})
        self.fh = self.temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        self.dataloader = SMILESDataLoader(2, 4, dataset)

    def test_iter(self):
        sample_size = (self.dataloader.batch_size, self.dataloader.n_steps)

        for x, y, valid_lengths in self.dataloader:
            self.assertEqual(x.shape, sample_size)
            self.assertEqual(y.shape, sample_size)
            self.assertEqual(x.shape[0], valid_lengths.shape[0])

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
