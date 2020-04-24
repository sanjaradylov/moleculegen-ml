"""
Test `SMILESDataset` and `SMILESDataLoader` classes and their components.
"""

import unittest

from moleculegen import (
    SMILESDataset,
    SMILESDataLoader,
    Token,
)
from moleculegen.tests.utils import TempSMILESFile


class DataTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'dataset'})
        self.fh = self.temp_file.open()

        self.item_list = self.temp_file.smiles_strings.split('\n')

    def test_read(self):
        dataset = SMILESDataset(self.fh.name)

        self.assertTrue(
            all(
                s.startswith(Token.BOS) and s.endswith(Token.EOS)
                for s in dataset
            )
        )
        self.assertListEqual(
            self.item_list,
            [Token.crop(s) for s in dataset],
        )

        self.assertEqual(
            len(self.item_list),
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

        for batch in self.dataloader:
            self.assertEqual(batch.x.shape, sample_size)
            self.assertEqual(batch.y.shape, sample_size)
            self.assertEqual(batch.v_x.size, self.dataloader.batch_size)
            self.assertEqual(batch.v_y.size, self.dataloader.batch_size)

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
