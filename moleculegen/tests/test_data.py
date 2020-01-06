import os
import tempfile
import unittest

from moleculegen import SMILESDataset


class DataTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles_strings = (
            'CC(=O)NCCC1=CNc2c1cc(OC)cc2\n'
            'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
            'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
        )

        self.fh = tempfile.NamedTemporaryFile(mode='w+', encoding='ascii')
        self.fh.write(self.smiles_strings)
        self.fh.seek(os.SEEK_SET)

    def test_read(self):
        dataset = SMILESDataset(self.fh.name)
        self.assertEqual(
            len(self.smiles_strings.split('\n')),
            len([sample for sample in dataset]),
            len(dataset),
        )

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
