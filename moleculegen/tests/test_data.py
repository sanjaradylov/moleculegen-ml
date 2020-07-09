"""
Test `SMILESDataset` and `SMILESDataLoader` classes and their components.
"""

import unittest

from moleculegen import (
    SMILESConsecutiveSampler,
    SMILESDataset,
    SMILESDataLoader,
    Token,
    Vocabulary,
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


class SMILESConsecutiveSamplerTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles_string = 'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1'
        with TempSMILESFile(smiles_strings=self.smiles_string) as temp_fh:
            dataset = SMILESDataset(temp_fh.file_handler.name)
        self.vocabulary = Vocabulary(dataset, need_corpus=True)

    def test_sampling_with_padding(self):
        smiles_string = Token.tokenize(Token.augment(self.smiles_string))
        n_steps = 20
        sampler = SMILESConsecutiveSampler(self.vocabulary, n_steps=n_steps)

        step_i = 0
        for sample in sampler:
            input_s = ''.join(smiles_string[step_i:step_i+n_steps])
            output_s = ''.join(smiles_string[step_i+1:step_i+n_steps+1])

            if sample.valid_length < n_steps:
                input_s += Token.PAD * (n_steps-sample.valid_length-1)
                output_s += Token.PAD * (n_steps-sample.valid_length)

            self.assertEqual(
                input_s,
                ''.join(self.vocabulary.get_tokens(sample.inputs)),
            )
            self.assertEqual(
                output_s,
                ''.join(self.vocabulary.get_tokens(sample.outputs)),
            )

            step_i += n_steps

    def test_sampling_without_padding(self):
        tokens = Token.tokenize(Token.augment(self.smiles_string))
        n_steps = len(tokens) - 1
        sampler = SMILESConsecutiveSampler(self.vocabulary, n_steps=n_steps)

        step_i = 0
        for n_samples, sample in enumerate(sampler, start=1):
            self.assertListEqual(
                tokens[step_i:step_i+n_steps],
                self.vocabulary.get_tokens(sample.inputs),
            )
            self.assertListEqual(
                tokens[step_i+1:step_i+n_steps+1],
                self.vocabulary.get_tokens(sample.outputs),
            )
            self.assertEqual(n_steps, sample.valid_length)

            step_i += n_steps

        self.assertEqual(n_samples, 1)


if __name__ == '__main__':
    unittest.main()
