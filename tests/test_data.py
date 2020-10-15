"""
Test `SMILESDataset` and `SMILESBatchColumnSampler` classes and their
components.
"""

import tempfile
import unittest
from typing import Iterator

from moleculegen import Token
from moleculegen.data import (
    SMILESBatchColumnSampler,
    SMILESConsecutiveSampler,
    SMILESDataset,
    SMILESTargetDataset,
    SMILESVocabulary,
)
from .utils import TempSMILESFile


class SMILESDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile(tempfile_kwargs={'prefix': 'dataset'})
        self.fh = self.temp_file.open()

        self.item_list = self.temp_file.smiles_strings.split('\n')

        self.dataset = SMILESDataset(self.fh.name)

    def test_1_read(self):

        self.assertTrue(
            all(
                s.startswith(Token.BOS) and s.endswith(Token.EOS)
                for s in self.dataset
            )
        )
        self.assertListEqual(
            self.item_list,
            [Token.crop(s) for s in self.dataset],
        )

        self.assertEqual(
            len(self.item_list),
            len(self.dataset),
        )

    def test_2_map_(self):
        mapped_data_iter = self.dataset.map_(Token.crop)

        self.assertIsInstance(mapped_data_iter, Iterator)
        self.assertListEqual(self.item_list, list(mapped_data_iter))

    def test_3_filter_(self):
        filtered_data_iter = self.dataset.filter_(lambda s: len(s) < 10)

        self.assertIsInstance(filtered_data_iter, Iterator)
        self.assertListEqual(['{N#N}', '{CN=C=O}'], list(filtered_data_iter))

    def tearDown(self):
        self.fh.close()


class SMILESTargetDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.data = (
            'SMILES,Activity\n'
            'CC[C@H](O1)CC[C@@]12CCCO2,1\n'
            'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2,0\n'
            'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N,0\n'
            'CC(=O)NCCC1=CNc2c1cc(OC)cc2,1\n'
            'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1,0\n'
        )
        self.only_smiles = [line[:-2] for line in self.data.split('\n')[1:-1]]

        self.file_handler = tempfile.NamedTemporaryFile(mode='w+', encoding='ascii')
        self.file_handler.write(self.data)
        self.file_handler.seek(0)

        self.dataset = SMILESTargetDataset(
            filename=self.file_handler.name,
            target_column='Activity',
            smiles_column='SMILES',
            augment=True,
            generate_only_active=True,
        )

    def test_1_len(self):
        self.assertEqual(5, len(self.dataset))

    def test_2_getitem(self):

        for i in range(len(self.only_smiles)):
            self.assertEqual(
                self.only_smiles[i],
                Token.crop(self.dataset[i]),  # `augment` was set to `True` during init.
            )

    def test_3_iter(self):
        only_active = ['CC[C@H](O1)CC[C@@]12CCCO2', 'CC(=O)NCCC1=CNc2c1cc(OC)cc2']

        self.assertIsInstance(self.dataset, Iterator)
        self.assertListEqual(only_active, list(map(Token.crop, self.dataset)))

        self.dataset.generate_only_active = False

        self.assertListEqual(self.only_smiles, list(map(Token.crop, self.dataset)))

    def test_4_smiles_data(self):
        self.assertSequenceEqual(
            self.only_smiles,
            self.dataset.get_smiles_data(crop=True).tolist(),
        )

    def test_5_target_data(self):
        self.assertSequenceEqual(
            [True, False, False, True, False],
            self.dataset.get_target_data().tolist(),
        )

    def tearDown(self):
        self.file_handler.close()


class SMILESBatchColumnSamplerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile(
            tempfile_kwargs={'prefix': 'dataloader'})
        self.fh = self.temp_file.open()

        dataset = SMILESDataset(self.fh.name)
        vocabulary = SMILESVocabulary(dataset=dataset, need_corpus=True)
        self.dataloader = SMILESBatchColumnSampler(
            corpus=vocabulary.corpus,
            batch_size=2,
            n_steps=4,
            shuffle=True,
        )

    def test_iter(self):
        sample_size = (self.dataloader.batch_size, self.dataloader.n_steps)

        for batch in self.dataloader:
            self.assertEqual(batch.inputs.shape, sample_size)
            self.assertEqual(batch.outputs.shape, sample_size)
            self.assertEqual(
                batch.valid_lengths.size, self.dataloader.batch_size)

    def tearDown(self):
        self.fh.close()


class SMILESConsecutiveSamplerTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles_string = 'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1'
        with TempSMILESFile(smiles_strings=self.smiles_string) as temp_fh:
            dataset = SMILESDataset(temp_fh.file_handler.name)
        self.vocabulary = SMILESVocabulary(dataset, need_corpus=True)

    def test_sampling_with_padding(self):
        smiles_string = Token.tokenize(Token.augment(self.smiles_string))
        n_steps = 20
        sampler = SMILESConsecutiveSampler(self.vocabulary.corpus, n_steps=n_steps)

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
        sampler = SMILESConsecutiveSampler(self.vocabulary.corpus, n_steps=n_steps)

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


class SMILESVocabularyTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_file = TempSMILESFile()
        self.fh = self.temp_file.open()

        # See `test_data.py` for data set test cases.
        self.dataset = SMILESDataset(self.fh.name)
        self.vocab = SMILESVocabulary(self.dataset, need_corpus=True)

    def test_tokens_and_idx(self):
        self.assertSequenceEqual(
            # Tokenize the entire dataset to get a set of unique tokens.
            sorted(
                set(
                    Token.tokenize(
                        self.temp_file.smiles_strings.replace('\n', '')
                    )
                )
            ),
            # The temporary file is not augmented by the special tokens.
            sorted(set(self.vocab.token_to_idx) - Token.SPECIAL),
        )
        self.assertSequenceEqual(
            sorted(
                set(self.vocab.token_to_idx)
                # Pad and unknown tokens does not appear in the original set.
                - {Token.PAD, Token.UNK}
            ),
            sorted(set(self.vocab.token_freqs)),
        )

    def test_corpus(self):
        # Original SMILES list without padded special tokens.
        smiles_list = self.temp_file.smiles_strings.split('\n')

        self.assertEqual(len(self.vocab.corpus), len(smiles_list))

        for idx, tokens in zip(self.vocab.corpus, smiles_list):
            # Add special tokens in order to correspond to the loaded corpus
            # for data sampling and model fitting.
            tokens = Token.augment(tokens)
            # Test id-to-token mapping.
            self.assertEqual(
                ''.join(self.vocab.get_tokens(idx)),
                tokens,
            )
            # Test token-to-id mapping.
            self.assertListEqual(idx, self.vocab[Token.tokenize(tokens)])

    def test_contains(self):
        self.assertNotIn(Token.UNK, self.vocab)

        all_tokens = Token.get_all_tokens()

        for token in self.vocab:
            if len(token) == 1 and token.islower():
                token = token.upper()
            self.assertIn(token, all_tokens)

    def tearDown(self):
        self.fh.close()


if __name__ == '__main__':
    unittest.main()
