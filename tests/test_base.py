"""
Test the Token enumeration class methods.
"""

import unittest

from moleculegen import Token


class TokenTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles = 'CN=C=O'
        self.n_pad = 4
        self.aug_no_pad = Token.BOS + self.smiles + Token.EOS
        self.aug_w_pad = (
            Token.BOS
            + self.smiles
            + Token.EOS
            + Token.PAD * self.n_pad
        )

    def test_augment(self):
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(self.smiles),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment((Token.BOS*5) + self.aug_no_pad),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(self.aug_no_pad + (Token.EOS*3)),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(
                (Token.BOS*4) + self.aug_no_pad + (Token.EOS*4))
        )

    def test_augment_without_padding(self):
        self.assertEqual(
            self.aug_w_pad,
            Token.augment(self.smiles, padding_len=self.n_pad),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(self.smiles, padding_len=0),
        )
        self.assertEqual(
            self.aug_w_pad,
            Token.augment(self.aug_w_pad),
        )

    def test_crop(self):
        self.assertEqual(
            self.smiles,
            Token.crop(self.aug_no_pad),
        )
        self.assertEqual(
            self.smiles,
            Token.crop((Token.BOS*5) + self.aug_no_pad),
        )
        self.assertEqual(
            self.smiles,
            Token.crop(self.aug_no_pad + (Token.EOS*3)),
        )
        self.assertEqual(
            self.smiles,
            Token.crop(
                (Token.BOS*4) + self.aug_no_pad + (Token.EOS*4))
        )

    def test_crop_without_padding(self):
        self.assertEqual(
            self.smiles,
            Token.crop(self.aug_w_pad, padding=True),
        )
        self.assertEqual(
            self.smiles,
            Token.crop(self.smiles),
        )

    def test_tokenize(self):

        def test(smiles, find_brackets=False):
            self.assertListEqual(
                Token.tokenize(''.join(smiles), find_brackets=find_brackets),
                smiles
            )

        # Only single-char tokens.
        test(['C', 'n', '(', '=', 'N', ')'])

        # Single-char and double-char tokens.
        test(['O', '@', 'N', '2', 'Os', 'i', '[', 'Ni', ']'])

        # Single-char and double-char tokens w/ two consecutive tokens
        # composing an atom (they should be divided into two separate tokens).
        test(['C', 'N', 'C', 'n', 'o', 'Cl', 'C', 'Cl', 'O'])

        # Subcompound tokens.
        test(['N', '@@', '(', 'C', ')', 'C', '(', '=', 'O', ')', 'O', '10'])

        # Includes subcompounds in square brackets.
        test(['c', '1', 'c', 'n', 'c', '[nH]', 'c', '(', '=', 'O', ')', '1'], True)
        test(['c', '1', 'c', 'n', 'c', '[', 'n', 'H', ']', 'c', '(', '=', 'O', ')', '1'])


if __name__ == '__main__':
    unittest.main()
