"""
Test the Token enumeration class methods.
"""

import unittest

from moleculegen import Token


class TokenTestCase(unittest.TestCase):
    def setUp(self):
        self.smiles = 'CN=C=O'
        self.n_pad = 4
        self.aug_no_pad = Token.BOS.token + self.smiles + Token.EOS.token
        self.aug_w_pad = (
            Token.BOS.token
            + self.smiles
            + Token.PAD.token * self.n_pad
            + Token.EOS.token
        )

    def test_augment(self):
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(self.smiles),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment((Token.BOS.token*5) + self.aug_no_pad),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(self.aug_no_pad + (Token.EOS.token*3)),
        )
        self.assertEqual(
            self.aug_no_pad,
            Token.augment(
                (Token.BOS.token*4) + self.aug_no_pad + (Token.EOS.token*4))
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
            Token.crop((Token.BOS.token*5) + self.aug_no_pad),
        )
        self.assertEqual(
            self.smiles,
            Token.crop(self.aug_no_pad + (Token.EOS.token*3)),
        )
        self.assertEqual(
            self.smiles,
            Token.crop(
                (Token.BOS.token*4) + self.aug_no_pad + (Token.EOS.token*4))
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

    def test_rule_class_set(self):
        self.assertEqual(
            frozenset('{}*_'),
            Token.get_rule_class_members(rule_class='special')
        )


if __name__ == '__main__':
    unittest.main()
