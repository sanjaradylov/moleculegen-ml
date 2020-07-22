"""
Utilities to load SMILES data sets.

Classes
-------
SMILESDataset
    Load text data set containing SMILES strings.
"""

__all__ = (
    'SMILESDataset',
)


import io
from typing import AnyStr, List

from mxnet.gluon.data import SimpleDataset

from ..base import Token


class SMILESDataset(SimpleDataset):
    """Text data set comprising SMILES strings. Every line is presented as
    a SMILES string.

    Parameters
    ----------
    filename : str
        Path to the text file.
    encoding : {'ascii', 'utf8'}, default 'ascii'
        File encoding format.
    """

    def __init__(
            self,
            filename: AnyStr,
            encoding: str = 'ascii',
    ):
        self.__filename = filename
        self.__encoding = encoding

        smiles_strings = self._read()
        super().__init__(smiles_strings)

    @property
    def filename(self):
        """The filename of the data set to read.

        Returns
        -------
        filename : str
        """
        return self.__filename

    @property
    def encoding(self):
        """The file encoding format.

        Returns
        -------
        encoding : {'ascii', 'utf8'}
        """
        return self.__encoding

    def _read(self) -> List[str]:
        """Read SMILES strings from the specified filename.

        Returns
        -------
        smiles_strings : list
            Non-empty SMILES strings.
        """
        smiles_strings: List[str] = []

        with io.open(self.__filename, encoding=self.__encoding) as fh:
            for line in fh:
                if not line:
                    continue

                # Add beginning-of-SMILES and end-of-SMILES tokens to prepare
                # the data for sampling and fitting.
                smiles = Token.augment(line.strip())
                smiles_strings.append(smiles)

        return smiles_strings
