"""
Additional common utilites for testing purposes.
"""

import os
import tempfile
from typing import TextIO


SMILES_STRINGS = (
    'N#N\n'
    'CN=C=O\n'
    '[Cu+2].[O-]S(=O)(=O)[O-]\n'
    'CN1CCC[C@H]1c2cccnc2\n'
    'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5\n'
    'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@@H](O)1\n'
    'OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2\n'
    'CC[C@H](O1)CC[C@@]12CCCO2\n'
    'CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1'
)


class TempSMILESFile:
    """Create temporary file containing SMILES strings.

    Parameters
    ----------
    smiles_strings : str, default utils.SMILES_STRINGS
        SMILES strings joined by '\n'.
    tempfile_kwargs : dict, default None
        Positional arguments for tempfile functions.
    """
    def __init__(
            self,
            smiles_strings: str = SMILES_STRINGS,
            tempfile_kwargs: dict = None,
    ):
        self.__smiles_strings = smiles_strings
        self.__tempfile_kwargs = tempfile_kwargs or dict()
        self.__file_handler = None

    @property
    def smiles_strings(self):
        return self.__smiles_strings

    @property
    def tempfile_kwargs(self):
        return self.__tempfile_kwargs

    @property
    def file_handler(self):
        return self.__file_handler

    def __enter__(self):
        self.__file_handler = self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__file_handler is not None:
            self.__file_handler.close()

    def open(self) -> TextIO:
        """Create temporary file with SMILES strings.

        Returns
        -------
        file_handler : file-like
            File handler.
        """
        file_handler = tempfile.NamedTemporaryFile(
            mode='w+', encoding='ascii', **self.tempfile_kwargs)
        file_handler.write(self.smiles_strings)
        file_handler.seek(os.SEEK_SET)
        return file_handler
