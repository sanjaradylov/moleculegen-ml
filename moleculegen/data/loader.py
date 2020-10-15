"""
Utilities to load SMILES data sets.

Classes
-------
SMILESDataset
    Load text data set containing SMILES strings.
SMILESTargetDataset
    SMILES-Activity dataset.
"""

__all__ = (
    'SMILESDataset',
    'SMILESTargetDataset',
)


from typing import Any, AnyStr, Callable, Iterator, List, Optional, TextIO, Union

import mxnet as mx
import numpy as np
import pandas as pd

from ..base import Token


class SMILESDataset(mx.gluon.data.SimpleDataset):
    """Text data set comprising SMILES strings. Every line is presented as
    a SMILES string. No headlines expected.

    Supports Gluon's SimpleDataset API, but also implements `map_`, `filter_`, and
    `take_` methods, all of which return an iterator, not Gluon's private datasets.
    Implemented to save memory.

    Parameters
    ----------
    filename : str
        Path to the text file.
    encoding : {'ascii', 'utf8'}, default 'ascii'
        File encoding format.
    augment : bool, default True
        Whether to prepend `moleculegen.Token.BOS` and append `moleculegen.Token.EOS` to
        SMILES strings.
    """

    def __init__(
            self,
            filename: AnyStr,
            encoding: str = 'ascii',
            augment: bool = True,
    ):
        self.__filename = filename
        self.__encoding = encoding
        self.__augment = augment

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

    def map_(self, function: Callable[[str], Any]) -> Iterator[Any]:
        """Apply `function` to every entry of the data and yield the result.

        Parameters
        ----------
        function : callable, str -> any

        Yields
        ------
        result : any
        """
        return (function(smiles) for smiles in self._data)

    def filter_(self, function: Callable[[str], bool]) -> Iterator[str]:
        """Filter out the entries with `function(entry) == False`.

        Parameters
        ----------
        function : callable, str -> bool

        Yields
        ------
        smiles : str
        """
        return (smiles for smiles in self._data if function(smiles))

    def take_(
            self,
            count: int,
            replace: bool = False,
            random_state: Optional[int] = None,
    ) -> Iterator[str]:
        """Sample `count` entries, with replacement if `replace=True`.

        Parameters
        ----------
        count : int
            The number of entries to sample.
        replace : bool, default False
            Whether to sample with replacement.
        random_state : int, default None
            Random seed.

        Yields
        ------
        smiles : str
        """
        rng = np.random.RandomState(seed=random_state)
        for index in rng.choice(len(self), size=count, replace=replace):
            yield self._data[index]

    def _read(self) -> List[str]:
        """Read SMILES strings from the specified filename.

        Returns
        -------
        smiles_strings : list
            Non-empty SMILES strings.
        """
        smiles_strings: List[str] = []

        if self.__augment:
            def smiles_fn(s: str) -> str:
                return Token.augment(s.strip())
        else:
            def smiles_fn(s: str) -> str:
                return s.strip()

        with open(self.__filename, encoding=self.__encoding) as fh:
            for line in fh:
                if not line:
                    continue

                smiles = smiles_fn(line)
                smiles_strings.append(smiles)

        return smiles_strings


class SMILESTargetDataset(mx.gluon.data.Dataset):
    """SMILES-Activity dataset. Loads data from the csv-file named `filename` and stores
    (augmented) compounds and activities separately in pandas Series. An iterator itself,
    iterates over active compounds if `generate_only_active=True`; otherwise, generates
    every compound. Activities are expected to be booleans or integers (0 -- inactive,
    1 -- active).

    To optionally crop the series of loaded compounds and return them, use
    `get_smiles_data`. To return the series of activities, use `get_target_data`.

    Parameters
    ----------
    filename : str
        The path to a csv-file.
    target_column : str
        The name of the column in the file indicating activities.
    smiles_column : str, default 'SMILES'
        The name of the column in the file indicating SMILES strings.
    augment : bool, default True
        Whether ot augment the SMILES strings.
    generate_only_active : bool, default True
        During iteration, whether to generate only active compounds towards
        `target_column`.
    """

    def __init__(
            self,
            filename: Union[AnyStr, TextIO],
            target_column: str,
            smiles_column: str = 'SMILES',
            augment: bool = True,
            generate_only_active: bool = True,
    ):
        self._smiles_data = pd.read_csv(filename, usecols=[smiles_column], squeeze=True)
        self._augmented = augment
        if augment:
            self._smiles_data = self._smiles_data.apply(Token.augment)

        self._target_data = pd.read_csv(filename, usecols=[target_column], squeeze=True)
        self._target_data = self._target_data.astype(np.bool)

        self.generate_only_active = generate_only_active
        self.__current_index = 0

    def __len__(self) -> int:
        """Return the total number of compounds.
        """
        return self._smiles_data.shape[0]

    def __getitem__(self, index: int) -> str:
        """Return the SMILES string by index `index`.
        """
        return self._smiles_data[index]

    def __iter__(self):
        return self

    def __next__(self) -> str:
        try:
            smiles = self._smiles_data[self.__current_index]
            smiles_is_active = self._target_data[self.__current_index]
        except (IndexError, KeyError):
            self.__current_index = 0
            raise StopIteration
        else:
            self.__current_index += 1

        if (
                not self.generate_only_active
                or self.generate_only_active and smiles_is_active
        ):
            return smiles
        else:
            return self.__next__()

    def get_smiles_data(self, crop: bool = True) -> pd.Series:
        """If `crop == True`, crop every SMILES string in the series and return them.
        Otherwise, return the previously loaded series.

        Parameters
        ----------
        crop : bool, default True
            Whether to use `moleculegen.Token.crop` to remove BOS and EOS tokens.

        Returns
        -------
        pandas.Series of str
        """
        if self._augmented and crop:
            return self._smiles_data.apply(Token.crop)
        return self._smiles_data

    def get_target_data(self) -> pd.Series:
        """Return the series of activities.

        Returns
        -------
        pandas.Series of bool
        """
        return self._target_data
