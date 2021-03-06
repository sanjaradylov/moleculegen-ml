"""
Utilities to create vocabularies of SMILES documents.

Classes
-------
SMILESVocabulary
    Map SMILES characters into their numerical representation.

Functions
---------
count_tokens
    Create token counter.
"""

__all__ = (
    'count_tokens',
    'SMILESVocabulary',
)


import collections
import functools
import itertools
import pickle
from typing import Counter, Dict, FrozenSet, Generator, List, Optional, Sequence, Union

from mxnet.gluon.data import SimpleDataset

from ..base import Token
from .loader import SMILESDataset


class SMILESVocabulary:
    """A dictionary to map SMILES characters into numerical representation.
    Read SMILES strings from `dataset` and create a token counter,
    or work directly with `tokens` if specified. Eventually obtain unique
    SMILES tokens and their frequencies to create token-to-index and
    index-to-token variables.
    If you want the class to manually process your data, pass `dataset` or `tokens`.
    If you have a pickled file with all metadata (see the documentation below), pass the
    file path `load_from_pickle`.

    Parameters
    ----------
    dataset : SMILESDataset, default None
        SMILES data set.
    tokens : list of list of str, default None
        SMILES tokens.
    need_corpus : bool, default False
        If True, load `corpus`property of token IDs for every SMILES sequence.
        Pass non-empty `dataset` or `tokens`.
    min_count : int, default 1
        The minimum number of token occurrences. If the frequency of a token is less than
        `min_count`, then the SMILES string containing this token will not be saved in a
        corpus.
        The parameter will be ignored if `load_from_pickle` is passed.

        ??? Note that while ignoring tokens and corresponding SMILES strings, we keep the
        statistics of the whole data in `dataset`. We do not recalculate the numbers even
        when encountering redundant tokens.
    match_bracket_atoms : bool, default False
        During tokenization, whether to treat the subcompounds enclosed in []
        as separate tokens.

    load_from_pickle : str, default None
        If passed, load all the attributes from the file named `load_from_pickle`.

    The following metadata will be stored for every vocabulary instance. To save them in
    a pickled file, use the `to_pickle` method. To load them from the file to initialize
    a new vocabulary instance, pass the `load_from_pickle` parameter.

    Attributes
    ----------
    idx_to_token : list of str
        List of tokens.
    token_to_idx : dict, str -> int
        Token-index mapping.
    token_freqs : dict, str -> int
        The token-to-count mapping.
    corpus : list of list of int or None
        Original data set corpus. Accessible only if corpus is needed.

    Raises
    ------
    ValueError
        If all parameters `dataset`, `tokens`, and `load_from_pickle` are empty.
    """

    # Special token IDs.
    UNK_ID = -1
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(
            self,
            dataset: Optional[SimpleDataset] = None,
            tokens: Optional[List[List[str]]] = None,
            need_corpus: bool = False,
            min_count: int = 1,
            match_bracket_atoms: bool = False,
            load_from_pickle: Optional[str] = None,
    ):
        self._corpus: Optional[List[List[int]]] = None

        if load_from_pickle is not None:
            with open(load_from_pickle, 'rb') as fh:
                data_map = pickle.load(fh)
                self._token_freqs = data_map['token_freqs']
                self._idx_to_token = data_map['idx_to_token']
                self._token_to_idx = data_map['token_to_idx']
                self._corpus = data_map['corpus']

        else:
            if dataset is None and tokens is None:
                raise ValueError(
                    'At least one of the arguments `dataset` and `tokens` '
                    'must be passed if `load_from_pickle` (a path to the '
                    'pickled file with data) is empty.'
                )

            tokens: List[List[str]] = (
                tokens
                or [Token.tokenize(sample, match_bracket_atoms) for sample in dataset]
            )
            counter: Counter[str] = count_tokens(tokens)
            redundant_tokens: FrozenSet[str] = frozenset(
                token
                for token, count in counter.items()
                if count < min_count
            )
            # All numbers counts will remain the same (see docs).
            for token in redundant_tokens:
                counter.pop(token)
            has_redundant_tokens = functools.partial(
                has_tokens, token_set=redundant_tokens)

            self._token_freqs: Dict[str, int] = dict(sorted(
                counter.items(), key=lambda c: c[1], reverse=True))

            # 'Unknown' token is not used in training/prediction.
            self._idx_to_token: List[str] = [Token.PAD, Token.BOS, Token.EOS]
            self._token_to_idx: Dict[str, int] = {
                token: id_ for id_, token in enumerate(self._idx_to_token)
            }
            for token in set(counter.keys()) - set(self._idx_to_token):
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._token_to_idx)

            if need_corpus:
                self._corpus: List[List[int]] = [
                    self[line] for line in tokens if not has_redundant_tokens(line)
                ]

    def __repr__(self) -> str:
        tokens = ''
        max_length = 60
        current_length = 0

        for token in self._token_to_idx:
            token = f'{token!r}, '
            current_length += len(token)
            if current_length < max_length:
                tokens += token

        if current_length > max_length:
            tokens += '...'
        else:
            tokens = tokens.rstrip(', ')

        return f'{self.__class__.__name__}{{ {tokens} }}'

    def __len__(self) -> int:
        """Return the number of unique tokens.

        Returns
        -------
        n : int
        """
        return len(self._idx_to_token)

    def __contains__(self, token: str) -> bool:
        """Check if `token` is in the vocabulary.

        Parameters
        ----------
        token : str

        Returns
        -------
        check : bool
        """
        return token in self._token_freqs

    def __iter__(self) -> Generator[str, None, None]:
        """Generate the tokens from the vocabulary.

        Yields
        ------
        token : str
        """
        return (token for token in self._token_to_idx)

    def __getitem__(self, tokens: Union[str, Sequence[str]]) \
            -> Union[int, List[int]]:
        """Get the index/indices of the token/tokens.

        Parameters
        ----------
        tokens : str or list of str
            Token(s).

        Returns
        -------
        idx : int or list of int
            Token index/indices.

        Raises
        ------
        KeyError
            When `tokens` are of unsupported type.
        """
        if isinstance(tokens, str):
            return self._token_to_idx.get(tokens, self.UNK_ID)
        elif isinstance(tokens, (list, tuple)):
            return [
                self._token_to_idx.get(token, self.UNK_ID)
                for token in tokens
            ]
        else:
            raise KeyError(
                f"`tokens` must be of type str or list/tuple of str, "
                f"not {type(tokens)}."
            )

    def get_token_id_corpus(self, dataset: SMILESDataset) -> List[List[int]]:
        """Transform the sequence of SMILES strings `dataset` into a list of token ID
        lists.

        Parameters
        ----------
        dataset : SMILESDataset
            SMILES strings.

        Returns
        -------
        corpus : list of list of int
        """
        tokens: Generator[List[str], None, None] = (
            Token.tokenize(sample) for sample in dataset
        )
        corpus: List[List[int]] = [self.__getitem__(line) for line in tokens]

        return corpus

    @property
    def corpus(self) -> List[List[int]]:
        """The original data set corpus. Accessible only if corpus is needed.

        Returns
        -------
        corpus : list of list of int

        Raises
        ------
        AttributeError
            If initially `need_corpus` was set to False or `load_from_pickle`
            didn't contain a non-empty corpus.
        """
        if self._corpus is None:
            raise AttributeError(
                'a corpus is loaded during the initialization of an instance '
                'if `need_corpus=True` or if the pickled file '
                '`load_from_pickle`contains non-empty corpus data.'
            )
        return self._corpus

    @property
    def idx_to_token(self) -> List[str]:
        """The list of unique tokens.

        Returns
        -------
        idx_to_token : list of str
        """
        return self._idx_to_token

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """The token-to-id mapping.

        Returns
        -------
        token_to_idx : dict, str -> int
        """
        return self._token_to_idx

    @property
    def token_freqs(self) -> Dict[str, int]:
        """The token-to-count mapping.

        Returns
        -------
        token_freqs : dict, str -> int
        """
        return self._token_freqs

    def get_tokens(self, indices: List[int]) -> List[str]:
        """Return the tokens corresponding to `indices`.

        Returns
        -------
        tokens : list of str
        """
        return [self._idx_to_token[index] for index in indices]

    def to_pickle(self, filename: str):
        """Save all the attributes (`token_freqs`, `idx_to_token`, `token_to_idx`, and
        `corpus`) as a name-to-data dictionary to a binary file `filename`.

        Parameters
        ----------
        filename : str
        """
        with open(filename, 'wb') as fh:
            data_map = {
                'token_freqs': self._token_freqs,
                'idx_to_token': self._idx_to_token,
                'token_to_idx': self._token_to_idx,
                'corpus': self._corpus,
            }
            pickle.dump(data_map, fh, protocol=pickle.HIGHEST_PROTOCOL)


def count_tokens(tokens: List[List[str]]) -> Counter[str]:
    """Count the unique SMILES tokens.

    Parameters
    ----------
    tokens : list
        List of token lists.

    Returns
    -------
    counter : collections.Counter
        Token counter.
    """
    tokens_flattened = itertools.chain(*tokens)
    tokens_counter = collections.Counter(tokens_flattened)
    return tokens_counter


def has_tokens(
        smiles_list: List[str],
        token_set: FrozenSet[str],
) -> bool:
    for token in smiles_list:
        if token in token_set:
            return True
    return False
