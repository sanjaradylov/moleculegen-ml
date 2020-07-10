"""
Utilities to create vocabularies of SMILES documents.

Classes
-------
Vocabulary
    Mapping SMILES characters into their numerical representation.

Functions
---------
count_tokens
    Create token counter.
"""

import collections
import itertools
import pickle
from typing import Counter, Dict, Generator, List, Optional, Sequence, Union

from mxnet.gluon.data import SimpleDataset

from .base import Token


class Vocabulary:
    """A dictionary to map SMILES characters into numerical representation.
    Read SMILES strings from `dataset` and create a token counter,
    or work directly with `tokens` if specified. Eventually obtain unique
    SMILES tokens and their frequencies to create token-to-index and
    index-to-token variables.

    Parameters
    ----------
    dataset : SMILESDataset, default None
        SMILES data set.
    tokens : list, default None
        SMILES tokens.
    need_corpus : bool, default False
        If True, property `corpus` is loaded from descriptor `Corpus`.
        To keep original data as tokens and load a corpus,
        pass non-empty `dataset` or `tokens`.
    load_from_pickle : str, default None
        If passed, load all the attributes from the file named
        `load_from_pickle`.
    save_to_pickle : str, default None
        If passed, save all the attributes to the file named
        `save_to_pickle`.

    Attributes
    ----------
    idx_to_token : list
        List of tokens.
    token_to_idx : dict
        Token-index mapping.
    token_freqs : dict
        The token-to-count mapping.
    corpus : list
        Original data set corpus. Accessible only if corpus is needed.
    """

    def __init__(
            self,
            dataset: Optional[SimpleDataset] = None,
            tokens: Optional[List[str]] = None,
            need_corpus: bool = False,
            load_from_pickle: Optional[str] = None,
            save_to_pickle: Optional[str] = None,
    ):
        self._corpus = None

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
                or [Token.tokenize(sample) for sample in dataset]
            )
            counter: Counter[str] = count_tokens(tokens)

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
                self._corpus: List[List[int]] = [self[line] for line in tokens]

        if save_to_pickle is not None:
            with open(save_to_pickle, 'wb') as fh:
                data_map = {
                    'token_freqs': self._token_freqs,
                    'idx_to_token': self._idx_to_token,
                    'token_to_idx': self._token_to_idx,
                    'corpus': self._corpus,
                }
                pickle.dump(data_map, fh, protocol=pickle.HIGHEST_PROTOCOL)

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
        unknown_idx = len(self)

        if isinstance(tokens, str):
            return self._token_to_idx.get(tokens, unknown_idx)
        elif isinstance(tokens, (list, tuple)):
            return [
                self._token_to_idx.get(token, unknown_idx)
                for token in tokens
            ]
        else:
            raise KeyError(
                f"`tokens` must be of type str or list/tuple of str, "
                f"not {type(tokens)}."
            )

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
