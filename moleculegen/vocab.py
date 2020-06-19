"""
Utilities to create corpus and vocabulary of SMILES documents.

Classes
-------
Vocabulary
    Mapping SMILES characters into their numerical representation.

Functions
---------
count_tokens
    Create token counter.
tokenize
    Tokenize documents.
"""

import collections
import itertools
from typing import Counter, Dict, Generator, List, Optional, Sequence, Union

from mxnet.gluon.data import SimpleDataset

from .base import Token


class Vocabulary:
    """Dictionary to map SMILES characters into numerical representation.
    Read strings from `dataset` and create token (character) counter,
    or work directly with `tokens` and `counter` if at least the former is
    specified. Eventually obtain unique SMILES characters and their frequencies
    to create token-to-index and index-to-token variables.

    Parameters
    ----------
    dataset : SMILESDataset, default None
        SMILES data set.
    tokens : list, default None
        SMILES tokens.
    counter : collections.Counter, default None
        Token counter.
    need_corpus : bool, default False
        If True, property `corpus` is loaded from descriptor `Corpus`.
        To keep original data as tokens and load a corpus,
        pass non-empty `dataset` or `tokens`.

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
            counter: Optional[Counter[str]] = None,
            need_corpus: bool = False,
    ):
        assert not (need_corpus and dataset is None and tokens is None), \
            "If you wish to load corpus, specify the original data set."
        assert any(arg for arg in (dataset, tokens, counter)), \
            "At least one of `dataset`, `tokens`, `counter` must be specified."

        tokens: List[List[str]] = (
            tokens
            or [Token.tokenize(sample) for sample in dataset]
        )
        counter: Counter[str] = counter or count_tokens(tokens)

        self._token_freqs: Dict[str, int] = dict(sorted(
            counter.items(), key=lambda c: c[1], reverse=True))

        # 'Unknown' token is not used in training/prediction.
        self._idx_to_token: List[str] = list(Token.SPECIAL - {Token.UNK})
        self._token_to_idx: Dict[str, int] = {
            token: id_ for id_, token in enumerate(self._idx_to_token)
        }
        for token in set(counter.keys()) - set(self._idx_to_token):
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._token_to_idx)

        self._need_corpus = need_corpus
        if self._need_corpus:
            self._corpus: List[List[int]] = [self[line] for line in tokens]

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
        """Return the number of unique tokens (SMILES characters).

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
        """Original data set corpus. Accessible only if corpus is needed.

        Returns
        -------
        corpus : list of list of int
        """
        if not self._need_corpus:
            raise AttributeError(
                'a corpus is loaded during initialization if parameter '
                '`need_corpus`=True.'
            )
        return self._corpus

    @property
    def idx_to_token(self) -> List[str]:
        """The list of unique tokens.

        Returns
        -------
        idx_to_token : list
        """
        return self._idx_to_token

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """The token-to-id mapping.

        Returns
        -------
        token_to_idx : dict
        """
        return self._token_to_idx

    @property
    def token_freqs(self) -> Dict[str, int]:
        """The token-to-count mapping.

        Returns
        -------
        token_freqs : dict
        """
        return self._token_freqs

    def get_tokens(self, indices: List[int]) -> List[str]:
        """Return the tokens corresponding to the indices.

        Returns
        -------
        tokens : list
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


def tokenize(strings: List[str]) -> List[List[str]]:
    """Split every SMILES string into characters to get tokens.

    Parameters
    ----------
    strings : list
        SMILES strings.

    Returns
    -------
    strings : list
        List of SMILES tokens.
    """
    return [list(s) for s in strings]
