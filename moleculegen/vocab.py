"""
Utilities to create corpus and vocabulary of SMILES documents.

Classes
-------
Corpus
    Descriptor that stores corpus of `Vocabulary` or similar instances.
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
import warnings
from typing import Counter, Dict, List, Optional, Sequence, Union

from mxnet.gluon.data import SimpleDataset

from .base import Corpus
from .utils import SpecialTokens


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
    all_tokens : list
        List of all tokens from original data set.
        Accessible only if corpus is needed.
    corpus : list
        Original data set corpus. Accessible only if corpus is needed.
    """

    corpus: List[List[int]] = Corpus('all_tokens')

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

        self._need_corpus = need_corpus
        if self._need_corpus:
            self._all_tokens = (
                tokens or tokenize([sample for sample in dataset]))
        else:
            self._all_tokens = None

        tokens = (
            tokens
            or self._all_tokens
            or tokenize([sample for sample in dataset])
        )
        counter = counter or count_tokens(tokens)

        self.token_freqs = dict(sorted(
            counter.items(), key=lambda c: c[1], reverse=True))

        self._idx_to_token: List[str] = []
        self._token_to_idx: Dict[str, int] = {}
        for token in counter.keys():
            self._idx_to_token.append(token)
            self._token_to_idx[token] = len(self._token_to_idx)
        self._idx_to_token.append(SpecialTokens.PAD.value)
        self._token_to_idx[SpecialTokens.PAD.value] = len(self._token_to_idx)

    def __len__(self) -> int:
        """Return the number of unique tokens (SMILES characters).

        Returns
        -------
        n : int
        """
        return len(self._idx_to_token)

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
        if isinstance(tokens, str) and len(tokens) == 1:
            return self._token_to_idx.get(tokens, SpecialTokens.UNK.value)
        elif isinstance(tokens, Sequence):
            return [
                self._token_to_idx.get(token, SpecialTokens.UNK.value)
                for token in tokens
            ]
        else:
            raise KeyError("`tokens` must be of type str of sequence of str.")

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
    def all_tokens(self) -> List[List[str]]:
        """The original data set as list of tokens.

        Returns
        -------
        all_tokens : list

        Warns
        -----
        UserWarning
            If self._need_corpus was set to False, which means that the tokens
            were not saved. Refer to your `SMILESDataset` instance to load the
            original set of tokens.
        """
        if not self._need_corpus:
            warnings.warn(
                "Tokens were not obtained from dataset; set need_corpus=True."
            )
        return self._all_tokens

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
