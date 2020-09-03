"""
Base objects for all modules.

Classes
-------
StateInitializerMixin
    A mixin class for specific state initialization techniques during
    model training.
Token
    A token enumeration, which stores a diverse set of atomic, bond, and
    composite tokens.
"""

__all__ = (
    'StateInitializerMixin',
    'Token',
)


import functools
from typing import Any, Callable, FrozenSet, List, Optional
from mxnet import nd, np


class Token:
    """Token class containing sets of valid SMILES symbols grouped by rule
    class (atoms, non-atomic symbols like bonds and branches, and special
    symbols like beginning-of-SMILES and padding).
    """

    # Atomic symbols. (We store the original ones, although lowercase symbols
    # should also be considered during tokenization).
    ATOMS = frozenset([
        'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
        'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
        'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
        'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
        'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
        'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
        'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
        'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
        'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
        'Zn', 'Zr'
    ])

    # Bonds, charges, etc.
    NON_ATOMS = frozenset([
        '-', '=', '#', ':', '(', ')', '%', '.', '[', ']', '@', '+', '-',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\', '/'
    ])

    # Subcompounds.
    AGGREGATE = frozenset([
        '[C@H]', '[C@@H]', '@@', '10'
    ])

    # Special tokens not presented in the SMILES vocabulary.
    BOS = '{'  # Beginning of SMILES.
    EOS = '}'  # End of SMILES.
    PAD = '_'  # Padding.
    UNK = '*'  # Unknown.
    SPECIAL = frozenset(BOS + EOS + PAD + UNK)

    @classmethod
    def get_all_tokens(cls) -> FrozenSet[str]:
        """Get a set of all the valid tokens defined in the class.

        Returns
        -------
        tokens : frozenset of str
        """
        return cls.SPECIAL.union(
            cls.AGGREGATE.union(cls.NON_ATOMS.union(cls.ATOMS)))

    @classmethod
    def augment(
            cls,
            smiles: str,
            padding_len: int = 0,
    ) -> str:
        """Prepend Beginning-of-SMILES token and append End-of-SMILES token
        to the specified SMILES string `smiles`. If specified, append Padding
        token `padding_len` times.

        Parameters
        ----------
        smiles : str
            Original SMILES string.
        padding_len : int, default 0
            The number of Padding tokens to include.

        Returns
        -------
        modified_smiles : str
            Modified SMILES string.
        """
        modified_smiles = f'{Token.BOS}{smiles.lstrip(Token.BOS)}'
        if not (
                modified_smiles.endswith(Token.PAD)
                or modified_smiles.endswith(Token.EOS)
        ):
            modified_smiles += Token.EOS
        if padding_len > 0:
            modified_smiles += cls.PAD * padding_len
        if modified_smiles.endswith(Token.EOS):
            modified_smiles = f'{modified_smiles.rstrip(Token.EOS)}{Token.EOS}'

        return modified_smiles

    @classmethod
    def crop(
            cls,
            smiles: str,
            padding: bool = False,
    ) -> str:
        """Remove Beginning-of-SMILES, End-of-SMILES, and, if specified,
        Padding tokens from `smiles`.

        Parameters
        ----------
        smiles : str
            Original SMILES string.
        padding : bool, default False
            Whether to remove the Padding token.

        Returns
        -------
        modified_smiles : str
            Modified SMILES string.
        """
        modified_smiles = smiles.lstrip(cls.BOS)
        if padding:
            modified_smiles = modified_smiles.replace(cls.PAD, '')
            modified_smiles = modified_smiles.rstrip(cls.EOS)
        else:
            modified_smiles = modified_smiles.replace(cls.EOS, '')

        return modified_smiles

    @classmethod
    def tokenize(cls, smiles: str) -> Optional[List[str]]:
        """Tokenize `smiles` string.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        tokens : list of str or None
            None, if at least one character not from cls.
            list of tokens from cls, otherwise.

        TODO:
        Notes
        -----
        It is probably not applicable to all cases of data.
        However, it is lot better than simply treating every character as a
        token when in fact two or more characters (e.g. [nH] and Cl) are all
        single entities. To comprehend how to tokenize universally and
        effectively, some research on the domain topic is required.
        """
        token_list: List[str] = []  # The resulting list of tokens.

        char_no = 0  # Points to the current position.
        while char_no < len(smiles):
            # Check if tokens of length `n_chars` are in our `smiles`.
            for n_chars in range(max(map(len, cls.AGGREGATE)), 1, -1):
                token = smiles[char_no:char_no + n_chars]
                if token in cls.AGGREGATE:
                    token_list.append(token)
                    char_no += n_chars
                    break
            # If not, then try processing single- and double-char tokens.
            else:
                one_char_token = smiles[char_no]
                two_char_token = smiles[char_no:char_no + 2]
                if (
                        # Double-char token that cannot be represented as
                        # two separate atoms; 'no' will be treated as two
                        # single-char tokens 'n' and 'o', while 'Se' or 'Na' as
                        # double-char.
                        two_char_token.title() in cls.ATOMS
                        and two_char_token[-1].title() not in cls.ATOMS
                        or
                        two_char_token[0].isupper()
                        and two_char_token in cls.ATOMS
                ):
                    token_list.append(two_char_token)
                    char_no += 2
                elif (
                        one_char_token.title() in cls.ATOMS  # n, o, etc.;
                        or one_char_token in cls.NON_ATOMS   # -, #, \., etc.;
                        or one_char_token in cls.SPECIAL     # {, }, _, *.
                ):
                    token_list.append(one_char_token)
                    char_no += 1
                # If we didn't find any valid token, then return an empty list.
                else:
                    return []

        return token_list


class StateInitializerMixin:
    """A mixin class for specific state initialization techniques during
    model training.

    The main purpose is to provide additional API for batch samplers that
    encourage nontrivial ways of state (re-)initialization during training.
    For example, while sampling a mini-batch from SMILESBatchColumnSampler,
    one should reinitialize states when finally all the samples from the
    mini-batch encounter Token.EOS.
    """

    @classmethod
    def init_states(
            cls,
            model: Any,
            mini_batch: Any,
            states: Optional[List[np.ndarray]] = None,
            detach: bool = True,
            init_state_func: Optional[Callable[[Any], nd.ndarray.NDArray]] = None,
            *args,
            **kwargs,
    ) -> List[np.ndarray]:
        """(Re-)initialize the hidden state(s) of `model`.

        Parameters
        ----------
        model : any
            A Gluon model.
        mini_batch : any
            A (mini-)batch instance. Basically, tuples or dataclasses.
        states : list of mxnet.nd.ndarray.NDArray, default None
            The previous hidden states of `model`.
            If None, then a new state list will be initialized.
        detach : bool, default True
            Whether to detach the sates from the computational graph.
        init_state_func : callable, any -> mxnet.nd.ndarray.NDArray,
                default None
            A distribution function to initialize `states`.
            If None, the previous states will be returned.
            Recommended to use `StateInitializerMixin.init_state_func` method
            to declare this callable.

        Returns
        -------
        states : list of mxnet.np.ndarray
            A state list.

        Raises
        ------
        AttributeError
            If `mini_batch` does not have `shape` attribute.
            If `model` does not implement `begin_state` method.

        Notes
        -----
        The context of the hidden states is the same as the `model`s context.
        """
        if not hasattr(mini_batch, 'shape'):
            raise AttributeError(
                '`mini_batch` must store `shape` attribute.'
            )
        if not hasattr(model, 'begin_state'):
            raise AttributeError(
                '`model` must implement `begin_state` method.'
            )

        if states is None:
            states = model.begin_state(
                batch_size=mini_batch.shape[0],
                func=init_state_func,
            )

        if detach:
            states = [state.detach() for state in states]

        return states

    @staticmethod
    def init_state_func(
            func: Callable[[Any], nd.ndarray.NDArray] = nd.zeros,
            **func_kwargs,
    ) -> Callable[[Any], nd.ndarray.NDArray]:
        """Return distribution callable `func` with arbitrary
        (non-default) arguments `func_kwargs` specified in advance. Use
        primarily for state initialization.

        Parameters
        ----------
        func : callable, any -> mxnet.nd.ndarray.NDArray
            One of the distribution functions from mxnet.nd.random or
            functions like nd.zeros.
        func_kwargs : dict, default None
            Parameters of `func` excluding `shape`.

        Returns
        -------
        func : callable
            Partial function callable.

        Raises
        ------
        ValueError
            If `shape` parameter is included in `distribution_args`.
            This parameter will be used separately in state initialization.
        """
        if 'shape' in func_kwargs or 'ctx' in func_kwargs:
            raise ValueError(
                '`shape` and `ctx` parameters should not be passed, since '
                'they are processed internally by the model.'
            )

        return functools.partial(func, **func_kwargs)
