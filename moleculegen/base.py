"""
Base objects for all modules.

Classes
-------
Token
    Token enumeration containing possible tokens from SMILES Vocabulary.
Corpus
    Descriptor that stores corpus of `Vocabulary` or similar instances.
"""

from typing import Any, FrozenSet, List, NamedTuple, Optional, Type
from mxnet import np


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
        '[nH]', '[C@H]', '[C@@H]', '(=O)', '@@'
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
        """Prepend Beginnig-of-SMILES token and append End-of-SMILES token
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
        """Remove Beginnig-of-SMILES, End-of-SMILES, and, if specified, Padding
        tokens from `smiles`.

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


class Batch(NamedTuple):
    """Named tuple that stores mini-batch items.

    Attributes
    ----------
    x : mxnet.np.ndarray
        Input sample.
    y : mxnet.np.ndarray
        Output sample.
    v_x : mxnet.np.ndarray
        Valid lengths for input sample.
    v_y : mxnet.np.ndarray
        Valid lengths for output sample.
    s : bool
        Whether to (re-)initialize state or not.
    """
    x: np.ndarray
    y: np.ndarray
    v_x: np.ndarray
    v_y: np.ndarray
    s: bool


class Corpus:
    """Descriptor that stores corpus of `Vocabulary` or similar instances.

    Parameters
    ----------
    attribute_name : str
        The attribute name of the processed instance.

    ??? It is not bound to Vocabulary anymore.
    """

    __slots__ = (
        'attribute_name',
    )
    __cache = dict()

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name

    def __get__(
            self,
            instance: Any,
            owner: Optional[Type] = None,
    ) -> List[List[int]]:
        """Obtain a corpus from instance (e.g. all tokens from `Vocabulary`).

        Returns
        -------
        corpus : list of list of int
            Original data as list of token id lists.

        Raises
        ------
        AttributeError
            If getattr(instance, self.attribute_name, None) is None.
        """
        result = self.__cache.get(id(instance))
        if result is not None:
            return result

        try:
            return self.__cache.setdefault(
                id(instance),
                [
                    instance[line]
                    for line in getattr(instance, self.attribute_name)
                ],
            )
        except AttributeError as err:
            err.args = (
                f"{self.attribute_name} of {instance!r} is empty; "
                f"see documentation of {instance!r}.",
            )
            raise

    def __set__(
            self,
            instance: Any,
            value: Any,
    ):
        """Modify the attribute of `instance`.

        Raises
        ------
        AttributeError
            The descriptor is read-only.
        """
        raise AttributeError(
            f"Cannot set attribute {self.attribute_name} of {instance!r}."
        )
