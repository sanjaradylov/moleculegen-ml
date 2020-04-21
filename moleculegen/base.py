"""
Base objects for all modules.

Classes
-------
Token
    Token enumeration containing possible tokens from SMILES Vocabulary.
Corpus
    Descriptor that stores corpus of `Vocabulary` or similar instances.
"""

import enum
from typing import Any, FrozenSet, List, Optional, Type


class Token(enum.Enum):
    """Token enumeration class containing possible tokens from SMILES
    Vocabulary. Every token entry is an attribute, which can be attained by
    calling `Token.TOKEN_NAME`. The values of tokens come in form
    (value, rule_class), in which value is SMILES token and rule class is one
    of {'atom', 'other', 'special'}. In case of atoms, the names of tokens are
    identical to the values (atomic symbols), i.e.
    `Token.TOKEN_NAME.name == Token.TOKEN_NAME.value[0]`. Otherwise, the names
    are self-explanatory, i.e. `'=' == TOKEN.BOND_DOUBLE.value[0]`.
    """

    def _generate_next_value_(name, start, count, last_values):
        """Return the name of an atomic symbol as its value and 'atom' as a
        rule class. See class attributes for the complete list of members.
        """
        return name, 'atom'

    # Atomic symbols.
    Ac = enum.auto()
    Ag = enum.auto()
    Al = enum.auto()
    Am = enum.auto()
    Ar = enum.auto()
    As = enum.auto()
    At = enum.auto()
    Au = enum.auto()
    B = enum.auto()
    Ba = enum.auto()
    Be = enum.auto()
    Bh = enum.auto()
    Bi = enum.auto()
    Bk = enum.auto()
    Br = enum.auto()
    C = enum.auto()
    Ca = enum.auto()
    Cd = enum.auto()
    Ce = enum.auto()
    Cf = enum.auto()
    Cl = enum.auto()
    Cm = enum.auto()
    Co = enum.auto()
    Cr = enum.auto()
    Cs = enum.auto()
    Cu = enum.auto()
    Db = enum.auto()
    Dy = enum.auto()
    Er = enum.auto()
    Es = enum.auto()
    Eu = enum.auto()
    F = enum.auto()
    Fe = enum.auto()
    Fm = enum.auto()
    Fr = enum.auto()
    Ga = enum.auto()
    Gd = enum.auto()
    Ge = enum.auto()
    H = enum.auto()
    He = enum.auto()
    Hf = enum.auto()
    Hg = enum.auto()
    Ho = enum.auto()
    Hs = enum.auto()
    I = enum.auto()
    In = enum.auto()
    Ir = enum.auto()
    K = enum.auto()
    Kr = enum.auto()
    La = enum.auto()
    Li = enum.auto()
    Lr = enum.auto()
    Lu = enum.auto()
    Md = enum.auto()
    Mg = enum.auto()
    Mn = enum.auto()
    Mo = enum.auto()
    Mt = enum.auto()
    N = enum.auto()
    Na = enum.auto()
    Nb = enum.auto()
    Nd = enum.auto()
    Ne = enum.auto()
    Ni = enum.auto()
    No = enum.auto()
    Np = enum.auto()
    O = enum.auto()
    Os = enum.auto()
    P = enum.auto()
    Pa = enum.auto()
    Pb = enum.auto()
    Pd = enum.auto()
    Pm = enum.auto()
    Po = enum.auto()
    Pr = enum.auto()
    Pt = enum.auto()
    Pu = enum.auto()
    Ra = enum.auto()
    Rb = enum.auto()
    Re = enum.auto()
    Rf = enum.auto()
    Rh = enum.auto()
    Rn = enum.auto()
    Ru = enum.auto()
    S = enum.auto()
    Sb = enum.auto()
    Sc = enum.auto()
    Se = enum.auto()
    Sg = enum.auto()
    Si = enum.auto()
    Sm = enum.auto()
    Sn = enum.auto()
    Sr = enum.auto()
    Ta = enum.auto()
    Tb = enum.auto()
    Tc = enum.auto()
    Te = enum.auto()
    Th = enum.auto()
    Ti = enum.auto()
    Tl = enum.auto()
    Tm = enum.auto()
    U = enum.auto()
    V = enum.auto()
    W = enum.auto()
    Xe = enum.auto()
    Y = enum.auto()
    Yb = enum.auto()
    Zn = enum.auto()
    Zr = enum.auto()

    # Other specifications.
    BOND_SINGLE = ('-', 'other')
    BOND_DOUBLE = ('=', 'other')
    BOND_TRIPLE = ('#', 'other')
    BOND_AROMATIC = (':', 'other')
    BRANCH_LEFT = ('(', 'other')
    BRANCH_RIGHT = (')', 'other')
    HIGHER_RING_CLOSURE = ('%', 'other')
    PERIOD = ('.', 'other')
    BRACKET_LEFT = ('[', 'other')
    BRACKET_RIGHT = (']', 'other')
    CHIRAL_SPEC = ('@', 'other')
    CHARGE_POSITIVE = ('+', 'other')
    CHARGE_NEGATIVE = ('-', 'other')
    nH = ('[nH]', 'other')
    ONE = ('1', 'other')
    TWO = ('2', 'other')
    THREE = ('3', 'other')
    FOUR = ('4', 'other')
    FIVE = ('5', 'other')
    SIX = ('6', 'other')
    SEVEN = ('7', 'other')
    EIGHT = ('8', 'other')
    NINE = ('9', 'other')

    # Special characters.
    BOS = ('{', 'special')  # Beginning of SMILES.
    EOS = ('}', 'special')  # End of SMILES.
    PAD = ('_', 'special')  # Padding.
    UNK = ('*', 'special')  # Unknown.

    def __init__(self, token, rule_class):
        self.token = token
        self.rule_class = rule_class

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
        return (
            f"{cls.BOS.token}"
            f"{cls.crop(smiles)}"
            f"{cls.PAD.token * padding_len}"
            f"{cls.EOS.token}"
        )

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
        modified_smiles = smiles.lstrip(cls.BOS.token).rstrip(cls.EOS.token)
        if padding:
            modified_smiles = modified_smiles.replace(cls.PAD.token, '')

        return modified_smiles

    @classmethod
    def get_rule_class_members(cls, rule_class: str) -> FrozenSet[str]:
        """Return immutable set of token values from rule class `rule_class`.

        Parameters
        ----------
        cls : Token
            Token enumeration.
        rule_class : {'atom', 'other', 'special'}
            Rule class.

        Returns
        -------
        members : frozenset of str
            Token values from `rule_class`.
        """
        return frozenset(
            map(
                lambda entry: entry.token,
                filter(
                    lambda entry: entry.rule_class == rule_class,
                    iter(cls)
                )
            )
        )


class Corpus:
    """Descriptor that stores corpus of `Vocabulary` or similar instances.

    Parameters
    ----------
    attribute_name : str
        The attribute name of the processed instance.
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
