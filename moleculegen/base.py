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
from typing import Any, Callable, List, Optional, Tuple, Type


# (For `Token` class) Atomic symbols.
_ATOMIC_SYMBOLS = [
    'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
    'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
    'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
    'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
    'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
    'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
    'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn', 'Ru',
    'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc',
    'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr'
]
# (For `Token` class) Bonds, branches, ring closures, disconnections, others.
_BBRDO = {
    'BOND_SINGLE': '-',
    'BOND_DOUBLE': '=',
    'BOND_TRIPLE': '#',
    'BOND_AROMATIC': ':',

    'BRANCH_LEFT': '(',
    'BRANCH_RIGHT': ')',

    'HIGHER_RING_CLOSURE': '%',

    'PERIOD': '.',

    'BRACKET_LEFT': '[',
    'BRACKET_RIGHT': ']',

    'CHIRAL_SPEC': '@',

    'CHARGE_POSITIVE': '+',
    'CHARGE_NEGATIVE': '-',

    'nH': '[nH]',
}
_BBRDO.update({
    key: value for key, value in zip(
        ['ONE', 'TWO', 'THREE', 'FOUR',
         'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'],
        range(1, 10),
    )
})
# (For `Token` class) Special tokens.
_SPECIAL = {
    'BOS': '{',  # Beginning of SMILES.
    'EOS': '}',  # End of SMILES.
    'PAD': '_',  # Padding token.
    'UNK': '*',  # Unknown token.
}


# (For `Token` class).
def __init_token_instance_attrs(
        self: 'Token',
        token: str,
        rule_class: str,
):
    """This function will be used to initialize (__init__) Token enumeration.
    """
    setattr(self, 'token', token)
    setattr(self, 'rule_class', rule_class)


# (For `Token` class) Token methods and properties.
__TOKEN_METHODS: List[Tuple[str, Callable]] = [
    (
        '__init__',
        lambda self, token, rule_class: __init_token_instance_attrs(
            self, token, rule_class)
    ),
]
# (For `Token` class) SMILES token keys, values, and their rule classes.
__TOKEN_NAMES: List[Tuple[str, Tuple[str, str]]] = [
    (element, (element, 'atom'))
    for element in _ATOMIC_SYMBOLS + [e.lower() for e in _ATOMIC_SYMBOLS]
]
__TOKEN_NAMES += [
    (key, (value, 'bbrdo'))
    for key, value in _BBRDO.items()
]
__TOKEN_NAMES += [
    (key, (value, 'special'))
    for key, value in _SPECIAL.items()
]


Token = enum.Enum(value='Token', names=__TOKEN_NAMES, module=__name__)
Token.__doc__ = """
Token enumeration class containing possible tokens from SMILES Vocabulary.
Every token entry is an attribute, which can be attained by calling
`Token.TOKEN_NAME`. The values of tokens come in form (value, rule_class),
in which value is SMILES token and rule class is one of
{'atom', 'bbord', 'special'}. In case of atoms, the names of tokens are
identical to the values (atomic symbols), i.e.
`Token.TOKEN_NAME.name == Token.TOKEN_NAME.value[0]`. Otherwise, the names are
self-explanatory, i.e. `'=' == TOKEN.BOND_DOUBLE.value[0]`.
""".lstrip('\n')


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
