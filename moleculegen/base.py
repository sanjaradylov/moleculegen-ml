"""
Base objects for all modules.

Classes:
    Token: A token enumeration, which stores a diverse set of atomic, bond,
           and composite tokens.
"""

__all__ = (
    'Token',
)


import re
from typing import FrozenSet, List


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
        '-', '=', '#', ':', '(', ')', '.', '[', ']', '@', '+', '-', '*',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\', '/'
    ])

    # Subcompounds, chiral specification.
    AGGREGATE = frozenset([
        '@@', '@TH', '@AL', '@SP', '@TB', '@OH',
        # '(=O)', '[nH]',
    ])

    # Special tokens not presented in the SMILES vocabulary.
    BOS = '{'  # Beginning of SMILES.
    EOS = '}'  # End of SMILES.
    PAD = '_'  # Padding.
    UNK = '^'  # Unknown.
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
    def tokenize(
            cls,
            smiles: str,
            match_bracket_atoms: bool = False,
    ) -> List[str]:
        """Tokenize `smiles` string.

        Parameters
        ----------
        smiles : str
            SMILES string.
        match_bracket_atoms : bool, default False
            Whether to treat the subcompounds enclosed in [] as separate tokens.

        Returns
        -------
        tokens : list of str
            An empty list, if at least one character not from `cls`.
            A list of tokens from `cls`, otherwise.
        """
        if match_bracket_atoms:
            token_list: List[str] = []

            for subcompound in cls._BRACKETS_RE.split(smiles):
                if subcompound.startswith('['):
                    token_list.append(subcompound)
                else:
                    token_list.extend(cls._tokenize(subcompound))

            return token_list
        else:
            return cls._tokenize(smiles)

    _BRACKETS_RE = re.compile(
        pattern=r"""
            (?P<brackets>     # Begin a capture group.
                \[            # Match opening bracket square to capture an atom.
                    [^\[\]]+  # Match atoms, charges, etc., except '[' and ']'.
                \]            # Match closing bracket square to capture an atom.
            )                 # End a capture group.
        """,
        flags=re.VERBOSE,
    )
    _DIGITS_RE = re.compile(r'(?P<digits>\d{2,}).*')

    @classmethod
    def _tokenize(cls, smiles: str) -> List[str]:
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
                four_char_token = smiles[char_no:char_no + 4]
                if (
                        # Double-char atoms like '[se]'.
                        four_char_token.startswith('[')
                        and four_char_token.endswith(']')
                        and four_char_token[1:].islower()
                        and four_char_token[1:-1].title() in cls.ATOMS
                ):
                    token_list.append(four_char_token)
                    char_no += 4
                elif (
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
                        or one_char_token in cls.SPECIAL     # {, }, _, ^.
                ):
                    token_list.append(one_char_token)
                    char_no += 1
                elif one_char_token.startswith('%'):  # Digits > 9 like %10 or %108.
                    match = cls._DIGITS_RE.match(smiles, char_no + 1)
                    if match is not None:
                        tokens = f'%{match.group("digits")}'
                        token_list.append(tokens)
                        char_no += len(tokens)
                    else:
                        return []
                # If we didn't find any valid token, then return an empty list.
                else:
                    return []

        return token_list
