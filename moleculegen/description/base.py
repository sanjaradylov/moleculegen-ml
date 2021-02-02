"""
Utilities to create tables of descriptors.

Functions
---------
check_compounds_valid
    Convert SMILES compounds into RDKit molecules.
get_descriptors
    Create a dictionary of descriptors.
get_descriptors_df
    Create a pandas.DataFrame of descriptors.
"""

__all__ = (
    'check_compounds_valid',
    'get_descriptors',
    'get_descriptors_df',
    'get_descriptor_df_from_mol',
    'InvalidSMILESError',
)


import array
from numbers import Real
from typing import Callable, Dict, Iterable, List

from numpy import NaN
from pandas import DataFrame
from rdkit.Chem import Mol, MolFromSmiles


class InvalidSMILESError(ValueError):
    pass


def check_compounds_valid(
        compounds: Iterable[str],
        invalid: str = 'skip',
        **converter_kwargs,
) -> List[Mol]:
    """Convert SMILES compounds into RDKit molecules.

    Parameters
    ----------
    compounds : iterable of str
        SMILES compounds.
    invalid : {'nan', 'raise', 'skip'}, default='skip'
        Whether to a) replace invalid SMILES with `numpy.NaN`s, b) raise
        `InvalidSMILESError`, or c) ignore them.

    Other Parameters
    ----------------
    converter_kwargs : dict, default={}
        Optional key-word arguments for `rdkit.Chem.MolFromSmiles`.

    Returns
    -------
    molecules : list of rdkit.Chem.Mol

    Raises
    ------
    TypeError
        If any compound is invalid.
    """
    molecules: List[Mol] = []

    for compound in compounds:
        molecule = MolFromSmiles(compound, **converter_kwargs)
        if molecule is not None:
            molecules.append(molecule)
        elif invalid == 'nan':
            molecules.append(NaN)
        elif invalid == 'raise':
            raise InvalidSMILESError(
                f'cannot convert {compound!r} into molecule: invalid compound')
        elif invalid == 'skip':
            continue

    return molecules


def get_descriptors_df(
        compounds: Iterable[str],
        function_map: Dict[str, Callable],
        invalid: str = 'skip',
        **converter_kwargs,
) -> DataFrame:
    """Create a pandas.DataFrame instance comprising the descriptors of
    `compounds` declared in `function_map`. The resulting data frame's index
    is `compounds`, the column names are `function_map`s keys, and the values
    of the columns are the calculated descriptors.

    Parameters
    ----------
    compounds : iterable of str
        SMILES strings.
    function_map : dict, str -> (callable, rdkit.Chem.Mol -> numbers.Real)
        Keys are the column names being assigned to a data frame,
        values are the descriptor function callables to apply to `compounds`.
    invalid : {'nan', 'raise', 'skip'}, default='skip'
        Whether to a) replace invalid SMILES with `numpy.NaN`s, b) raise
        `InvalidSMILESError`, or c) ignore them.

    Other Parameters
    ----------------
    converter_kwargs : dict, default={}
        Optional key-word arguments for `rdkit.Chem.MolFromSmiles`.

    Returns
    -------
    pd.DataFrame
        Column names -- keys of `function_map`.
        Columns -- descriptor results from `function_map`s mapping.

    Raises
    ------
    TypeError
        If any compound is invalid.

    ??? Should we include SMILES strings?
    """
    molecules: List[Mol] = check_compounds_valid(
        compounds, invalid=invalid, **converter_kwargs)

    if not isinstance(compounds, DataFrame):
        compounds = DataFrame()

    for name, function in function_map.items():
        results = array.array('f', map(function, molecules))
        compounds = compounds.assign(**{name: results})

    return compounds


def get_descriptors(
        compounds: Iterable[str],
        function_map: Dict[str, Callable],
        **converter_kwargs,
) -> Dict[str, array.array]:
    """Create a dictionary of descriptors, where keys are descriptor names
    from `function_map.keys()` and values are the calculated descriptors
    on `compounds`.

    Parameters
    ----------
    compounds : iterable of str
        SMILES strings.
    function_map : dict, str -> (callable, rdkit.Chem.Mol -> numbers.Real)
        Keys are the names being assigned to a resulting dictionary,
        values are the descriptor function callables to apply to `compounds`.

    Other Parameters
    ----------------
    converter_kwargs : dict, default={}
        Optional key-word arguments for `rdkit.Chem.MolFromSmiles`.

    Returns
    -------
    dict, str -> array.array of float
    """
    molecules: List[Mol] = check_compounds_valid(compounds, **converter_kwargs)
    results: Dict[str, array.array] = {}

    for name, fn in function_map.items():
        results[name] = array.array('f', (fn(molecule) for molecule in molecules))

    return results


def get_descriptor_df_from_mol(
        molecules: Iterable[Mol],
        function_map: Dict[str, Callable[[Mol], Real]],
) -> DataFrame:
    data_frame = DataFrame()

    for name, function in function_map.items():
        descriptor: List[Real] = [function(molecule) for molecule in molecules]
        data_frame = data_frame.assign(**{name: descriptor})

    return data_frame
