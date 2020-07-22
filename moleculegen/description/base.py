"""
Utilities to create tables of descriptors.

Functions
---------
get_descriptors
    Create a dictionary of descriptors.
get_descriptors_df
    Create a pandas.DataFrame of descriptors.
"""

__all__ = (
    'get_descriptors',
    'get_descriptors_df',
)


import array
from typing import Callable, Dict, MutableSequence

import pandas as pd
from rdkit.Chem import MolFromSmiles


def get_descriptors_df(
        compounds: MutableSequence[str],
        function_map: Dict[str, Callable],
) -> pd.DataFrame:
    """Create a pandas.DataFrame instance comprising the descriptors of
    `compounds` declared in `function_map`. The resulting data frame's index
    is `compounds`, the column names are `function_map`s keys, and the values
    of the columns are the calculated descriptors.

    Parameters
    ----------
    compounds : mutable sequence of str
        (Mutable) Sequence of SMILES strings.
    function_map : dict
        Keys are the column names being assigned to a data frame,
        values are the descriptor function callables to apply to `compounds`.

    Returns
    -------
    data_frame : pd.DataFrame
    """
    if not isinstance(compounds, pd.DataFrame):
        compounds = pd.DataFrame({'smiles': compounds})

    for name, function in function_map.items():
        results = (
            compounds['smiles']
            .pipe(lambda series: series.apply(MolFromSmiles))
            .pipe(lambda series: series.apply(function))
        )
        compounds = compounds.assign(**{name: results})

    compounds.set_index(keys='smiles', drop=True, inplace=True)

    return compounds


def get_descriptors(
        compounds: MutableSequence[str],
        function_map: Dict[str, Callable],
) -> Dict[str, array.array]:
    """Create a dictionary of descriptors, where keys are descriptor names
    from `function_map.keys()` and values are the calculated descriptors
    on `compounds`.

    Parameters
    ----------
    compounds : mutable sequence of str
        (Mutable) Sequence of SMILES strings.
    function_map : dict
        Keys are the names being assigned to a resulting dictionary,
        values are the descriptor function callables to apply to `compounds`.

    Returns
    -------
    result : dict
    """
    results: Dict[str, array.array] = {}
    for name, fn in function_map.items():
        results[name] = array.array(
            'f',
            (fn(MolFromSmiles(smiles)) for smiles in compounds)
        )
    return results
