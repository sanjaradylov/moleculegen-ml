"""
Create physicochemical descriptors.

Functions
---------
get_physchem_descriptors
    Create physicochemical descriptors.
"""

__all__ = (
    'get_physchem_descriptors',
)


import array
from numbers import Real
from typing import Callable, Dict, MutableSequence

from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAliphaticRings, CalcNumAromaticRings, CalcNumRotatableBonds, CalcTPSA)
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors

from .base import get_descriptors


DESCRIPTOR_FUNCTIONS: Dict[str, Callable[[Mol], Real]] = {
    '#BertzCT': BertzCT,
    '#RotatableBonds': CalcNumRotatableBonds,
    'TotalPolarSurfaceArea': CalcTPSA,
    'MolecularWeight': ExactMolWt,
    'LogP': MolLogP,
    '#HAcceptors': NumHAcceptors,
    '#HDonors': NumHDonors,
    '#AliphaticRings': CalcNumAliphaticRings,
    '#AromaticRings': CalcNumAromaticRings,
}


def get_physchem_descriptors(compounds: MutableSequence[str]) \
        -> Dict[str, array.array]:
    """Return physicochemical descriptors.

    Parameters
    ----------
    compounds : mutable sequence of str
        (Mutable) Sequence of SMILES strings.

    Returns
    -------
    result : dict
        A dictionary of descriptors, where keys are descriptor names and
        values are the calculated descriptors.
    """
    return get_descriptors(compounds, DESCRIPTOR_FUNCTIONS)
