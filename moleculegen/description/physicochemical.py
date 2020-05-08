"""
Create physicochemical descriptors.
"""

import array
from typing import Dict, MutableSequence

from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors

from .base import get_descriptors


DESCRIPTOR_FUNCTIONS = {
    '#BertzCT': BertzCT,
    '#RotatableBonds': CalcNumRotatableBonds,
    'TotalPolarSurfaceArea': CalcTPSA,
    'MolecularWeight': ExactMolWt,
    'LogP': MolLogP,
    '#HAcceptors': NumHAcceptors,
    '#HDonors': NumHDonors,
}


def get_physchem_descriptors(compounds: MutableSequence[str]) \
        -> Dict[str, array.array]:
    return get_descriptors(compounds, DESCRIPTOR_FUNCTIONS)
