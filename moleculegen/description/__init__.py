"""
Describe molecular feature space w/ encoders and feature transformers.

Classes
-------
OneHotEncoder
    One-hot encoder functor.

MorganFingerprint
    Apply the Morgan algorithm to a set of compounds to get circular fingerprints.
MoleculeTransformer
    Convert SMILES compounds into RDKit molecules.


Functions
---------
check_compounds_valid
    Convert SMILES compounds into RDKit molecules.

get_descriptors
    Create a dictionary of descriptors.
get_descriptors_df
    Create a pandas.DataFrame of descriptors.

get_physchem_descriptors
    Create physicochemical descriptors.
"""

__all__ = (
    'check_compounds_valid',
    'get_descriptors',
    'get_descriptors_df',
    'get_physchem_descriptors',
    'MoleculeTransformer',
    'MorganFingerprint',
    'OneHotEncoder',
)


from .base import (
    check_compounds_valid,
    get_descriptors,
    get_descriptors_df,
)
from .common import (
    MoleculeTransformer,
    OneHotEncoder,
)
from .fingerprints import (
    MorganFingerprint,
)
from .physicochemical import (
    get_physchem_descriptors,
)
