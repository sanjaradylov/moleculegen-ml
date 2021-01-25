"""
Describe molecular feature space w/ encoders and feature transformers.

Constants
---------
PHYSCHEM_DESCRIPTOR_MAP
    Key - descriptor name, value - descriptor callable.

Classes
-------
OneHotEncoder
    One-hot encoder functor.

InternalTanimoto
    Build internal Tanimoto similarity matrix.
MorganFingerprint
    Apply the Morgan algorithm to a set of compounds to get circular fingerprints.
MoleculeTransformer
    Convert SMILES compounds into RDKit molecules.
RDKitDescriptorTransformer
    Calculate RDKit descriptors.


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
    'InternalTanimoto',
    'MoleculeTransformer',
    'MorganFingerprint',
    'OneHotEncoder',
    'PHYSCHEM_DESCRIPTOR_MAP',
    'RDKitDescriptorTransformer',
)


from .base import (
    check_compounds_valid,
    get_descriptors,
    get_descriptors_df,
)
from .common import (
    MoleculeTransformer,
    OneHotEncoder,
    RDKitDescriptorTransformer,
)
from .fingerprints import (
    InternalTanimoto,
    MorganFingerprint,
)
from .physicochemical import (
    PHYSCHEM_DESCRIPTOR_MAP,
    get_physchem_descriptors,
)
