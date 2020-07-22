"""
Describe molecular feature space w/ encoders and feature transformers.

Classes
-------
OneHotEncoder
    One-hot encoder functor.

MorganFingerprint
    Apply the Morgan algorithm to a set of compounds to get circular
    fingerprints.


Functions
---------
get_descriptors
    Create a dictionary of descriptors.
get_descriptors_df
    Create a pandas.DataFrame of descriptors.

get_physchem_descriptors
    Create physicochemical descriptors.
"""

__all__ = (
    'get_descriptors',
    'get_descriptors_df',
    'get_physchem_descriptors',
    'MorganFingerprint',
    'OneHotEncoder',
)


from .base import (
    get_descriptors,
    get_descriptors_df,
)
from .common import (
    OneHotEncoder,
)
from .fingerprints import (
    MorganFingerprint,
)
from .physicochemical import (
    get_physchem_descriptors,
)
