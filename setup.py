"""
Set up moleculegen-ml package.
"""

from setuptools import find_packages, setup
from urllib.parse import urljoin


PACKAGE_URL = 'http://github.com/sanjaradylov/moleculegen-ml/'


setup(
    name='moleculegen',
    description='Generate novel molecules using recurrent neural networks',
    version='1.0.dev',
    author='Sanjar Ad[iy]lov',
    url=PACKAGE_URL,
    project_urls={
        'Documentation': urljoin(PACKAGE_URL, 'wiki'),
        'Source Code': urljoin(PACKAGE_URL, 'tree/master/moleculegen'),
    },
    packages=find_packages(exclude=['*tests']),
    include_package_data=False,
    install_requires=[
        'mxnet-cu101mkl',
    ],
)
