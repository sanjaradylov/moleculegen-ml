# moleculegen

![moleculegen](https://github.com/sanjaradylov/moleculegen-ml/workflows/moleculegen/badge.svg)
[![release](https://img.shields.io/github/release/sanjaradylov/moleculegen-ml.svg)](https://github.com/sanjaradylov/moleculegen-ml/releases)
[![PythonVersion](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-388/)
[![issues](https://img.shields.io/github/issues/sanjaradylov/moleculegen-ml)](https://github.com/sanjaradylov/moleculegen-ml/issues)

**Moleculegen-ML** [[1]](#reference) is a Python package for *de novo* drug design based
on generative language modeling. It comprises tools for molecular data processing,
SMILES-based language modeling (recurrent networks, autoregressive transformers, convolutional networks) and transfer learning.

## Documentation
For now, our [wiki](https://github.com/sanjaradylov/moleculegen-ml/wiki) serves as a
documentation (or rather a user guide) for the project. Our paper [[1]](#references) is a
survey of various machine learning methods for SMILES-based molecule generation.

If you find **Moleculegen-ML** useful in your research, please consider citing
[[1]](#references).

## Installation

It is convenient to set up dependencies using environment management systems like
[conda](https://conda.io/en/latest/index.html) or
[virtualenv](https://virtualenv.pypa.io/en/stable/).
We use the latest stable version of Ubuntu to test our project.

Download, install, and set up [Miniconda](https://conda.io/en/latest/miniconda.html).

Create a new environment and install dependencies (see `environment.yml` and
`requirements.txt`):
```bash
$ conda env create -f environment.yml
$ conda activate moleculegen
```

If you wish to run experiments on GPU (recommended), please install [CUDA](https://developer.nvidia.com/cuda-toolkit)
(we use version 10.1) and run
```bash
$ pip install -r requirements.txt
```

And finally, install the package:

```bash
$ pip install git+https://github.com/sanjaradylov/moleculegen-ml.git
```


## Usage

See [wiki](https://github.com/sanjaradylov/moleculegen-ml/wiki) for feature overview and documentation.

We provide the benchmarking script `scripts/run.py`. To observe the command line 
arguments print a help message:
```bash
$ python3 run.py --help
```
The paper uses standardized [ChEMBL](https://www.ebi.ac.uk/chembl/) data from
[[4]](#references). If you wish to experiment with your own dataset, consider also the
preprocessing scripts in `queries/` and `scripts/`.


## References

1. Adilov, Sanjar (2021): Neural Language Modeling for Molecule Generation. ChemRxiv.
   Preprint. https://doi.org/10.26434/chemrxiv.14700831.v1 
2. Segler et al. Generating Focused Molecule Libraries for Drug Discovery with 
   Recurrent Neural Networks. https://arxiv.org/pdf/1701.01329.pdf
3. Gupta et al. Generative Recurrent Networks for De Novo Drug Design. DOI: 10.
   1002/minf.201700111
4. Brown et al. GuacaMol: Benchmarking Models for de Novo Molecular Design. J. Chem. 
   Inf. Model. 2019, 59, 1096−1108
5. D.  Polykovskiy  et  al. Molecular  sets  (moses):  a  benchmarking platform for
   molecular generation models. 2020. Front Pharmacol 11:58.
