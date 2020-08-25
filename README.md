# moleculegen

![example workflow name](https://github.com/sanjaradylov/moleculegen-ml/workflows/moleculegen/badge.svg)
![PythonVersion](https://img.shields.io/badge/python-3.7-blue)

This project is an attempt to create a Python package for the complete *de novo* drug design
cycle similar to [Segler et al](#reference).

In brief, the project comprises
[two stages](https://github.com/sanjaradylov/moleculegen-ml/projects) of analysis.
In [Stage 1](https://github.com/sanjaradylov/moleculegen-ml/projects/1), we train and evaluate
a recurrent neural network on a large, general set of molecules to generate novel molecules.
In [Stage 2](https://github.com/sanjaradylov/moleculegen-ml/projects/2), we choose specific
targets of interest to create a predictive model, and then perform transfer learning on a
focused set of active molecules.


## Documentation
For now, our [wiki](https://github.com/sanjaradylov/moleculegen-ml/wiki) serves as a
documentation (or rather a user guide) for the project.

[Projects](https://github.com/sanjaradylov/moleculegen-ml/projects) keep track of the
current state of the project. 

## Installation

It is convenient to set up dependencies using environment management systems like
[conda](https://conda.io/en/latest/index.html) or
[virtualenv](https://virtualenv.pypa.io/en/stable/).
We use the latest stable version of Ubuntu to test our project.

Download, install, and set up [Miniconda](https://conda.io/en/latest/miniconda.html).

Create a new environment and install dependencies (see `environment.yml` and `requirements.txt`):
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

We also provide an example script `scripts/run.py`. To observe the command line arguments print a help message:
```bash
$ python3 run.py --help
```

If you do not have your own data set, please download [ChEMBL database](https://www.ebi.ac.uk/chembl/) and
follow the instructions above.

Create a text file from ChEMBL database according to a specified SQL query.
You can create a dataset manually (without post-processing):

```bash
$ cd data
$ sqlite3 -csv YourChEMBL.db < stage1_smiles.sql > stage1_compounds.csv
```
or applying several filters (e.g. removal of long SMILES strings) to the loaded dataset:
```bash
$ cd scripts
$ export DB_FILENAME=YourChEMBL.db
$ python3 process_stage1_data.py -o ../data/stage1_compounds_post_processed.csv
```


## Reference

1. Segler et al. Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks. https://arxiv.org/pdf/1701.01329.pdf