# moleculegen

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

Download, install, and set up [Miniconda](https://conda.io/en/latest/miniconda.html).

Create a new environment and install dependencies (see `environment.yml` and `requirements.txt`):
```bash
$ conda env create -f environment.yml
$ conda activate moleculegen
```

Clone the repository:

```bash
$ mkdir moleculegen-ml && cd moleculegen-ml
$ git clone https://github.com/sanjaradylov/moleculegen-ml.git
```

## Usage

Download [ChEMBL database](https://www.ebi.ac.uk/chembl/).

Create a text file from ChEMBL database according to a specified SQL query
(note [issues](#issues)). You can create a dataset manually (without post-processing):

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

The main module is `run.py`. To observe the command line arguments including parameter
loading/saving and hyperparameters, print a help message:
```bash
$ python3 run.py --help
``` 

Run the main module (currently, it only trains and evaluates an RNN):
```
$ cd ..
$ python3 run.py data/stage1_compounds.csv
```

## Issues
The project is in development stage. The author is currently concerned with the implementation
of [Stage 1](https://github.com/sanjaradylov/moleculegen-ml/projects/1), which has several
pending [issues](https://github.com/sanjaradylov/moleculegen-ml/issues).

## Reference

1. Segler et al. Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks. https://arxiv.org/pdf/1701.01329.pdf