#### moleculegen

---

Generate novel molecules using recurrent neural networks.

This project is an attempt to reproduce experiments from
*Segler et al. Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks*.

---

**Install dependencies**

Create and activate a virtual environment and run:
```
$ pip install mxnet-cu101mkl
```

---

**Run experiments**

- Create a csv-file from ChEMBL database according to a specified SQL query:
```
$ cd data
$ sqlite3 -header -csv YourChEMBL.db < stage1_smiles.sql > stage1_compounds.csv
```
- Run the main module:
```
$ cd ..
$ python3 run.py data/stage1_compounds.csv
```

---

**Notes**

- Run `python3 run.py --help` to print help message.