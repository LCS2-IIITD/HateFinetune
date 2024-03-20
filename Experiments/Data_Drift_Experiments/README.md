# Data Drift Experiments

First run `dataDirBuild.ipynb` to generate the datasets in the formats needed for this experiments.

Run `run.py` to run all the experiments for all 7x6 dataset pairs using BERT Model

You can choose to modify these values by -
- Modify models to use by changing line 54 and 55 in `run.py`
- Modify skip certain datasets by adding a if statement with continue after line 9 in `run.py`