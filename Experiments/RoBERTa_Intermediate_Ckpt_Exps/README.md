# Evaluation on Intermediate Checkpoints from RoBERTa Pretraining (RQ2)

Run `run_{complex/simple}.py` to run all the experiments for all 84 intermediate checkpoints across 3 seeds and 7 datasets depending on the classification head of choice.

You can choose to modify these values by -
- Modify models to use by changing line 24-28 in `run.py`
- Modify seeds to use by changing line 9 in `run.py`
- Modify datasets to use by changing line 10 in `run.py`