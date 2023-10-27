This folder contains scripts for analyzing meta-in-context learning from LLMs on one-dimensional regression experiments.

## 1. query.py
The file which queries the LLM to generate tasks and train on them. It also stores the generated data in CSV files.

Usage example:
'python meta_query.py --engines random --no_trainings 5 --no_tasks 5 --function_type linear --params -2 1 -100 1 0 0 1'

## 2. plot.py
The file which plots the generated data stored from meta_query.py. 

Usage example:
'python plot.py --engines random --no_trainings 5 --no_tasks 5 --plotting_experiment all --function_type linear --params -2 1 -100 1 0 0 1 --bayesian_preds ground truth random'

## 3. utils.py
Contains utility functions for meta_query.py and plot.py. 

## Directory Structure:
data/: Contains generated data stored in CSV files.

plots/: Contains generated plots saved as PNG files. There is two subdirectories: 'training' and 'prior' which contain plots for the training and prior experiments respectively (even if ran simultaneously).
