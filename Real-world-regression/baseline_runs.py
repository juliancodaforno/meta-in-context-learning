import sys
import time
from utils import load_data, select_subset, ExactInference
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
import pandas as pd
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import argparse


def run(model, model_name, num_features, num_simulations, num_points, testing_tasks, sklearn=True):
    '''
    Run the model on the testing tasks and save the results in a csv file. 

    Args:
        model (sklearn or ExactInference): model to run
        model_name (str): name of the model
        num_features (int): number of features to use
        num_simulations (int): number of simulations to run
        num_points (int): number of points to use for each task
        testing_tasks (list): list of tasks to run the model on
        sklearn (bool): whether the model is sklearn or ExactInference
    '''
    folder_name = f'data/{model_name}_{num_features}features_{num_points}points'
    #create folder if does not exist
    os.makedirs(folder_name, exist_ok=True)

    for task in tqdm(testing_tasks):
        for i in range(num_simulations):
            #testing
            data = []
            df = load_data(task, num_features)
            X, y, df_subset = select_subset(df, num_points)
            for n in range(num_points):
                if n == 0:
                    ypred = 0
                else:
                    if sklearn:
                        reg = model().fit(X[:n], y[:n])
                        ypred = reg.predict(X[[n]]).item()
                    else:
                        model.fit(X[:n], y[:n])
                        ypred = model.predict(X[[n]]).item()
                row = [i, n, y[n], ypred]
                data.append(row)

            df_result = pd.DataFrame(data, columns=['run', 'trial', 'ytrue', 'ypred'])
            df_result.to_csv(folder_name + f'/{model_name}=' + task + '.csv', index=False, header=False if os.path.exists(folder_name + f'/{model_name}=' + task + '.csv') else True, mode='a')



#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=5, help='Number of features to use')
parser.add_argument('--num_simulations', type=int, default=100, help='Number of simulations to run')
parser.add_argument('--num_points', type=int, default=5, help='Number of points to use for each task')
args = parser.parse_args()


#List of tasks
lists_of_tasks =  ['pitcher', 'movie', 'whitewine', 'concrete', 'car', 'fish', 'occupation', 'medexp', 'tip', 'bone', 'recycle', 'plasma', 'prefecture', 'airfoil', 'mine', 'reactor', 'mammal', 'diabetes', 'air', 'vote', 'homeless', 'obesity', 'birthweight', 'algae', 'cigarette', 'schooling', 'mussel', 'bodyfat', 'sat', 'pinot', 'infant', 'lake', 'afl', 'mileage', 'wage', 'news', 'hitter', 'rent', 'men', 'rebellion', 'mortality', 'abalone', 'basketball', 'monet', 'athlete', 'excavator', 'contraception', 'home', 'laborsupply', 'dropout', 'cpu', 'fuel', 'land', 'highway', 'prostate', 'gambling', 'lung', 'crime', 'diamond', 'salary']
#Delete the tasks that have less than num_features features when loaded
idx = 0
while idx < len(lists_of_tasks):
    df = load_data(lists_of_tasks[idx], normalize_data=False)
    if df.shape[1] < args.num_features + 1:
        print("Deleting task " + lists_of_tasks[idx] + " because it has less than " + str(args.num_features) + " features")
        lists_of_tasks.pop(idx)
    else:
        idx += 1    
testing_tasks = lists_of_tasks.copy()

# Sklearn runs - For paper only plotted RF and BLR_skl
run(RandomForestRegressor, 'RF', args.num_features, args.num_simulations, args.num_points, testing_tasks)
# run(SVR, 'SVM', args.num_features, args.num_simulations, args.num_points, testing_tasks)
# run(LinearRegression, 'LR', args.num_features, args.num_simulations, args.num_points, testing_tasks)
run( BayesianRidge, 'BLR_skl', args.num_features, args.num_simulations, args.num_points, testing_tasks)

#! Exact inference run - somehow did not work compared to sklearn BLR.
# blr_model = ExactInference(num_features=args.num_features, pred_logvar=torch.log(torch.Tensor([1]))) 
# run(blr_model, 'BLR', args.num_features, args.num_simulations, args.num_points, testing_tasks, sklearn=False)


