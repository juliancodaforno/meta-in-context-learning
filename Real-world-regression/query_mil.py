import time
from utils import load_data, select_subset, get_instructions, act_micl
import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm import tqdm
import sys
import openai
from dotenv import load_dotenv
import argparse
import math

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=5, help='number of features to use for each task')
parser.add_argument('--num_points', type=int, default=5, help='number of points to use for each task') 
parser.add_argument('--num_simulations', type=int, default=50, help='number of simulations to run for each task') #In the paper used 50 and appendix 30
parser.add_argument('--num_mil_tasks', type=int, default=4, help='number of tasks to use for meta-in-context-learning before testing') # So if I want a total of 5 tasks, I need to set this to 4
# The following two do not matter for the paper - just used to play around with micl and see if training on the best tasks for a linear regression model is better than training on all tasks
# Did not make it into the paper because it did not make any substantial difference
parser.add_argument('--get_best_meta_training', type=bool, default=False, help='If true, the best tasks for meta-training are selected based on the accuracy of a linear regression model fitted to each task')
parser.add_argument('--best_num_for_meta_training', type=int, default=42, help='Number of best subtasks to use for meta-training') 
parser.add_argument('--engine',default='random')
args = parser.parse_args()  
num_features = args.num_features
num_points = args.num_points
num_simulations = args.num_simulations
engine = args.engine
best_num_for_meta_training = args.best_num_for_meta_training
llm = act_micl #TODO: Change this to your own LLM function
folder_name = f'data/{args.num_mil_tasks}meta_{num_features}features_{num_points}points'
#create folder if does not exist
os.makedirs(folder_name, exist_ok=True)
 
# Pre-processing of the tasks
lists_of_tasks =  ['pitcher', 'movie', 'whitewine', 'concrete', 'car', 'fish', 'occupation', 'medexp', 'tip', 'bone', 'recycle', 'plasma', 'prefecture', 'airfoil', 'mine', 'reactor', 'mammal', 'diabetes', 'air', 'vote', 'homeless', 'obesity', 'birthweight', 'algae', 'cigarette', 'schooling', 'mussel', 'bodyfat', 'sat', 'pinot', 'infant', 'lake', 'afl', 'mileage', 'wage', 'news', 'hitter', 'rent', 'men', 'rebellion', 'mortality', 'abalone', 'basketball', 'monet', 'athlete', 'excavator', 'contraception', 'home', 'laborsupply', 'dropout', 'cpu', 'fuel', 'land', 'highway', 'prostate', 'gambling', 'lung', 'crime', 'diamond', 'salary']
#Delete the tasks that have less than num_features features when loaded
idx = 0
while idx < len(lists_of_tasks):
    df = load_data(lists_of_tasks[idx], normalize_data=False)
    if df.shape[1] < num_features + 1:
        print("Deleting task " + lists_of_tasks[idx] + " because it has less than " + str(num_features) + " features")
        lists_of_tasks.pop(idx)
    else:
        idx += 1    

testing_tasks = lists_of_tasks.copy()

# If we want the best training tasks for micl, we fit a linear regression model to each task and then select the tasks with the highest accuracy. NB: For paper we kept all tasks 
if args.get_best_meta_training:
    lists_of_tasks_rmse = []
    for idx in range(len(lists_of_tasks)):
        df = load_data(lists_of_tasks[idx], num_features)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy()
        # reg = RandomForestRegressor().fit(X, y)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        #calculate RMSE
        rmse = np.sqrt(((y_pred - y) ** 2).mean())
        lists_of_tasks_rmse.append((rmse, idx))
    #Sort the lists_of_tasks based on the accuracy
    lists_of_tasks = [lists_of_tasks[idx] for rmse, idx in sorted(lists_of_tasks_rmse)]

    #Only keep most accurate tasks 
    lists_of_tasks = lists_of_tasks[:best_num_for_meta_training]
    #print the minimum accuracy of the best tasks
    print(f"Best tasks for meta training are: {str(lists_of_tasks)} with an rmse of max {str(sorted(lists_of_tasks_rmse)[best_num_for_meta_training-1][0])}")

# Run the experiments
for task in tqdm(testing_tasks):
    #start equals the number of runs already done if the file exists
    start = pd.read_csv(folder_name + f'/meta_{engine}_=' + task + '.csv'  )['run'].max() + 1 if os.path.exists(folder_name + f'/meta_{engine}_=' + task + '.csv') else 0
    for i in range(start, num_simulations):
        all_tasks = lists_of_tasks.copy()
        meta_tasks = [all_tasks.pop(random.randrange(len(all_tasks))) for _ in range(args.num_mil_tasks)]
        prompt = get_instructions('meta')
        #Meta-training
        #create a dictionary with the data from the meta-training tasks with task1x, task1y, task2x, task2y, etc. to recover the data from the meta-training tasks
        meta_task_data = {f"task{idx}x": math.inf * np.ones((num_features, args.num_mil_tasks))for idx in range(1, 1+args.num_mil_tasks)} #x values
        meta_task_data.update({f"task{idx}y": math.inf * np.ones((args.num_mil_tasks)) for idx in range(1, 1+args.num_mil_tasks)}) #y values
        for meta_training in range(args.num_mil_tasks):
            df = load_data(meta_tasks[meta_training], num_features)  #load the data
            meta_task_data.update({f"task{meta_training+1}name": meta_tasks[meta_training]})  #task name stored
            X, y, df_subset = select_subset(df, num_points)
            prompt += "\nTask " + str(meta_training + 1) + ":\n"
            meta_task_data[f"task{meta_training+1}x"] = df_subset[df_subset.columns[:-1]].to_numpy() #x values stored
            meta_task_data[f"task{meta_training+1}y"] = df_subset[df_subset.columns[-1]].to_numpy() #y values stored

            #prompt concatanation
            for n in range(num_points):
                x_vector = [df_subset[column].iloc[n] for column in df.columns[:-1]]
                prompt = prompt +  "x=[ " +  ', '.join([str(int(x)) if x.is_integer() else str(x) for x in x_vector]) + "], y="
                prompt = prompt + " " + str(y[n])  + "\n"

        #testing
        data = []
        df = load_data(task, num_features)
        X, y, df_subset = select_subset(df, num_points) 
        prompt += "\nTask " + str(args.num_mil_tasks + 1) + ":\n"
        for n in range(num_points): 
            x_vector = [df_subset[column].iloc[n] for column in df.columns[:-1]]
            prompt = prompt +  "x=[ " +  ', '.join([str(int(x)) if x.is_integer() else str(x) for x in x_vector]) + "], y="
            ypred = llm(prompt, engine)
            row = [i, n, y[n], ypred, x_vector, task]
            data.append(row)
            prompt = prompt + " " + str(y[n])  + "\n"

        # store the data
        df_result = pd.DataFrame(data, columns=['run', 'trial', 'ytrue', 'ypred', 'x', 'name'])
        #add the meta-learning data to data for each rows
        for key_, value in meta_task_data.items():
            if 'name' in key_:
                df_result[key_] = value
            else: #if it is not the name, it is the x or y values
                #create the column names for the x and y values
                value_list = [value.tolist() for _ in range(num_points)]
                df_result[key_] = value_list

        df_result.to_csv(folder_name + f'/meta_{engine}_=' + task + '.csv', index=False, header=False if os.path.exists(folder_name + f'/meta_{engine}_=' + task + '.csv') else True, mode='a')
