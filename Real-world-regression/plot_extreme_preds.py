import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


plt.rcParams.update({
    "text.usetex": True
})

def count_no_preds_outside_range(path, tasks, range_targ, no_trials):
    '''
    Count the number of predictions outside the range of the target variable for each trial
    Args:
        path (str): path to the csv file
        tasks (list): list of tasks to consider
        range_targ (list): range of the target variable
        no_trials (int): number of trials to consider
    Returns:
        count (np.array): array of counts for each trial
    '''
    count = np.zeros(no_trials)
    for task in tasks:
        df = pd.read_csv(f'{path}{task}.csv')
        for trial in range(no_trials):
            df_trial = df[df['trial'] == trial]
            for index, row in df_trial.iterrows(): 
                if (row['ypred'] <= range_targ[0]) or (row['ypred'] >= range_targ[1]):
                    count[trial] += 1

    len_df_trial = len(df[df['trial'] == trial])
    count  /= (len_df_trial*len(tasks))

    return count

def barplot_comp(percs1, percs2, no_trials, engine, features):
    '''
    Barplot the percentage of predictions outside the range of the target variable for each trial for two models. 
    Save the plot - Figure 4.B in the paper
    Args:
        percs1 (list): list of percentages for the first model
        percs2 (list): list of percentages for the second model
        no_trials (int): number of trials to consider
        engine (str): name of the engine
        features (int): number of features to use
    '''
    plt.rcParams["figure.figsize"] = (1.5,2)
    x = np.arange(1, 1+no_trials)
    plt.bar(x-0.2, 100*percs1, 0.4, label=args.engine, color='C0', alpha=0.8)
    plt.bar(x+0.2, 100*percs2, 0.4, label=args.engine, color='C1', alpha=0.8)
    plt.ylabel("Extreme \n" +r"predictions ($\%$)")
    plt.xlabel('Trials')
    plt.xticks(x)
    plt.legend(bbox_to_anchor=(-0.2,1.02,1,0.2), loc='lower center', ncol=1 , frameon=False, prop={'size': 8})
    sns.despine()
    plt.tight_layout(pad=0)
    plt.savefig(f'./plots/{engine}/extremepreds_{features}features_{no_trials}points.pdf')

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=5, help='Number of features used for each task')
parser.add_argument('--points', type=int, default=5, help='Number of points used for each task')
parser.add_argument('--engine', type=str, default='random', help='Which engine used')
parser.add_argument('--mil_tasks', type=int, default=4, help='Number of meta-in-context learning tasks used before testing')
args = parser.parse_args()

#Tasks are the ones in the llama ML folder
tasks = []
for filename in os.listdir(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points'):
    if (filename.endswith(".csv")) and filename.startswith(f'meta_{args.engine}_='):
        task = filename.split(f'meta_{args.engine}_=')[1][:-4]
        tasks.append(task)

# Get path to the csv files
metapath = f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points/meta_{args.engine}_='
path = f'./data/{args.engine}_{args.features}features_{args.points}points/{args.engine}_='

# Get the percentage of predictions outside the range of the target variable for each trial
percs1 = count_no_preds_outside_range(path, tasks, [-1, 1], args.points)
percs2 = count_no_preds_outside_range(metapath, tasks, [-1, 1], args.points)
# Plot the results
barplot_comp(percs1, percs2, args.points, args.engine, args.features)



