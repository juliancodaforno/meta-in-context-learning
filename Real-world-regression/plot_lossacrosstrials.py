import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from utils import get_mse_ci, get_mse_ci_indivtasks

plt.rcParams.update({
    "text.usetex": True
})

def plot(mses, cis, labels, points, features, engine, indiv_task = False, task_name=False):
    '''
    Plot the RMSE across trials for each model and save the plot
    
    Args:
        mses (list): list of mean squared errors for each model
        cis (list): list of confidence intervals for each model
        labels (list): list of labels for each model
        points (int): number of points to use for each task
        features (int): number of features to use
        engine (str): name of the engine
        indiv_task (bool): whether to plot the results for each task individually
        task_name (str): name of the task to plot the results for
        '''
    plt.rcParams["figure.figsize"] = (2.4,2)
    if indiv_task:
        os.makedirs(f'./plots/{features}feats_indiv_tasks', exist_ok=True)
    for idx, ci in enumerate(cis):
        colour = None
        if labels[idx].startswith('GPT-3'):
            colour = 'C0'
        if labels[idx].startswith('GPT-4'):
            colour = 'C5'
        if labels[idx] == 'BLR':
            colour = 'C4'
        if labels[idx] == 'RandomForest':
            colour = 'C2'
        if colour is None:
            colour = f"C{idx - 1if idx >= 2 else 0}"
        plt.plot(np.arange(1, 1+points), mses[idx],  label=labels[idx], linestyle='--' if 'after 5 tasks' in labels[idx] else None, color=colour)
        plt.fill_between(np.arange(1, 1+points), mses[idx] - ci, mses[idx] + ci,  alpha=0.2, color=colour)
    #Ensure the labelling is for each color C0, C1, C2, etc. not for the fill_
    plt.xticks(np.arange(1, 1+points))    
    plt.ylabel('RMSE')
    plt.xlabel('Trials')
    sns.despine()
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(-0.1,1.05,1,0.2), ncol=2, prop={'size': 7}, columnspacing=0.5)
    plt.tight_layout(pad=0.0)
    os.makedirs(f'./plots/{engine}', exist_ok=True)
    plt.savefig(f'./plots/{engine}/{features}feats{"_indiv_tasks/" + str(task_name) if indiv_task else "_average_over_tasks"}.pdf', dpi=300)

parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=5, help='Number of features used for each task')
parser.add_argument('--points', type=int, default=5, help='Number of points used for each task')
parser.add_argument('--engine', type=str, default='random', help='Which engine used')
parser.add_argument('--mil_tasks', type=int, default=4, help='Number of meta-in-context learning tasks used before testing')
parser.add_argument('--ploting_indiv_tasks', type=bool, default=False, help='Whether to plot the results for each task individually')
parser.add_argument('--ploting_averaged_over_tasks', type=bool, default=True, help='Whether to plot the results averaged over tasks') # Figure 4.A in the paper
parser.add_argument('--baselines_to_include', type=list, default=['RF', 'BLR_skl'], choices=['RF', 'LR', 'SVM', 'BLR_skl'], help='Which baselines to include in the plot')
args = parser.parse_args()


#Tasks ran are the ones stored in the mil data folder - so can just list the files in that folder
tasks = []
for filename in os.listdir(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points'):
    if (filename.endswith(".csv")) and filename.startswith(f'meta_{args.engine}_='):
        task = filename.split(f'meta_{args.engine}_=')[1][:-4]
        tasks.append(task)

models = []
datapaths = []
# ICL data retrieval - ensure you have run the corresponding llm runs for the engine
if args.engine == 'gpt-4':
    models.append('GPT-4')
    datapaths.append(f'./data/gpt-4_{args.features}features_{args.points}points/gpt-4_=')
    # Hash the following two lines if you don't want to plot the results for GPT-3. Here because was used to compare in the appendix
    models.append('GPT-3')
    datapaths.append(f'./data/text-davinci-002_{args.features}features_{args.points}points/text-davinci-002_=')
elif args.engine.startswith('text'):
    models.append('GPT-3')
    datapaths.append(f'./data/{args.engine}_{args.features}features_{args.points}points/{args.engine}_=')
else:
    models.append(args.engine)
    datapaths.append(f'./data/{args.engine}_{args.features}features_{args.points}points/{args.engine}_=')    

# MICL data retrieval - ensure you have run the corresponding llm runs for the engine
if args.engine == 'gpt-4':
    models.append('GPT-4 after 5 tasks')
    datapaths.append(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points/meta_gpt-4_=')
    # Hash the following two lines if you don't want to plot the results for GPT-3. Here because was used to compare in the appendix
    models.append('GPT-3 after 5 tasks')
    datapaths.append(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points/meta_text-davinci-002_=')
elif args.engine.startswith('text'):
    models.append('GPT-3 after 5 tasks')
    datapaths.append(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points/meta_{args.engine}_=')
else:
    models.append(f'{args.engine} after 5 tasks')
    datapaths.append(f'./data/{args.engine}_{args.features}features_{args.points}points/{args.engine}_=')    
   
# Baselines data retrieval - ensure you have run the corresponding baseline runs 
if 'RF' in args.baselines_to_include:
    models.append('RandomForest')
    datapaths.append(f'./data/RF_{args.features}features_{args.points}points/RF=')
if 'LR' in args.baselines_to_include:
    models.append('LinearRegression')
    datapaths.append(f'./data/LR_{args.features}features_{args.points}points/LR=')
if 'SVM' in args.baselines_to_include:
    models.append('SVM')
    datapaths.append(f'./data/SVM_{args.features}features_10points/SVM=')
if 'BLR' in args.baselines_to_include: 
    models.append('BLR')
    datapaths.append(f'./data/BLR_{args.features}features_{args.points}points/BLR=')
if 'BLR_skl' in args.baselines_to_include:
    models.append('BLR')
    datapaths.append(f'./data/BLR_skl_{args.features}features_{args.points}points/BLR_skl=')

# Visualisation purposes for each task. Not included in the paper
if args.ploting_indiv_tasks:
    for task in tasks:
        mses = []
        cis = []
        for path in datapaths:
            mse, ci = get_mse_ci_indivtasks(path, task, args.points)
            mses.append(mse)
            cis.append(ci)
        plot(mses, cis, models, args.points, args.features, indiv_task = True, task_name=task)
        plt.close()
    
# Figure 4.A in the paper
if args.ploting_averaged_over_tasks:
    mses = []
    cis = []
    for path in datapaths:
        mse, ci = get_mse_ci(path, tasks,args.points)
        mses.append(mse)
        cis.append(ci)
    plot(mses, cis, models, args.points, args.features, args.engine)