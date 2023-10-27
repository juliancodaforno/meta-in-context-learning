import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances, rbf_kernel, laplacian_kernel, sigmoid_kernel, polynomial_kernel
import statsmodels.regression.mixed_linear_model as smlm
import statsmodels.api as smapi
import statsmodels.tools.tools as sm
from tqdm import tqdm
import argparse

plt.rcParams.update({
    "text.usetex": True
})

def get_regressors(engine_name,  tasks, no_points, no_meta_trainings, no_feats, similarity, tasks_sim_included=True, best_meta_included=False, current_task_sim_included=False):
    """
    This function accesses the csv files of the meta-in-context learning tasks and get the regressors to use for the regression: similarities, Trials and MSE.
    Args:
        engine_name (str): name of the engine
        tasks (list): list of tasks to consider
        no_points (int): number of points to use for each task
        no_meta_trainings (int): number of meta-in-context learning tasks used before testing
        no_feats (int): number of features to use
        similarity (function): similarity measure to use
        tasks_sim_included (bool): whether to include the similarity between the meta training points and the test points
        best_meta_included (bool): whether to include the similarity between the best meta task and the test task
        current_task_sim_included (bool): whether to include the similarity between the current test point and the previous test points within the same task
    Returns:
        dict (dict): dictionary of the regressors to use for the regression: similarities, Trials and MSE.

    """
    dict = {'mse': []}
    dict.update({'Trial': []})
    if current_task_sim_included:
        dict.update({r"$\kappa_{task}$": []})
    if tasks_sim_included:
        dict.update({"Task\nsimilarity": []})
    if best_meta_included:
        dict.update({r"$\kappa_{best}$": []})
    path_meta = f'./data/{no_meta_trainings}meta_{no_feats}features_{no_points}points/meta_{engine_name}_='
    # Loop over the tasks
    for idx, task in tqdm(enumerate(tasks)):
        df = pd.read_csv(f'{path_meta}{task}.csv')
        # Loop over each trial of testing (after meta-in-context learning) and compute the similarity measure between the meta training points and the test points. Store it in a list within a dictionary for each similarity measure.
        for row in range(len(df)):
            x = np.array(eval(df['x'][row])).reshape((1, -1)) #Test points
            meta_training = np.array([eval(df[f'task{i}x'][row]) for i in range(1, no_meta_trainings+1)]) #Matrix of meta training points
            meta_training = meta_training.reshape((no_meta_trainings*no_points, no_feats)) #flatten the matrix from (no_meta_trainings, no_points, no_features) to (no_meta_trainings*no_points, features)
            dict['mse'].append((df['ytrue'][row] - df['ypred'][row] )**2)
            if tasks_sim_included:
                dict["Task\nsimilarity"].append(np.mean(np.abs(similarity(x, meta_training)))) #Similarity measure between the meta training points and the test points
            if best_meta_included:
                dict[r"$\kappa_{best}$"].append(np.max(np.abs(similarity(x, meta_training))))
            dict['Trial'].append(df['trial'][row])
            if df['trial'][row] != 0:
                task_previous_x = np.array([eval(df['x'][prev_idx])for prev_idx in range(0, df['trial'][row])])
                if current_task_sim_included:
                    dict[r"$\kappa_{task}$"].append(np.mean(np.abs(similarity(x, task_previous_x))))
            else:
                if current_task_sim_included:
                    dict[r"$\kappa_{task}$"].append(0)
           
    return dict


def plot_barplot(dict, sim_name, engine_name, include_intercept):    
    """
    This function plots the barplot of the regressors stored in dict on the MSE. Figure 4.C in the paper.

    Args:
        dict (dict): dictionary of the regressors to use for the regression: similarities, Trials and MSE.
        sim_name (str): name of the similarity measure used
        engine_name (str): name of the engine
        include_intercept (bool): whether to include the intercept in the regression
    """
    plt.rcParams["figure.figsize"] = (1.2,2)
    df = pd.DataFrame(dict)
        
    Y = df['mse']
    Y = (Y - Y.mean())/Y.std() 
    X = df.drop(columns=['mse'])
    #standardize each column in X
    for column in X:
        if column != 'Tasks':
            X[column] = (X[column] - X[column].mean())/X[column].std()
    if include_intercept:
        X = sm.add_constant(X)

    # Fit regression model    
    model = smapi.OLS(Y, X).fit()

    # Plot regression coefficients
    for idx in range(len(model.params)):
        plt.bar(idx, model.params[idx], yerr=model.conf_int()[1][idx] - model.params[idx], alpha=0.6, color='C3')
    plt.xticks(range(len(model.params)), X.keys(), rotation=25, fontsize=7)
    plt.ylabel("Regression\n coefficients")
    plt.tight_layout(pad=0.05)
    print(model.summary())     #Print fitting statistics
    #Add some space to the right of the plot
    plt.subplots_adjust(right=0.9)
    sns.despine()
    plt.savefig(f'./plots/{engine_name}/{sim_name}_sim.pdf')

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--features', type=int, default=5, help='Number of features used for each task')
parser.add_argument('--points', type=int, default=5, help='Number of points used for each task')
parser.add_argument('--engine', type=str, default='random', help='Which engine used')
parser.add_argument('--mil_tasks', type=int, default=4, help='Number of meta-in-context learning tasks used before testing')
parser.add_argument('--tasks_sim_included', type=bool, default=True, help='Whether to include the similarity between the meta training points and the test points')
parser.add_argument('--sim_measures', nargs='+', default=['rbf'], help='Similarity measures to use')
parser.add_argument('--include_intercept', type=bool, default=False, help='Whether to include the intercept in the regression')
args = parser.parse_args()
    
# Create list of similarity measures to use
sim_measures = []; names = []
if 'rbf' in args.sim_measures:
    sim_measures.append(rbf_kernel); names.append('rbf')
if 'cosine' in args.sim_measures:
    sim_measures.append(cosine_similarity); names.append('cosine')
if 'euclidean' in args.sim_measures:
    sim_measures.append(euclidean_distances); names.append('euclidean')
if 'manhattan' in args.sim_measures:
    sim_measures.append(manhattan_distances); names.append('manhattan')

#Tasks are the ones in the llama ML folder
tasks = []
for filename in os.listdir(f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points'):
    if (filename.endswith(".csv")) and filename.startswith(f'meta_{args.engine}_='):
        task = filename.split(f'meta_{args.engine}_=')[1][:-4]
        tasks.append(task)

metapath = f'./data/{args.mil_tasks}meta_{args.features}features_{args.points}points/meta_{args.engine}_='

#Run
for idx, sim_measure in tqdm(enumerate(sim_measures)):
    print(f' for {names[idx]}--------------------------')
    dict = get_regressors(args.engine, tasks, args.points, args.mil_tasks, args.features, sim_measure, args.tasks_sim_included)
    plot_barplot(dict, names[idx], args.engine, args.include_intercept)

