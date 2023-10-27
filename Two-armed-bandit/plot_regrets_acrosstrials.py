from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
import pandas as pd 
import numpy as np
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
plt.rcParams.update({
    "text.usetex": True
})


def get_regret(df, no_trials):    
    '''
    Get the regret for each trial
    Args:
        df (pd.DataFrame): DataFrame containing the data
        no_trials (int): number of trials per game
    '''
    mean_regret = np.zeros(no_trials)
    ste_regret = np.zeros(no_trials)
    for t in range(no_trials):
        df_i = df[df['trial'] == t]['regret']
        mean_regret[t] = np.mean(df_i)
        ste_regret[t] = np.std(df_i)/np.sqrt(len(df_i))
    _95ci = 1.96 * ste_regret
    return mean_regret, _95ci

def plot(regret_dict, engine):
    '''
    Plot the regret for each trial for each engine
    Args:
        regret_dict (dict): dictionary containing the regret data for each trial
        engine (str): engine used
    '''
    names, regrets = zip(*regret_dict.items())
    plt.rcParams["figure.figsize"] = (2.6,2)
    gpt_count = 0
    for idx, name in enumerate(names):
        style = '-'
        if name.startswith('GPT'):
            gpt_count += 1
            colour = 'C0'
            # if start and end of meta-in-context learning are plotted, make the line dashed to distinguish them
            if gpt_count > 1:
                style = '--'
        elif name.startswith('UCB'):
            colour = 'C2'
        elif name.startswith('Greedy'):
            colour = 'C4'
        else:
            colour = f'C{idx}'
        plt.plot(np.arange(1, 11), regrets[idx][0], label=name, linestyle=style, color=colour, alpha=0.8)
        plt.fill_between(np.arange(1, 11), regrets[idx][0] - regrets[idx][1], regrets[idx][0] + regrets[idx][1], linestyle=style, color=colour, alpha=0.1)
    plt.ylabel('Regret')
    plt.xlabel('Trials')
    plt.legend(bbox_to_anchor=(-0.1,1.02,1,0.2),  loc='lower center', ncol=len(names)/2, frameon=False, prop={'size': 8}, columnspacing=0.7)
    sns.despine()
    plt.tight_layout(pad=0.05)
    data_path = f'./plots/{task}/{engine}'
    os.makedirs(data_path, exist_ok=True) #creates directory if not created, it it exists, it does nothing
    plt.yticks([0, 2, 4])
    plt.savefig(data_path + '/regrets_pertrialstep.pdf')

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--no_trials', type=int, default=10, help='Number of trials per game')
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
# parser.add_argument('--engine', default='text-davinci-002', help='Engine to be used')
parser.add_argument('--engine', default='random', help='Engine to be used')
parser.add_argument('--baselines_to_include', nargs='+', default=['Greedy', 'UCB'])

args = parser.parse_args()
task = f"var{args.mean_var}_{args.reward_var}"
#if path exists
regrets = {}

#DataFrames for regret - get the regret for each trial
if args.engine.startswith('text-davinci'):
    gpt_start = pd.read_csv(f'./data/{task}/{args.engine}/meta-learning.csv'); gpt_start = gpt_start[gpt_start['game'] ==0]; regrets['GPT-3_start'] = get_regret(gpt_start, args.no_trials)
    gpt_end = pd.read_csv(f'./data/{task}/{args.engine}/meta-learning.csv'); gpt_end = gpt_end[gpt_end['game'] ==args.no_games-1]; regrets[f'GPT-3 after {args.no_games} tasks'] = get_regret(gpt_end, args.no_trials)
if args.engine.startswith('random'):
    random_df = pd.read_csv(f'./data/{task}/{args.engine}/meta-learning.csv'); regrets['random'] = get_regret(random_df, args.no_trials)
if 'Greedy' in args.baselines_to_include:
    greedy_df = pd.read_csv(f'./data/{task}/greedy_data.csv'); regrets['Greedy'] = get_regret(greedy_df, args.no_trials)
if 'UCB' in args.baselines_to_include:
    ucb_df = pd.read_csv(f'./data/{task}/UCB_data.csv') ; regrets['UCB'] = get_regret(ucb_df, args.no_trials)
if 'Thompson Sampling' in args.baselines_to_include:
    ts_df = pd.read_csv(f'./data/{task}/TS_data.csv'); regrets['Thompson Sampling'] = get_regret(ts_df, args.no_trials)

#Plot regrets
plot(regrets, args.engine)













