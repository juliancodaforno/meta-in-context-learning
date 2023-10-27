from sys import stderr
from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
import pandas as pd 
import numpy as np
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
plt.rcParams.update({
    "text.usetex": True,
})



def get_regrets_across_games(no_games, df):
    """
    Outputs the regrets across games.

    Args:

    no_games (int): Number of games
    df (pd.dataframe): Dataframe.

    Returns:

    means (np.array): list of regrets accross games
    std_errs (np.array): list of standard errors accross games    
    """
    no_participants = int(np.max(df['subject']))
    means = np.ones((no_games, no_participants))  * np.inf   #will average over the no_participants at the end of the function
    std_errs = np.ones((no_games))  * np.inf

    for n in range(no_participants):
        for game in range(no_games):
            means[game, n] = np.mean(df[(df['game']==game) & (df['subject']==n+1)]['regret'])

    for game in range(no_games):
        std_errs[game] = np.std(means[game])/np.sqrt(no_participants)

    means = np.mean(means, axis=1)
    return means, std_errs

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
parser.add_argument('--engine', default='random', help='Engine to be used')
parser.add_argument('--filetype', default='.pdf', choices=['.png', '.pdf'], help='Filetype of the plot')
parser.add_argument('--y_lower_bound_isgreedy', type=bool, default=True, help='If true, the lower bound of the y axis is the greedy regret. If false, the lower bound is 0.')
args = parser.parse_args()

task = f'var{args.mean_var}_{args.reward_var}'
path = f'./data/{task}/{args.engine}/meta-learning.csv'

llm_df = pd.read_csv(path); llm_df = llm_df[llm_df['game'] < args.no_games].reset_index(drop=True)  #Get rid of excess games! 

#LLM regrets
mean_regrets_across_games,  std_errs_regrets_across_games = get_regrets_across_games(args.no_games, llm_df)
if args.y_lower_bound_isgreedy:
    greedy_df = pd.read_csv(f'./data/{task}/greedy_data.csv')
    #rename game to participant
    greedy_df = greedy_df.rename(columns={'game':'subject'})
    #Add game column which is only equal to 0
    greedy_df['game'] = 0
    greedy_regrets, _ = get_regrets_across_games(1, greedy_df)


#Plotting
#Squeeze width by two to fit in overleaf
plt.rcParams["figure.figsize"] = (1.3,2)
X = { f'{args.engine}': mean_regrets_across_games}
X_CI = { f'{args.engine}': std_errs_regrets_across_games*1.96} 
idxs = X.keys()
for idx in idxs:
    plt.plot(np.arange(1, args.no_games+1), X[idx], label=idx, alpha=0.8)
    plt.xticks(np.arange(1, args.no_games+1, 1))  #Only show xticks for the integers
    plt.fill_between(np.arange(1, args.no_games+1), X[idx] - X_CI[idx], X[idx] + X_CI[idx], alpha=0.1)
plt.ylabel('Regret')
plt.xlabel('Tasks')
sns.despine()
#Add to the ytick the greedy regret
if args.y_lower_bound_isgreedy:
    #add a line for the greedy regret
    plt.axhline(y= float(np.round(greedy_regrets, 2)), color='C4', linewidth=1.5, label='Greedy', alpha=0.8)
    #lower bound of the y axis is the greedy regret - 0.1 to include the greedy line
    plt.ylim(bottom=float(np.round(greedy_regrets, 2)-0.1))

plt.legend(bbox_to_anchor=(0,1.02,1,0.2),  loc='lower center', ncol=1, frameon=False, prop={'size': 8})

plt.tight_layout(pad=0.05)
plt.savefig(f'./plots/{task}/{args.engine}/meta_fullregrets_acrosstasks{args.filetype}')

