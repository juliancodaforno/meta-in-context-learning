import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import is_float
import argparse
import os

plt.rcParams.update({
    "text.usetex": True,
})


#Hyperparameters
parser = argparse.ArgumentParser()
use_swarmplot = True
use_violinplot = True
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
parser.add_argument('--filetype', default='pdf', choices=['png', 'pdf'], help='Filetype of the plot')
parser.add_argument('--ylims', nargs='+', default=[-50, 50], type=int, help='Y limits of the plot')
parser.add_argument('--engine', default='random', help='Engine to be plotted')
args = parser.parse_args()


task = f"var{args.mean_var}_{args.reward_var}"
df = pd.read_csv(f'data/{task}/{args.engine}/priors.csv')
df[df['game'] < args.no_games].reset_index(drop=True)
no_queries = max(df['query_idx']) + 1
no_runs = int(len(df)/(args.no_games*no_queries ))
real_prior = lambda: np.random.normal(0, np.sqrt(args.mean_var))
real_priors = [real_prior() for _ in range(no_runs* no_queries)]

plt.rcParams["figure.figsize"] = (2.6,1.6)
observations = np.empty((args.no_games, no_runs * no_queries))
for game in range(args.no_games):
    observations[game] = df[df['game'] == game]['answers'].values
    #print std of observations
    print(f'game {game}: std {np.std(observations[game])}')
    #Print mean of observations
    print(f'game {game}: mean {np.mean(observations[game])}')
            
if use_swarmplot:
    observations = np.concatenate((observations, np.array(real_priors).reshape(1, -1)), axis=0)
    #Make observations a list of length args.no_games + 1 of arrays of length no_runs * no_queries
    observations = [observations[i] for i in range(args.no_games + 1)]
    colors = ['C0' for _ in range(args.no_games)] + ['C1']

    #use stripplot to show all points
    sns.stripplot(data=observations,palette=colors, orient='v', size=2, jitter=0.1, alpha=0.2)
      
if use_violinplot:
    for game in range(args.no_games):
        #plot the observations
        plt.violinplot(observations[game], positions=[game], showmeans=True, showmedians=True)          
    plt.violinplot(real_priors, positions=[args.no_games], showmeans=True, showmedians=True)

    #add color  and one label to all the violinplot
    for idx, pc in enumerate(plt.gca().collections):
        if idx > args.no_games:
            pc.set_alpha(0.2)
            if idx < len(plt.gca().collections) - 6:  # the last 6 are all the features of the last violinplot
                pc.set_facecolor('C0')
                pc.set_edgecolor('C0')
            else:
                pc.set_facecolor('C1')  
                pc.set_edgecolor('C1')


#Add the real prior as x tick name
xtick_names = [f'{i}' for i in range(1, args.no_games + 1)]
#add the real prior as x tick name 
xtick_names.append('True')
plt.xticks(np.arange(0, args.no_games + 1), xtick_names)
plt.ylabel('Reward priors')
plt.ylim(top=args.ylims[1], bottom=args.ylims[0])
sns.despine()
plt.xlabel('Tasks')
plt.tight_layout(pad=0.05)
os.makedirs(f'./plots/{task}/{args.engine}', exist_ok=True)
plt.savefig(f'./plots/{task}/{args.engine}/priors.{args.filetype}')