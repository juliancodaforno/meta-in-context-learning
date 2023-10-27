from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as smapi
import seaborn as sns
import os
import argparse

plt.rcParams.update({
    "text.usetex": True,
})


def plot_behaviour_regression(reg_data, game_range, interaction_effect, intercept, engine):
    ''' Probit regression plot:
    Args:

    reg_data: pandas dataframe with all data
    game_range: [start, end]  game interval to include in regression (both included)
    interaction_effect: boolean, if True include interaction effect between games and V, RU and V/TU
    
    Returns: 
    
    None, saves plot to ./plots/'''

    start_idx, end_idx = game_range[0], game_range[1]
    reg_data = reg_data[reg_data['game']>=start_idx][reg_data['game']<=end_idx]# Slice data between game_range interval
    Y = reg_data['action1']
    X = reg_data
    for col in reg_data.columns:
        if interaction_effect==True:
            reg_data['game'] = (reg_data['game'] - np.mean(reg_data['game']))/np.std(reg_data['game']) #Standardize
            #Add columns for interaction effect with games
            reg_data[r'Task$\times$V/TU'] = reg_data['game'] *  reg_data['V/TU'] 
            reg_data[r'Task$\times$V'] = reg_data['game'] *  reg_data['V'] 
            reg_data[r'Task$\times$RU'] = reg_data['game'] *  reg_data['RU'] 

            if col not in ['V', 'RU', 'V/TU', r'Task$\times$V', r'Task$\times$RU', r'Task$\times$V/TU']:
                X = X.drop(columns=col)
        else:
            if col not in ['V', 'RU', 'V/TU']:
                X = X.drop(columns=col)
    if intercept==True:
        X = sm.add_constant(X)
    model = Probit(Y, X.astype(float))
    probit_model = model.fit()
    plt.rcParams["figure.figsize"] = (2.6,1.6)
    print(probit_model.summary())
    
    predictors = ['V', 'RU', 'V/TU']
    if interaction_effect==True:
        predictors += [r'Task$\times$V', r'Task$\times$RU', r'Task$\times$V/TU']
    if intercept==True:
        predictors += ['const']
    for idx in predictors:
        plt.bar(idx, probit_model.params[idx], yerr=probit_model.conf_int()[1][idx] - probit_model.params[idx], color='C3', alpha=0.6)# 95% confidence interval 
    plt.xticks(rotation=20, fontsize=8)
    plt.ylabel('Regression\n coefficients' )
    sns.despine()
    #add pad to the bottom to make room for legend
    plt.tight_layout(pad=0.05)
    #if directory does not exist, create it
    if not os.path.exists(f'./plots/{task}/{engine}/action1_regressors_barplots'):
        os.makedirs(f'./plots/{task}/{engine}/action1_regressors_barplots')
    plt.savefig(f'./plots/{task}/{engine}/action1_regressors_barplots/games{start_idx if start_idx == end_idx else f"{start_idx}-{end_idx}"}{"_with InteractionEff" if interaction_effect else ""}{"_withIntercept" if intercept else ""}.pdf')

def plot_regret_regression(reg_data, game_range, intercept,  engine, interaction_effect=False):
    ''' LMEM Probit regression plot:
    Args:

    reg_data: pandas dataframe with all data
    game_range: [start, end]  game interval to include in regression
    interaction_effect: boolean, if True include interaction effect between games and V/TU
    
    Returns: 
    
    None, saves plot to ./plots/'''

    variable = 'regret' #Have this here to make it easier to change the variable to plot and before I had reward 
    start_idx, end_idx = game_range[0], game_range[1]
    reg_data = reg_data[reg_data['game']>=start_idx][reg_data['game']<=end_idx]# Slice data between game_range interval
    reg_data['game'] = (reg_data['game'] - np.mean(reg_data['game']))/np.std(reg_data['game']) #Standardize games
    reg_data['trial'] = (reg_data['trial'] - np.mean(reg_data['trial']))/np.std(reg_data['trial']) #Standardize trials
    reg_data[variable] = (reg_data[variable] - np.mean(reg_data[variable]))/np.std(reg_data[variable]) #Standardize trials
    reg_data[r'Task$\times$Trial'] = reg_data['game'] *  reg_data['trial'] #Interaction effect2
    reg_data = reg_data.rename(columns={'game': 'Task', 'trial': 'Trial'}) #renaming for nice plots

    Y = reg_data[variable]
    X = reg_data

    for col in reg_data.columns:
        if interaction_effect==True:
            if col not in ['Task', 'Trial', r'Task$\times$Trial']:
                X = X.drop(columns=col)
        else:
            # if col not in ['game', 'trial']:
            if col not in ['Trial', 'Task']:
                X = X.drop(columns=col)
    
    if intercept==True:
        X = sm.add_constant(X)
    model = smapi.OLS(Y, X)
    model = model.fit()
    print(model.summary())
    plt.rcParams["figure.figsize"] = (1.3,2)

    predictors = ['Trial', 'Task']
    if interaction_effect==True:
        predictors += [r'Task$\times$Trial']
    if intercept==True:
        predictors += ['const']
    for idx in predictors:
        print(f'Beta_{idx}: {model.params[idx]} +- {model.conf_int()[1][idx] - model.params[idx]}')
        plt.bar(idx, model.params[idx], yerr=model.conf_int()[1][idx] - model.params[idx], alpha=0.6, color='C3')# 95% confidence interval 
    plt.ylabel('Regression\n coefficients')
    sns.despine()
    #Add white space at the bottom to make room for legend:
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(bottom=0.19)
    # if directory does not exist, create it
    if not os.path.exists(f'./plots/{task}/{engine}/{variable}_regressors_barplot'):
        os.makedirs(f'./plots/{task}/{engine}/{variable}_regressors_barplot')
    plt.savefig(f'./plots/{task}/{engine}/{variable}_regressors_barplot/games{start_idx if start_idx == end_idx else f"{start_idx}-{end_idx}"}{"_with InteractionEff" if interaction_effect else ""}{"_withIntercept" if intercept else ""}.pdf')


def plot_regression_predictorcoef_across_games(reg_data, bin_size, compare_start_to_end, predictor, intercept, no_games, engine):
    '''Plot an indiviual predictor's Beta coefficient across games for the probit regression -> action1 

    Args:
    reg_data: pandas dataframe with all data
    bin_size: int, number of games to include in each bin
    compare_start_to_end: boolean, if True compare first bin_size games to last bin_size games
    predictor: [str, str], predictor to plot, first str is the name of the predictor, second str is the name of the plot
    intercept: boolean, if True include intercept in regression

    Returns:
    None, saves plot to ./plots/

    '''
    plt.clf()
    Y = reg_data['action1']
    X = reg_data

    for col in reg_data.columns:
        if col not in ['V', 'RU', 'V/TU', 'game']:
            X = X.drop(columns=col)

    no_games = max(reg_data['game']) + 1  #Should be 10 games
    if compare_start_to_end:
        step_size = no_games - bin_size
        print(step_size)
    else:
        step_size = bin_size
    for game in range(0, no_games, step_size):
        game_idx = (X['game'] <  game + bin_size) & (X['game'] >= game)
        if intercept==True:
            X = sm.add_constant(X)
        model = Probit(Y[game_idx], X[game_idx].drop(columns='game'))
        probit_model = model.fit()
        plt.bar(game+bin_size/2, probit_model.params[predictor[0]], yerr=probit_model.bse[predictor[0]], width=bin_size)
    plt.xlabel('Games')
    plt.ylabel(f'{predictor[0]} coefficients within regression of (V, RU, V/TU) -> action1')
    plt.title(f'{predictor[1]} across  games')
    plt.plot()
    if '/' in predictor[0]:   #to avoid csv file error
        predictor[0] = predictor[0].replace('/', ':')
    # If directory does not exist create
    if not os.path.exists(f'./plots/{engine}/indiv_action1_regressor_barplots'):
        os.makedirs(f'./plots/{engine}/indiv_action1_regressor_barplots')
    plt.savefig(f'./plots/{engine}/indiv_action1_regressor_barplots/{predictor[0]}across_games_metalearn.pdf')

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
parser.add_argument('--engine', default='random', help='Engine to be used')
parser.add_argument('--bin_size', type=int, default=5, help='Number of games to include in each bin')
parser.add_argument('--interaction_effect', default=True, help='Whether to include interaction effect in the regression')
parser.add_argument('--intercept', default=False, help='Whether to include intercept in the regressions')

args = parser.parse_args()

task = f'var{args.mean_var}_{args.reward_var}'
path = f"data/{task}/{args.engine}/meta-learning.csv"
reg_data = pd.read_csv(path)
reg_data = reg_data[reg_data['game'] < args.no_games].reset_index(drop=True)  #Get rid of excess games!



#Run functions
plot_regret_regression(reg_data, [0, -1+args.bin_size], args.intercept, args.engine) #? Figure 3C
plt.close()
plot_behaviour_regression(reg_data, [0, -1+args.bin_size], args.interaction_effect, args.intercept, args.engine) #? Figure 3E 
# plot_behaviour_regression(reg_data, [1+max(reg_data['game'])-bin_size, max(reg_data['game'])], interaction_effect, intercept, args.engine)

#? To plot individual predictors across games for visualisation purposes
# compare_start_to_end = False
# plot_regression_predictorcoef_across_games(reg_data, bin_size, compare_start_to_end, ['V/TU' 'Random exploration'], intercept, no_games, args.engine)
# plot_regression_predictorcoef_across_games(reg_data, bin_size, compare_start_to_end, ['V', 'Relative value'], intercept, no_games, args.engine)
# plot_regression_predictorcoef_across_games(reg_data, bin_size, compare_start_to_end, ['RU', 'Directed exploration'], intercept, no_games, args.engine)
# if intercept:
#     plot_regression_predictorcoef_across_games(reg_data, bin_size, compare_start_to_end, ['const', 'Intercept'], intercept, no_games, args.engine)




