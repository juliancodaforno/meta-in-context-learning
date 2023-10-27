import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import statsmodels.api as smapi
import seaborn as sns
import statsmodels.tools.tools as sm
import torch
from utils import ExactInference, averaged_generated_function
import argparse

plt.rcParams.update({
    "text.usetex": True
})

def plot_priors(data, no_machines, gen_f, llm_name='GPT-3'):
    """
    Plot the priors for each machine. Save the figure in the plots folder.
    Args:
        data (pandas dataframe): dataframe containing the data.
        no_machines (int): number of machines.
        gen_f (tuple): tuple containing the generated function.
        llm_name (str): name of the LLM.
    """
    plt.rcParams["figure.figsize"] = (5.2,1.6)
    y_mean_gen, y_std_gen = gen_f
    no_subjects = max(data['subject'])
    fig, axis = plt.subplots(1, no_machines)
    if no_machines == 1:
        # in-context learning
        axis = np.array([axis])
    for machine in range(1, 1+no_machines):
        data_machine = data[data['machine'] == machine]
        #Plot mean generated function
        axis[machine-1].plot(np.linspace(0, 100, len(y_mean_gen)), y_mean_gen, color='C1', label='Task distribution' if machine == 1 else None, alpha=0.8)
        axis[machine-1].fill_between(np.linspace(0, 100, len(y_mean_gen)), y_mean_gen - 2*y_std_gen, y_mean_gen + 2*y_std_gen, alpha=0.1, color='C1')

        #Plot priors
        X = data_machine[data_machine['subject'] == 1]['x']
        Y = np.zeros((len(X), no_subjects))

        for subject in range(1, 1+ no_subjects):
            Y[:, subject-1] = data_machine[data_machine['subject'] == subject]['y_pred']
        
        #Get rid of outliers. If a prior is more than two std away from the mean of the priors, remove the entire prior. Also remove if the prior is more than 1e6
        std_priors = np.std(Y, axis=1).reshape(-1, 1)
        mean_priors = np.mean(Y, axis=1).reshape(-1, 1)
        mask = np.any([np.abs(Y - mean_priors) > 2 * std_priors, np.abs(Y) > 1e6], axis=0)

        Y[mask] = np.nan
        mask[mask== np.inf] = np.nan
        Y = Y[:, ~np.isnan(Y).any(axis=0)]

        #Plot priors mean and confidence intervals
        axis[machine-1].plot(X, np.mean(Y, axis=1), color='C0', label=f'{llm_name} priors' if machine == 1 else None, alpha=0.8) 
        axis[machine-1].fill_between(X, np.mean(Y, axis=1) - 1.96*np.std(Y, axis=1)/np.sqrt(no_subjects), np.mean(Y, axis=1) + 1.96*np.std(Y, axis=1)/np.sqrt(no_subjects), alpha=0.1, color='C0')

    for machine_no, ax in enumerate(axis.flat):
        ax.set(xlabel='x', ylabel='f(x)') #Labelling all plots
        # ax.set_ylim(ymin, ymax) #TODO: set y limits 
        #Make figure axes labels fontsize bigger
        ax.set_title(f'Task {machine_no+1}', fontsize=10)
        #Only keep the yticks on the leftmost plots
        if machine_no  != 0:
            ax.set_yticks([])
            ax.set_ylabel('')

        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout(pad=0)
        
        #Make the labels text font lighter
        fig.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', frameon=False, ncol=2, prop={'size': 8})
    sns.despine()
    

    #Create directories if they don't exist, if exists, don't do anything
    prevdir = data_path.replace('/'+data_path.split('/')[-1], '')
    prevprevdir = prevdir.replace('/'+prevdir.split('/')[-1], '')
    os.makedirs(f"./plots/{prevprevdir}", exist_ok=True)
    os.makedirs(f"./plots/{prevdir}",   exist_ok=True)
    os.makedirs(f"./plots/{data_path}", exist_ok=True)

    #Save figure
    plt.savefig(f"./plots/{data_path}/priors.png", bbox_inches='tight', dpi=300)

def plot_regression(df, interaction_effect=False, intercept=False, with_similarity=False):
    """
    Plot the MSE for each machine.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing the data.
    no_machines : int
        Number of machines.
    """
    plt.rcParams["figure.figsize"] = (1.3,1.6)
    Y = np.abs(df['y'] - df['y_pred'])
    
    if with_similarity:
        # Add similarity score to predictors
        from scipy.stats import linregress
        # Get the linear regression coefficients (a, b from a*x + b) for each generated task per subject and create extra columns a,b in the dataframe
        df['a'] = [np.nan]*df.shape[0]
        df['b'] = [np.nan]*df.shape[0]
        df['Task\nsimilarity'] = [np.nan]*df.shape[0]
        for task in range(1, 1+no_machines):
            for subject in df['subject'].unique():
                task_data = df[(df['machine'] == task) & (df['subject'] == subject)]
                slope, interc, r_value, p_value, std_err = linregress(task_data['trial'], task_data['y'])
                df.loc[(df['machine'] == task) & (df['subject'] == subject), 'a'] = slope
                df.loc[(df['machine'] == task) & (df['subject'] == subject), 'b'] = interc

                # Calculate similarity score for current task and subject
                if task == 1:
                    # Default values for the first task is a function y = x
                    a_default = 1
                    b_default = 0
                    similarity_score = np.sqrt((slope - a_default)**2 + (interc- b_default)**2) 
                else:
                    prev_tasks = range(1, task)
                    prev_vectors = [(df[(df['machine'] == prev_task) & (df['subject'] == subject)]['a'].iloc[0], df[(df['machine'] == prev_task) & (df['subject'] == subject)]['b'].iloc[0]) for prev_task in prev_tasks]
                    similarity_score = np.mean([np.sqrt((slope - prev_vector[0])**2 + (interc - prev_vector[1])**2) for prev_vector in prev_vectors])

                # Add similarity score to X
                df.loc[(df['machine'] == task) & (df['subject'] == subject), 'Task\nsimilarity'] = similarity_score

    df = df.rename(columns={'machine': 'Task', 'trial': 'Trial'})
    predictors = ['Trial', 'Task']  if not with_similarity else ['Trial', 'Task', 'Task\nsimilarity']
    X = df
    for col in df.columns:
        if col not in predictors:
            X = X.drop(columns=col)
        else:
            X[col] = (X[col] - X[col].mean())/X[col].std()

    #add other predictors
    if interaction_effect:
        X[r'Task$\times$tTial'] = X['Task'] *  X['Trial']
        predictors += [r'Task$\times$Trial']
    if intercept:
        X = sm.add_constant(X)
        predictors += ['const']
                
    #Fit the model
    model = smapi.OLS(Y, X).fit()
    # print(model.summary())
    for idx in predictors:
        print(f'Beta_{idx}: {model.params[idx]} +- {model.conf_int()[1][idx] - model.params[idx]}')
        plt.bar(idx, model.params[idx], yerr=model.conf_int()[1][idx] - model.params[idx], color='C3', alpha=0.6)    

    #Plot
    plt.ylabel("Regression\n coefficients")
    sns.despine()

    #Add space at the bottom of the figure
    plt.xticks( fontsize=7)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(bottom=0.2)

    #Create directories if they don't exist, if exists, don't do anything
    prevdir = data_path.replace('/'+data_path.split('/')[-1], '')
    prevprevdir = prevdir.replace('/'+prevdir.split('/')[-1], '')
    os.makedirs(f"./plots/{prevprevdir}", exist_ok=True)
    os.makedirs(f"./plots/{prevdir}",   exist_ok=True)
    os.makedirs(f"./plots/{data_path}", exist_ok=True)

    #Save figure
    plt.savefig(f"./plots/{data_path}/regression{'_with_similarity' if with_similarity else ''}.png", dpi=300)

def plot_loss_across_games(data, no_machines):
    """
    Plot the loss across games for each machine. Save the figure in the plots folder.
    Args:
        data (pandas dataframe): dataframe containing the data.
        no_machines (int): number of machines.
    """
    x_means = []
    x_cis = []
    for game in range(1, 1+no_machines):
        abs_losses = np.abs(data[data['machine'] == game]['y_pred'] - data[data['machine'] == game]['y'])
        x_means.append(np.mean(abs_losses))
        x_cis.append(1.96*np.std(abs_losses)/np.sqrt(len(abs_losses)))
    x_means = np.array(x_means); x_cis = np.array(x_cis)
    #Plot
    plt.rcParams["figure.figsize"] = (1.3,1.6)
    plt.plot(range(1, no_machines+1), x_means, color='C0', alpha=0.8)
    plt.fill_between(range(1, no_machines+1), x_means-x_cis, x_means+x_cis, color='C0', alpha=0.1)
    plt.xlabel('Tasks')
    plt.ylabel('MSE')
    plt.xticks(range(1, no_machines+1))
    sns.despine()
    plt.tight_layout(pad=0.05)
    plt.legend(frameon=False,bbox_to_anchor=(0,1.02,1,0.2), loc='lower center')

    #Create directories if they don't exist, if exists, don't do anything
    prevdir = data_path.replace('/'+data_path.split('/')[-1], '')
    prevprevdir = prevdir.replace('/'+prevdir.split('/')[-1], '')
    os.makedirs(f"./plots/{prevprevdir}", exist_ok=True)
    os.makedirs(f"./plots/{prevdir}",   exist_ok=True)
    os.makedirs(f"./plots/{data_path}", exist_ok=True)

    #Save figure
    plt.savefig(f"./plots/{data_path}/loss_across_games.png", bbox_inches='tight')

def plot_loss_across_trials(data, no_machines, no_trials, bayesian_preds: list, params: tuple, llm_name: str='GPT-3'):
    """
    Plot the loss across trials for each machine. Save the figure in the plots folder.
    Args:
        data (pandas dataframe): dataframe containing the data.
        no_machines (int): number of machines.
        no_trials (int): number of trials.
        bayesian_preds (list): list of booleans indicating whether to plot the bayesian predictions or not.
        params (tuple): tuple containing the parameters of the generative function.
        llm_name (str): name of the LLM.
    """
    #Simulate bayesian linear regressions on the data 
    plt.rcParams["figure.figsize"] = (2.6,1.6)
    if False not in bayesian_preds: #Else boolean which is False
       #ground truth
        pa1, pa2, pb1, pb2, noise_var = params
        bl_errs = np.zeros((no_machines*max(data['subject']), no_trials))
        total_no_games = no_machines*max(data['subject'])
        bias = {'use': True, 'mean_prior': torch.Tensor([pb1]), 'logvar_prior': torch.log(torch.Tensor([pb2]))}   
        blr_model = ExactInference(num_features=1, pred_logvar=torch.log(torch.Tensor([noise_var])), prior_mean = torch.Tensor([pa1]), prior_logvar=torch.log(torch.Tensor([pa2])),  bias=bias) 
        for game in range(total_no_games):
            blr_model.reset()
            for trial in range(no_trials):
                bl_errs[game, trial] = np.abs(blr_model.predict([data['x'][game*no_trials + trial]]) - data['y'][game*no_trials + trial])
                blr_model.fit(torch.Tensor([data['x'][game*no_trials + trial]]), torch.Tensor([data['y'][game*no_trials + trial]]))
        bl_gt_errs_mean = np.mean(bl_errs, axis=0)
        bl_gt_errs_ci = 1.96*np.std(bl_errs, axis=0)/np.sqrt(total_no_games)
       
        #Random prior
        bl_errs = np.zeros((no_machines*max(data['subject']), no_trials))
        total_no_games = no_machines*max(data['subject'])
        bias = {'use': True, 'mean_prior': torch.Tensor([0]), 'logvar_prior': torch.log(torch.Tensor([1]))}   #mean 0 and variance 1 by default
        blr_model = ExactInference(num_features=1, pred_logvar=torch.log(torch.Tensor([1])), bias=bias) 
        for game in range(total_no_games):
            blr_model.reset()
            for trial in range(no_trials):
                bl_errs[game, trial] = np.abs(blr_model.predict([data['x'][game*no_trials + trial]]) - data['y'][game*no_trials + trial])
                blr_model.fit(torch.Tensor([data['x'][game*no_trials + trial]]), torch.Tensor([data['y'][game*no_trials + trial]]))
        bl_errs_mean = np.mean(bl_errs, axis=0)
        bl_errs_ci = 1.96*np.std(bl_errs, axis=0)/np.sqrt(total_no_games)

    data_start = data[data['machine'] == 1]
    data_end = data[data['machine'] == no_machines]
    #Get MSE
    loss_start_means = []
    loss_end_means = []
    loss_start_ci = []
    loss_end_ci = []
    for trial in range(1, 1+no_trials):
        loss_start_means.append(np.mean(np.abs(data_start[data_start['trial']==trial]['y_pred'] - data_start[data_start['trial'] == trial]['y'])))
        loss_end_means.append(np.mean(np.abs(data_end[data_end['trial']==trial]['y_pred'] - data_end[data_end['trial'] == trial]['y'])))

        loss_start_ci.append(1.96* np.std(np.abs(data_start[data_start['trial']==trial]['y_pred'] - data_start[data_start['trial']==trial]['y']))/np.sqrt(len(data_start[data_start['trial']==trial])))  
        loss_end_ci.append(1.96* np.std(np.abs(data_end[data_end['trial']==trial]['y_pred'] - data_end[data_end['trial']==trial]['y']))/np.sqrt(len(data_end[data_end['trial']==trial])))  
    #Plot
    plt.plot(range(1, 1+no_trials), loss_start_means, label=f'{llm_name}', color = 'C0', alpha=0.8)
    plt.fill_between(range(1, 1+no_trials), np.array(loss_start_means) - np.array(loss_start_ci), np.array(loss_start_means) + np.array(loss_start_ci), alpha=0.1, color = 'C0')
    plt.plot(range(1, 1+no_trials), loss_end_means, label=f'{llm_name} after {no_machines} tasks', linestyle='--', color = 'C0', alpha=0.8)
    plt.fill_between(range(1, 1+no_trials), np.array(loss_end_means) - np.array(loss_end_ci), np.array(loss_end_means) + np.array(loss_end_ci), alpha=0.1, color = 'C0')
    if 'random' in bayesian_preds:
        plt.plot(range(1, 1+no_trials), bl_errs_mean, label='BLR (default)', color='C4', alpha=0.8)
        plt.fill_between(range(1, 1+no_trials), bl_errs_mean - bl_errs_ci, bl_errs_mean + bl_errs_ci, alpha=0.1, color = 'C4')
    if 'ground truth' in bayesian_preds:
        plt.plot(range(1, 1+no_trials), bl_gt_errs_mean, label='BLR (oracle)', linestyle='--', color='C4', alpha=0.8)
        plt.fill_between(range(1, 1+no_trials), bl_gt_errs_mean - bl_gt_errs_ci, bl_gt_errs_mean + bl_gt_errs_ci, alpha=0.1,  color = 'C4')
    plt.xlabel('Trial')
    plt.ylabel('MSE')
    plt.xticks(range(1, 1+no_trials))
    sns.despine()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(-0.1,1.02,1,0.2),  loc='lower center', ncol=2, frameon=False, prop={'size': 8}, columnspacing=0.7)

    #Create directories if they don't exist, if exists, don't do anything
    prevdir = data_path.replace('/'+data_path.split('/')[-1], '')
    prevprevdir = prevdir.replace('/'+prevdir.split('/')[-1], '')
    os.makedirs(f"./plots/{prevprevdir}", exist_ok=True)
    os.makedirs(f"./plots/{prevdir}",   exist_ok=True)
    os.makedirs(f"./plots/{data_path}", exist_ok=True)

    #Save figure
    plt.savefig(f"./plots/{data_path}/loss_across_trials.png", bbox_inches='tight')


#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--engines', nargs='+', default=['random'])
parser.add_argument('--no_trainings', type=int, default=5)
parser.add_argument('--no_tasks', type=int, default=5)
parser.add_argument('--function_type' ,  default='linear', choices=['linear', 'quadratic', 'periodic', 'step', 'exponential'], help='Type of function to be generated')
parser.add_argument('--params',nargs='+', default=[-2, 1, -100, 1, 0, 0, 1], help='Parameters for the function to be generated. They represent p1, p2, pb1, pb2, pc1, pc2, noise in the code'\
                    '. For linear, pc1, pc2 are n\a so we put 0 by default') #[0.1, 0.01,0,1,100,1,1] for quadratic in Appendix A.1.2
parser.add_argument('--bayesian_preds', nargs='+', default=['ground truth', 'random'], choices=[['ground truth', 'random'], ['ground truth'], ['random'], [False]], 
                    help='Whether to plot the bayesian predictions or not. Ground truth means the bayesian predictions are made with the true parameters of the generative function.'\
                         ' Random means the bayesian predictions are made with a random prior. False means no bayesian predictions are plotted.') #Only used in plot_loss_across_trials()
args = parser.parse_args()
no_trainings = args.no_trainings
no_machines = args.no_tasks
function_type= args.function_type
fct_params = tuple([float(x) for x in args.params]); p1, p2, pb1, pb2, pc1, pc2, noise = fct_params
bayesian_preds = args.bayesian_preds 
if function_type != 'linear':
    bayesian_preds = [False] #Can't do bayesian predictions in closed-form for non-linear functions

for engine in args.engines:
    gen_f = averaged_generated_function(fct_params, iters=1000, fct_type=function_type) 

    if function_type == 'linear':
        data_path = f"{engine}/{no_machines}M_{no_trainings}Tr_{function_type}Fct/N{p1}_{p2}_b{pb1}_{pb2}noiseN0{noise}" 
    else : 
        data_path =  f"{engine}/{no_machines}M_{no_trainings}Tr_{function_type}Fct/N{p1}_{p2}_b{pb1}_{pb2}c{pc1}_{pc2}noiseN0{noise}" 
    
    #Load data
    df_train = pd.read_csv(f"data/train/{data_path}.csv") 
    df_prior = pd.read_csv(f"data/prior/{data_path}.csv")

    # Plot all figures. 
    plot_loss_across_trials(df_train, no_machines, no_trainings, bayesian_preds, params=(p1, p2, pb1, pb2, noise), llm_name=engine)
    plt.close()
    plot_loss_across_games(df_train, no_machines)
    plt.close()
    plot_priors(df_prior, no_machines,  gen_f, engine)
    plt.close()
    plot_regression(df_train)
    plt.close()


