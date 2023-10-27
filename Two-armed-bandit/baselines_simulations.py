import gym
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
import torch
import human_exp.envs.bandits
from scipy.stats import norm
import os
from tqdm import tqdm
import argparse


class ThompsonSampling:
    def act(self, priors, obs=None, hx=None, zeta=None):
        p_a1 = norm.cdf((priors[0, 0] - priors[1, 0])/np.sqrt(priors[0, 1] + priors[1,1]))
        return  p_a1

class UCB:
    def __init__(self, gamma=1.0, lamb=1.0):
        self.gamma = gamma
        self.lamb = lamb

    def act(self, priors, obs=None, hx=None, zeta=None):
        p_a1 = norm.cdf((priors[0, 0] - priors[1, 0] + self.gamma * (math.sqrt(priors[0, 1]) - math.sqrt(priors[1, 1])))/ self.lamb)
        return p_a1

class greedy:
    def __init__(self, alpha=1.0):
        self.alpha = alpha


    def act(self, priors, obs=None, hx=None, zeta=None):
        p_a1 = 1 if priors[0, 0] - priors[1, 0] > 0 else 0
        return p_a1

def simulation(algo, name, mean_var, reward_var, games):
    '''
    Args:

    algo: algorithm to be used for simulation
    name: name of the algorithm
    mean_var: variance of the mean of the two arms
    reward_var: variance of the reward
    games: number of games to be played

    Returns:

    Stores the data in a csv file
    '''

    no_trials = 10
    data = {'V': [], 'RU': [], 'V/TU': [], 'action1': [], 'mu1': [], 'mu2': [], 'trial': [], 'game': [], 'regret':[]}
    sim  = gym.make('Sams2armedbandit_KM-v0', mean_var=mean_var, reward_var = reward_var, no_trials=no_trials)
    priors = np.zeros((2, 2))
    priors[:, 1] = sim._mean_std ** 2


    for game in tqdm(range(games)):
        sim.reset()
        for trial in range(no_trials):
            priors[0, :] = np.array([sim._exp_rew1, sim._exp_var1])
            priors[1, :] = np.array([sim._exp_rew2, sim._exp_var2])
            action = algo.act(priors)
            action = np.random.binomial(1, 1 - action)  #sampling from action2 (1-action) because it refers to action = 1
            _, _ = sim.step(action)

        data['V']     += sim.V
        data['RU'] += sim.RU
        data['V/TU'] += list(np.array(sim.V).reshape(no_trials)/np.array(sim.TU).reshape(no_trials))
        data['action1'] += sim.action1
        data['mu1'] += [sim.mean1.numpy() for _ in range(no_trials)]
        data['mu2'] += [sim.mean2.numpy() for _ in range(no_trials)]
        data['trial'] += list(range(10))
        data['game'] += list(np.ones(10)* game)
        data['regret'] += sim.Regret_list

    #Save data in csv
    data = pd.DataFrame.from_dict(data)
    data_path = f'./data/{task}'
    os.makedirs(data_path, exist_ok=True) #creates directory if not created, it it exists, it does nothing
    data.to_csv(data_path + f'/{name}_data.csv')
    
#Hyperparameters
mean_var = 20
reward_var = 10
no_games = 1000
task = f"var{mean_var}_{reward_var}"
TS = ThompsonSampling()
# greedy = UCB(gamma=0) #does not work well for some reason
greedy_algo = greedy()
UCB = UCB()

parser = argparse.ArgumentParser()
parser.add_argument("--UCB", type=bool, required=True)
parser.add_argument("--greedy", type=bool, required=True)
parser.add_argument("--TS", type=bool, required=True)
args = parser.parse_args()

if args.UCB:
    simulation(UCB, 'UCB', mean_var, reward_var, no_games)
if args.greedy:
    simulation(greedy_algo, 'greedy', mean_var, reward_var, no_games)
if args.TS:
    simulation(TS, 'TS', mean_var, reward_var, no_games)






