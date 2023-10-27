import math
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import torch.nn.functional as F
import torch
import pandas as pd

class Sams2armedbandit:
    '''
    Two-armed bandit environment with Gaussian rewards and Gaussian priors.
    The agent can choose between two actions: 0 and 1.
    The agent has a prior belief over the mean of the reward distribution of each action. 
    The agent updates its belief using a Kalman filter.

    Args:
        num_actions (int): number of actions
        mean_var (float): variance of the prior distribution
        reward_var (float): variance of the reward distribution
        no_trials (int): number of trials per game
        batch_size (int): batch size
        decimals (int): number of decimals to round the rewards to.
        '''
    def __init__(self, num_actions,  mean_var, reward_var, no_trials, batch_size=1, decimals=1):
        self._num_actions = num_actions
        self._no_trials = no_trials
        self._batch_size = batch_size
        self._device = None
        self._reward_std = np.sqrt(reward_var)
        self._mean_std = np.sqrt(mean_var)
        self._decimals = decimals

        #init
        self.R_chosen_list = []

    def reset(self,  action_letters=None, priors="default"):
        '''
        Resets the environment and samples new reward distributions and priors.        
        Args:
            action_letters (list): letters corresponding to each action
            priors (str): whether to use default priors or not
        '''
        self.t = 0
        self.action_letters = action_letters
        self.mean1 = Normal(0, self._mean_std).sample()
        self.mean2 = Normal(0, self._mean_std).sample()
        self.R1s = Normal(self.mean1 * np.ones(self._no_trials), self._reward_std).sample()
        self.R2s = Normal(self.mean2 * np.ones(self._no_trials), self._reward_std).sample()
        self._rewards = torch.stack((self.R1s, self.R2s), dim=1)

        #Priors for Kalman fiter belief updating
        if priors == "default":
            self._exp_rew1 = self._exp_rew2 = 0
            self._exp_var1 = self._exp_var2 = self._mean_std ** 2
        else:
            self._exp_rew1 = self._exp_rew2 = priors[0]
            self._exp_var1 = self._exp_var2 = priors[1]

        #Data gathering for probit regression
        self.V = []
        self.RU = []
        self.TU = []
        self.action1 = []
        self.Regret_list = []
        self.reward = []

    def step(self, action):
        '''
        Performs one step in the environment.
        Updates the belief over the mean of the reward distribution of each action.
        Args:
            action (int): action chosen by the agent

        Returns:
            R_chosen (float): reward of the chosen action
        '''
        if action== self.action_letters[0]:
            action = 0
            self.action1.append(True)
        elif action==self.action_letters[1]:
            action = 1
            self.action1.append(False)
        else:
            raise Exception(f"GPT3 has given me action {action} instead of {self.action_letters}")
        
        # Update V, RU, TU
        self.V.append(self._exp_rew1 - self._exp_rew2)
        self.RU.append(np.sqrt(self._exp_var1) - np.sqrt(self._exp_var2))
        self.TU.append(np.sqrt(self._exp_var1 + self._exp_var2))
        # Update reward, regret, R_chosen and time step
        R_chosen = self._rewards[self.t][action]
        self.reward.append(R_chosen.numpy())
        Regret = torch.max(self.mean1, self.mean2) - self.mean1 if action == 0 else torch.max(self.mean1, self.mean2) - self.mean2
        self.R_chosen_list.append(R_chosen)
        self.Regret_list.append(Regret.numpy())
        self.t += 1

        #Belief updating based on Gaussian assumptions using Kalman filter
        if action==0:
            lr = self._exp_var1 / (self._exp_var1 + self._reward_std ** 2)
            self._exp_var1 -= lr * self._exp_var1 
            self._exp_rew1 += lr * (R_chosen.numpy() - self._exp_rew1 )
        elif action==1:
            lr = self._exp_var2 / (self._exp_var2 + self._reward_std ** 2)
            self._exp_var2 -= lr * self._exp_var2
            self._exp_rew2 += lr * (R_chosen.numpy() - self._exp_rew2 )

        return R_chosen