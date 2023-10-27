import gym
import envs.bandits
import pandas as pd
import numpy as np
import os.path
from tqdm import tqdm
from torch.distributions import Binomial
import argparse
from utils import llm, sample_alphabet

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--no_trials', type=int, default=10, help='Number of trials per game') 
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
parser.add_argument('--no_runs', type=int, default=100, help='Number of runs to be made')
parser.add_argument('--engine', default='random', help='Engine to be used')
parser.add_argument('--store', default=True, help='Whether to store data or not')

args = parser.parse_args()
engine = args.engine
storing = args.store
# key = 'not interacting with GPT3'; storing=False
data_path = f'./data/var{args.mean_var}_{args.reward_var}/{engine}/meta-learning.csv'
env = envs.bandits.Sams2armedbandit(2,  args.mean_var, args.reward_var, args.no_trials)
act = llm(engine) #TODO: Change this to your own LLM

start =  0  
if os.path.exists(data_path):
    start = np.max(pd.read_csv(data_path)['subject'])
    print(f"Starting from {start}")
no_runs = args.no_runs - start      


regression_data = {'V': [], 'RU': [], 'V/TU': [], 'action1': [], 'regret': [], 'reward':[], 'mu1':[], 'mu2': [], 'game':[]}
for subject in tqdm(range(start+1, start+no_runs+1)):
    instructions = f"You are going to different casinos that own two slot machines."\
     " Choosing the same slot machine will not always give you the same points, but one slot machine is always better than the other."\
    " Within a casino, your goal is to choose the slot machine that will give you the most points over the course of {no_trials} rounds. Each casino owns a different pair of machines. \n\n"
    history= ""
    rewards = 0
    alphabet = 'ABCDEFGHJKLMNOPQRSTVWXYZ'
    for game in range(args.no_games):
        arm1, arm2, alphabet = sample_alphabet(alphabet)
        env.reset(action_letters=[arm1, arm2])
        for trial in range(env._no_trials):
            #randomly change the order of the arms to avoid order effects. 
            if Binomial(1, 0.5).sample() == 1:  
                arm1 , arm2 = arm2, arm1
            if trial == 0 and game != 0:
                history += f"\n\nIn total, you have received {np.round(rewards.numpy(), 1)} points when playing in casino {game} with machine {prev_arm1} and machine {prev_arm2}."
                rewards = 0
            prompt = instructions + history + f"\n\nQ: We are now performing round {trial +1} in casino {game + 1}. Which machine do you choose between machine {arm1} and machine {arm2}?\nA: Machine"
            action = act(prompt, action_letters=[arm1, arm2]) 
            reward = env.step(action)
            rewards += reward

            if trial == 0:
                history +=  f"\n\nYou have received the following money when playing in casino {game + 1}:\n"
            history += f"- Machine {action} delivered {np.round(reward, 1)} points.\n"
        prev_arm1, prev_arm2 = arm1, arm2  

        #Store data at the end of each game in dataframe:
        regression_data['V']     = env.V
        regression_data['RU'] = env.RU
        regression_data['V/TU'] = list(np.array(env.V)/np.array(env.TU))
        regression_data['action1'] = env.action1
        regression_data['regret'] = env.Regret_list
        regression_data['reward'] = env.reward
        regression_data['mu1'] = [env.mean1.numpy() for _ in range(env._no_trials)]
        regression_data['mu2'] = [env.mean2.numpy() for _ in range(env._no_trials)]
        regression_data['game'] = [game for _ in range(env._no_trials)]
        regression_data['trial'] = [tr for tr in range(env._no_trials)]
        regression_data['subject'] = [subject for _ in range(env._no_trials)]

        #Store dataframe in csv file
        if storing == True:
            reg_data = pd.DataFrame.from_dict(regression_data)
            if os.path.exists(data_path):
                reg_data.to_csv(data_path, mode='a', header=False, index=False) #? If already create
            else:
                reg_data.to_csv(data_path, mode='a', index=False)



