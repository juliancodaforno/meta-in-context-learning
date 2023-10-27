import openai
import gym
import envs.bandits
import pandas as pd
import numpy as np
import os.path
from tqdm import tqdm
from torch.distributions import Binomial
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from utils import llm_prior, sample_alphabet

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--no_trials', type=int, default=10, help='Number of trials per game')
parser.add_argument('--no_games', type=int, default=5, help='Number of games per subject')
parser.add_argument('--mean_var', type=int, default=64, help='Variance of the prior distribution')
parser.add_argument('--reward_var', type=int, default=32, help='Variance of the reward distribution')
parser.add_argument('--no_queries', type=int, default=5, help='Number of queries sampled to approximate the prior distribution')
parser.add_argument('--engine', default='random', help='Engine to be used')
parser.add_argument('--no_runs', default=100, help='Number of runs to be made')
args = parser.parse_args()

task = f'var{args.mean_var}_{args.reward_var}'
act = llm_prior(args.engine) #TODO: Change this to your own LLM
data_path = f'./data/{task}/{args.engine}/meta-learning.csv' 
saving_path = f'data/{task}/{args.engine}/priors.csv'

start = 0
if os.path.exists(saving_path):
    start = int(np.max(pd.read_csv(saving_path)['subject']))
    print(f"Starting from {start}")
no_runs = args.no_runs - start

instructions = f"You are going to different casinos that own two slot machines."\
    " Choosing the same slot machine will not always give you the same points, but one slot machine is always better than the other."\
" Within a casino, your goal is to choose the slot machine that will give you the most points over the course of {args.no_trials} rounds. Each casino owns a different pair of machines. \n\n"

df = pd.read_csv(data_path)
df = df[df['game'] < args.no_games].reset_index(drop=True)
for subject in tqdm(range(start, start + no_runs)):
    prompt = instructions
    alphabet = 'ABCDEFGHJKLMNOPQRSTVWXYZ'
    answers = []
    query_idx = []
    game_idxs = []
    for g in range(args.no_games):
        arm1, arm2, alphabet = sample_alphabet(alphabet)
        asked_prompt = prompt + f"\n\nQ: You are now playing in casino {g+1} with machines {arm1} and {arm2}. You chose to play machine {arm1} in round 1. How rewarding do you expect machine {arm1} to be? Give the answer as a number.\nA: I expect machine {arm1} to deliver an average of approximately"
        for query in range(args.no_queries):
            answer = act(asked_prompt, arm1) 
            answers.append(answer)
            query_idx.append(query)
            game_idxs.append(g)
        prompt += f"\nYou have received the following points when playing in casino {g+1} with machines {arm1} and {arm2}:\n"
        for t in range(args.no_trials):
            arm_chosen = arm1 if df['action1'][subject * (args.no_games * args.no_trials) + g*args.no_trials + t] == True else arm2
            prompt += f"- Machine {arm_chosen} delivered {np.round(df['reward'][subject * (args.no_games * args.no_trials) + g*args.no_trials + t], 1)} points.\n"
        
    #Store dataframe in csv file
    answers = {'answers' : answers, 'query_idx': query_idx, 'game': game_idxs, 'subject': [subject] * len(answers)}
    reg_data = pd.DataFrame.from_dict(answers)
    if os.path.exists(saving_path):
        reg_data.to_csv(saving_path, mode='a', header=False, index=False) #? If already create
    else:
        reg_data.to_csv(saving_path, mode='a', index=False)


