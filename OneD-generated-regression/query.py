
import numpy as np 
import os
import pandas as pd
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import argparse
from utils import generate_function, store_data, preprocess, llm

def check_prior(data, prompt,  X, Y, machine_count):
    '''
    Args:
        data (dict): data to be stored
        prompt (str): prompt to be sent to LLM
        X (np.array): input
        Y (np.array): output
        machine_count (int): machine count
    Returns:
        data (dict): data to be stored
    '''
    prompt_concat = ''
    for i in range(X.shape[0]):
        test_prompt = f'x={X[i]}, y='           
        llm_output = act(prompt + prompt_concat + test_prompt,  temp=1.0)
        llm_output = preprocess(llm_output)
        prompt_concat += test_prompt + str(llm_output) + ';\n'
        data['x'].append(X[i])
        data['y'].append(Y[i])
        data['y_pred'].append(llm_output)
        data['machine'].append(machine_count)
        data['trial'].append(i + 1)
    return data


def query(no_trainings, no_machines,  path, gen_function_params, function_type='linear', x_bounds=(0, 100), no_priors=21):
    '''
    1. Query LLM for a given number of training examples for multiple (no_machines) 1D functions and store the data.
    2. Query LLM for prior distribution before training of a task (machine) and store the data.

    Args:
        no_trainings (int): number of training examples
        no_machines (int): number of machines
        path (str): path to store data
        gen_function_params (tuple): parameters of the function to be generated
        function_type (str): type of function to be generated
        x_bounds (tuple): bounds of the x values
        no_priors (int): number of priors
    '''
    data_train = {'x': [], 'y':[], 'machine': [], 'trial': []}
    training_low_bound, training_upper_bound = x_bounds
    data_train['y_pred'] = [] 
    data_prior = {'x':[], 'y':[], 'y_pred': [], 'machine': [], 'trial': []}
    if no_machines == 1:
        # In-context learning
        prompt = f"You observe a machine that produces an output y for a given input x:\n"
    else:
        # Meta-in-context learning
        prompt = f"You observe {no_machines} different machines that produce an output y for a given input x. Each machine implements a different function.\n"
    for machine_count in range(1, no_machines+1):
        prompt += "If no previous examples, sample y from your prior distribution. But do not give any non numerical answer, just give the output as a scalar followed by ';'! Even if you are unsure, try to predict y as well as possible."
        if no_machines == 1:
            prompt += "\n"
        else:
            prompt += f"\nMachine {machine_count}:\n"
        f = generate_function(function_type, gen_function_params)

        #Prior storing before training
        X_prior = np.linspace(1, 100, no_priors, dtype=int)
        Y_prior = f(X_prior)        
        data_prior = check_prior(data_prior, prompt,  X_prior, Y_prior,  machine_count)
        
        #Training
        Xs = np.random.randint(low=training_low_bound, high=training_upper_bound, size=no_trainings) #shuffled training inputs
        for count, x in enumerate(Xs):  
            x = np.round(x, decimals=1)
            f_x = np.round(f(x), decimals=1)
            data_train['x'].append(x)
            data_train['y'].append(f_x)
            data_train['machine'].append(machine_count)
            data_train['trial'].append(count + 1)
            prompt += f'x={x}, y='
            raw_output = act(prompt, temp=0) 
            llm_output = float(preprocess(raw_output))
            data_train['y_pred'].append(llm_output)
            prompt += f'{f_x};\n'

    #Store dataframe in csv file
    store_data( "./data/train/"+ path, data_train)
    store_data("./data/prior/" + path, data_prior)


#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--engines', nargs='+', default=['random'])
parser.add_argument('--max_tokens', type=int, default=5)
parser.add_argument('--only_training' ,  default='True')
parser.add_argument('--no_trainings' ,  default=5)
parser.add_argument('--no_tasks' ,  default=5)
parser.add_argument('--no_priors' ,  default=21)
parser.add_argument('--no_runs' ,  default=100, help='Number of runs to be made')
parser.add_argument('--function_type' ,  default='linear', choices=['linear', 'quadratic', 'periodic', 'step', 'exponential'], help='Type of function to be generated')
parser.add_argument('--params',nargs='+', default=[-2, 1, -100, 1, 0, 0, 1], help='Parameters for the function to be generated. They represent p1, p2, pb1, pb2, pc1, pc2, noise in the code'\
                    '. For linear, pc1, pc2 are not used so we put 0 by default') #[0.1, 0.01,0,1,100,1,1] for quadratic in Appendix A.1.2

args = parser.parse_args()  
engines = args.engines
for engine in engines:
    print(f'Engine :------------------------------------ {engine} ------------------------------------')
    act = llm(engine, args.max_tokens) #TODO: Change this to your own LLM
    no_trainings = int(args.no_trainings)
    no_machines = int(args.no_tasks)
    no_priors = int(args.no_priors)
    no_queries = int(args.no_runs)
    function_type= args.function_type
    fct_params = tuple([float(x) for x in args.params]); p1, p2, pb1, pb2, pc1, pc2, noise = fct_params
    
    # Path to store data
    data_path = f"{engine}/{no_machines}M_{no_trainings}Tr_{function_type}Fct/N{p1}_{p2}_b{pb1}_{pb2}noiseN0{noise}" #Training shuffled always and temparture=1 now!
    if function_type != 'linear':
        data_path = f"{engine}/{no_machines}M_{no_trainings}Tr_{function_type}Fct/N{p1}_{p2}_b{pb1}_{pb2}c{pc1}_{pc2}noiseN0{noise}"

    # Counts number of queries already made
    query_past =  pd.read_csv("./data/train/"+ data_path + '.csv')['subject'].max() if os.path.exists("./data/train/"+ data_path + '.csv') else 0

    #Run
    for _ in tqdm(range(no_queries - query_past)):  
        query(no_trainings, no_machines,  data_path, fct_params,  function_type=function_type, no_priors=no_priors)


