import pyreadr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import math
import random
import time
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def load_data(name, num_features=10, replace_strings=True, normalize_data=True):
    result = pyreadr.read_r('data_srm/' + name + '.rdata')
    df = result[name].iloc[:, ::-1] # reverse order of columns

    #? The hashed code below was for giving the feature names but no used atm.
    # if name == 'rent':
    #     df = df.rename(columns={
    #         'rentm': 'rent per square-meter in euros',
    #         'size': 'size',
    #         'rooms': 'number of rooms',
    #         'year': 'year of construction',
    #         'good': 'located at a good address',
    #         'warm': 'has warm water',
    #         'central': 'has central heating',
    #     })
    #     df = df.drop(columns=['tiles', 'kitchen', 'bathextra', 'best'])

    #if columns is categorical, check if they are binary and if yes replace with 0 and 1
    if replace_strings:
        for column in df.columns:
            if df[column].dtype == 'category':
                if len(df[column].unique()) == 2: # >2 would need to be one-hot encoded but would create too many columns
                    df.replace([df[column].unique()[0], df[column].unique()[1]], [0, 1], inplace=True)
                    #make the column dtype to int because sometimes keep category dtype and therefore not filtered out afterwards
                    df[column] = df[column].astype(int)


    #only keep columns with numerical values
    df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'int', 'float', 'bool'])

    #Delete all rows with NaN values
    df = df.dropna().reset_index(drop=True)

    if normalize_data: #? Actually squashing btw -1 and 1, not really normalizing, but same idea 
        #Normalize data, or squash actually between -1 and 1
        features = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(df[df.columns[:-1]]).transform(df[df.columns[:-1]])
        targets = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(df[df.columns[-1:]]).transform(df[df.columns[-1:]])
        num_features = min(num_features, features.shape[1])
        features = SelectKBest(f_regression, k=num_features).fit_transform(features, targets.reshape(-1))
        df = pd.DataFrame(np.concatenate((features, targets), axis=1), columns=['x' + str(i) for i in range(num_features)] + ["y"]).round(2)

    return df

def select_subset(df, num_points):
    df_subset = df.sample(n=num_points)
    X = df_subset[df.columns[:-1]].to_numpy()
    y = df_subset[df.columns[-1]].to_numpy()

    return X, y, df_subset

def get_instructions(name):
    if name == 'generic':
        prompt = "You observe an input vector x and have to predict the corresponding output y as accurately as possible.\n\n"
    if name == 'rent':
        prompt = "Please try to predict the rent of an apartment in Munich in 1999 as accurately as possible: \n\n"
    if name == 'meta':
         prompt = "You observe an input vector x and have to predict the corresponding output y as accurately as possible. You are given 5 different tasks:\n"
    return prompt

def act_micl(text, engine, temp=0.0):
    if engine == 'random':
        return Normal(1, 1).sample().numpy()
    #GPT-3/4
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    for loop in range(10):
        try: 
            if engine == 'gpt-4':
                openai.api_key = key
                messages=[{"role": "user", "content": text}]
                raw_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens=10,
                    temperature=temp,
                    messages = messages,
                    stop=["\n", ";"]
                ).choices[0].message.content
                response = raw_response.replace(' ', '').replace(',', '').rstrip('\n').rstrip(';')
            else:
                openai.api_key = key
                raw_response = openai.Completion.create(
                    engine=engine,
                    prompt=text,
                    temperature=temp,
                    max_tokens=10,
                    stop=["\n", ";"],
                ).choices[0].text.strip()
                response = raw_response.replace(' ', '').replace(',', '').rstrip('\n').rstrip(';')
            break
        except:
            print(f'Loop {loop} failed')
            time.sleep(2**loop)
    try:
        return float(response)
    except:
        print(f'the following response was not compatible to float(): {response}') 
        import ipdb; ipdb.set_trace()
        return response
    
def act_icl(text, engine, temp=0.0, sample_prior=False):
    if engine == 'random':
        return Normal(1, 1).sample().numpy()
    
        
    #GPT-3/4
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    messages=[{"role": "user", "content": text}]  #This is only for GPT-4
    for loop in range(10):
        try: 
            if engine == 'gpt-4':
                openai.api_key = key
                if (sample_prior == True) and (loop ==0):
                    messages  = [{"role": "system", "content": "If no previous examples, sample y from your prior distribution. But do not give any non numerical answer! Even if you are unsure, try to predict y as well as possible."}] + messages
                raw_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens=10,
                    temperature=temp,
                    messages = messages,
                    stop=["\n", ";"]
                ).choices[0].message.content
                response = raw_response.replace(' ', '').replace(',', '').rstrip('\n').rstrip(';').replace("'", '')
            else:
                openai.api_key = key
                raw_response = openai.Completion.create(
                    engine=engine,
                    prompt=text,
                    temperature=temp,
                    max_tokens=10,
                    stop=["\n", ";"],
                ).choices[0].text.strip()
                response = raw_response.replace(' ', '').replace(',', '').replace('[', '').replace(']', '').rstrip('\n').rstrip(';')
            #If the string is only letters then continue and add to message
            if response[:5].isalpha():
                if loop == 9:
                    print(text)
                    import ipdb; ipdb.set_trace()
                    return 1
                temp = 1
                print(f'Loop {loop+1} running because response "{response}" was only letters')
                messages=[{"role": "system", "content": "Do not give any non numerical answer for y! Even if you are unsure, try to predict y as well as possible between -1 and 1."}, {"role": "user", "content": text}] 
            else:
                break 
        except:
            print(f'Loop {loop} failed')
            time.sleep(2**loop)
            continue
    #if there is more than two '.' in the response, then remove the second '.' and everything after
    if response.count('.') > 1:
        response = response[:response.find('.', response.find('.')+1)]
    if response.count('-') > 1:
        response = response[:response.find('-', response.find('-')+1)]
    #If there is - after a . then remove the - and everything after, e.g. 0.5-1000 -> 0.5
    if (response.find('-') != -1 ) and (response.find('.') != -1):
        if response.find('.') < response.find('-'):
            response = response[:response.find('-')]
    try:
        return float(response)
    except:
        print(f'the following response was not compatible to float(): {response}') 
        import ipdb; ipdb.set_trace()
        return response
    
class ExactInference(nn.Module):
    def __init__(self, num_features, pred_logvar, prior_mean='False', prior_logvar='False', empirical_bayes=False, bias={'use': False}):
        """
        Exact inference for Bayesian linear regression
        Args:
            num_features: number of features
            pred_logvar: log variance of the predictive distribution
            prior_mean: mean of the prior distribution
            prior_logvar: log variance of the prior distribution
            empirical_bayes: whether to use empirical Bayes or not
            bias: whether to use bias or not
        
        Attributes:
            mean: mean of the posterior distribution
            covariance: covariance of the posterior distribution
        """
        super(ExactInference, self).__init__()

        #By default, the priors are set to 0
        if prior_mean == 'False':                    
            prior_mean = torch.zeros(num_features)

        if prior_logvar == 'False':                 
            prior_logvar = torch.zeros(num_features)

        self.use_bias = bias['use']
        if self.use_bias:
            prior_mean = torch.cat((prior_mean, bias['mean_prior'] ))
            prior_logvar = torch.cat((prior_logvar, bias['logvar_prior']))

        
        self.pred_logvar = nn.Parameter(pred_logvar, requires_grad=empirical_bayes) 
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=empirical_bayes)
        self.prior_logvar = nn.Parameter(prior_logvar, requires_grad=empirical_bayes)


        self.reset()

    def reset(self):
        self.mean = self.prior_mean.clone()
        self.covariance = torch.diag(self.prior_logvar.exp())  #Assuming isotropic prior variance

    def fit(self, inputs, target):
        # inputs: (1, self.num_features)


        #If inputs and targets are numpy arrays, convert them to tensors
        if type(inputs) == np.ndarray:
            inputs = torch.Tensor(inputs)
        if type(target) == np.ndarray:
            target = torch.Tensor(target)

        if self.use_bias:
            inputs = torch.cat((inputs.reshape(-1, 1), torch.ones(1, 1)), dim=-1)

        # Equations below from Bishop 3.3.1
        inv_covariance = torch.inverse(self.covariance) + (1/self.pred_logvar.exp()) * inputs.t() @ inputs
        self. mean = torch.inverse(inv_covariance) @ ( torch.inverse(self.covariance) @ self.mean + (1/self.pred_logvar.exp()) * inputs.t() @ target)
        self.covariance = torch.inverse(inv_covariance)

    def predict(self, inputs):
        #If inputs are numpy arrays, convert them to tensors
        if type(inputs) == np.ndarray:
            inputs = torch.Tensor(inputs)

        if self.use_bias:
            inputs = torch.cat((torch.Tensor(inputs), torch.ones(1)))
        return Normal(inputs @ self.mean, torch.sqrt(inputs @ self.covariance @ inputs.t() + self.pred_logvar.exp())).sample()
    
def get_mse_ci_indivtasks(path, task, points):
    '''
    Get the mse and confidence interval for a given task and model

    Args:
        path (str): path to the folder where the results are stored
        task (str): task to retrieve the results for
        points (int): number of points to use for each task

    Returns:
        mse (np.array): mean squared error for each trial
        ci (np.array): confidence interval for each trial
    '''
    df = pd.read_csv(f'{path}{task}.csv')
    num_runs = df.run.max() + 1
    mse = np.zeros((points, num_runs))
    for run in range(num_runs):
        ytrue = (df[df['run'] == run].ytrue.to_numpy()[:points])
        ypred = (df[df['run'] == run].ypred.to_numpy()[:points])
        try:
            mse[:, run] = np.sqrt(((ytrue - ypred.astype('float')) ** 2))
        except:
            try:
                ypred = np.array([ y.replace('[', '').replace(']', '').replace(',', '').strip().split(' ')[0] for y in ypred]) #get rid of some processing errors I had
                mse[:, run] = np.sqrt(((ytrue - ypred.astype('float')) ** 2))
            except:
                print('error')
                import ipdb; ipdb.set_trace()
        # check if any element just stored in "mse" is > 2. If it is cap it to 2. 2 is chosen because it is squashed between -1 and 1 and therefore the max loss should be 2
        if np.any(mse[:, run] > 2):
            mse[:, run][mse[:, run] > 2] = 2

    ci =  1.96 * np.std(mse, axis=1) / np.sqrt(num_runs)
    mse  = np.mean(mse, axis=1)
    return mse, ci

def get_mse_ci(path, tasks, points):
    '''
    Get the mse and confidence interval for a given model averaged over all tasks

    Args:
        path (str): path to the folder where the results are stored
        tasks (list): list of tasks to retrieve the results for
        points (int): number of points to use for each task

    Returns:
        mse (np.array): mean squared error averaged over all tasks
        ci (np.array): confidence interval averaged over all tasks
    '''
    num_runs = pd.read_csv(f'{path}{tasks[0]}.csv').run.max() + 1
    print(f'num_runs: {num_runs} for engine {path.split("/")[2]}')
    mse = np.zeros((points, num_runs, len(tasks)))

    for idx, task in enumerate(tasks):
        df = pd.read_csv(f'{path}{task}.csv')
        for run in range(num_runs):
            ytrue = (df[df['run'] == run].ytrue.to_numpy()[:points])
            ypred = (df[df['run'] == run].ypred.to_numpy()[:points])
            try:
                mse[:, run, idx] = np.sqrt(((ytrue - ypred.astype('float')) ** 2))
            except:
                try:
                    ypred = np.array([ y.replace('[', '').replace(']', '').replace(',', '').strip().split(' ')[0] for y in ypred]) #get rid of some processing errors I had
                    mse[:, run, idx ] = np.sqrt(((ytrue - ypred.astype('float')) ** 2))
                except:
                    import ipdb; ipdb.set_trace()
            #check if any element just stored in "mse" is > 2. If it is cap it to 2. 2 is chosen because it is squashed between -1 and 1 and therefore the max loss should be 2
            if np.any(mse[:, run, idx] > 2):
                mse[:, run, idx][mse[:, run, idx] > 2] = 2

    ci =  1.96 * np.std(mse, axis=(1,2)) / np.sqrt(num_runs * len(tasks))
    mse = np.mean(mse, axis=(1,2))
    return mse, ci

def word2vec(string):
    '''
    Get the word2vec vector of a string
    Args:
        string (str): string to get the word2vec vector of
    Returns:
        word2vec vector (np.array)
    '''
    import gensim.downloader as api
    w2v = api.load('word2vec-google-news-300')
    #If the string has multiple words make the average of the word2vec vectors, else return the word2vec vector of the word
    string = string.replace('medexp', 'medical expenditure').replace('whitewine', 'white wine').replace('laborsupply', 'labor supply')
    if len(string.split()) > 1:
        return np.mean([w2v.get_vector(word) for word in string.split()], axis=0)
    return w2v.get_vector(string)