import torch 
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
import pandas as pd

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
        if self.use_bias:
            inputs = torch.cat((inputs.reshape(-1, 1), torch.ones(1, 1)), dim=-1)

        # Equations from Bishop 3.3.1
        inv_covariance = torch.inverse(self.covariance) + (1/self.pred_logvar.exp()) * inputs.t() @ inputs
        self. mean = torch.inverse(inv_covariance) @ ( torch.inverse(self.covariance) @ self.mean + (1/self.pred_logvar.exp()) * inputs.t() @ target)
        self.covariance = torch.inverse(inv_covariance)

    def predict(self, inputs):
        if self.use_bias:
            inputs = torch.cat((torch.Tensor(inputs), torch.ones(1)))
        return Normal(inputs @ self.mean, torch.sqrt(inputs @ self.covariance @ inputs.t() + self.pred_logvar.exp())).sample()

def averaged_generated_function(fct_params, iters=1000, fct_type='linear'):
    """
    Generate a function and average it over multiple iterations.
    Args:
        fct_params (tuple): tuple containing the parameters of the generative function.
        iters (int): number of iterations.
        fct_type (str): type of function to be generated.
    Returns:
        Y_mean (numpy array): mean of the generated function.
        Y_std (numpy array): standard deviation of the generated function.
    """
    # Set the parameters of the generative function
    p1, p2, pb1, pb2, pc1, pc2, noise_ = fct_params
    a_gen = lambda: np.random.normal(p1, p2)
    b_gen = lambda: np.random.normal(pb1, pb2)
    c_gen = lambda: np.random.normal(pc1, pc2)
    noise = lambda: np.random.normal(0, noise_)
    Xs = np.linspace(0, 100, 100)
    Ys = np.empty((iters, 100))

    for i in range(iters):
        a = a_gen()
        b =b_gen() 
        if fct_type == 'linear':
            Ys[i] = a*Xs+b + noise()
        elif fct_type == 'quadratic':
            c = c_gen()
            Ys[i] = a*Xs**2 + b*Xs + c + noise()
        elif fct_type == 'exponential':
            c = c_gen()
            Ys[i] = a * np.exp(b*Xs) + c + noise()
        elif fct_type == 'periodic':
            c = c_gen()
            Ys[i] = 100 * np.sin(a*Xs + b) + c + noise()
        elif fct_type == 'step':
            c = c_gen()
            Ys[i] = np.where(Xs < c, a, b) + noise()
    Y_mean = np.mean(Ys, axis=0)
    Y_std = np.std(Ys, axis=0)
    return Y_mean, Y_std

def llm(engine, max_tokens):
    '''
    LLM for interpolation task. Replace it with your own LLM.
    Args:
        engine (str): engine of the LLM
        max_tokens (int): max tokens
        temp (float): temperature
    Returns:
        llm (function): LLM
    '''
    #! NB: For text-davinci-002, I realized afterwards that I could only retrieve negative predictions with suffix ';'given as input to the OpenAI API.
    if engine == 'random':
        def random(text, temp, max_tokens=max_tokens): 
            return  str(np.random.randint(0, 100))
        return random
    else:
        ValueError('Add your own LLM interaction function here.')

def preprocess(text):
    '''
    Preprocesses the text output from LLM. Deletes all non numerical characters and deletes everything after 'Machine' if it is in the text. 
    Args:
        text (str): raw text output from LLM
    Returns:
        text (str): preprocessed text
    '''
    text = text.replace(' ', '').replace(',', '').replace('\n', '').replace(':', '').replace('\'', '').replace('(', '').replace(')', '').replace('x', '').replace('y', '').replace('=', '').replace(';', '')
    #Check if there is 'Machine' in text
    if 'Machine' in text:
        #del everything after 'Machine'
        text = text[:text.index('Machine')]

    # Deletes letters at the end of the string
    while text[-1].isalpha():
        text = text[:-1]

    # If the last character is a ., delete it
    if text[-1] == '.':
        text = text[:-1]
    return text

def generate_function(fct_name, params):
    '''
    Args:
        fct_name (str): name of the function to be generated
        params (tuple): parameters of the function to be generated
    Returns:
        fct (function): function to be generated
    '''
    p1, p2, pb1, pb2, pc1, pc2, noisevar = params
    a = np.random.normal(p1, p2)
    b = np.random.normal(pb1, pb2)
    c = np.random.normal(pc1, pc2)
    noise = lambda: np.random.normal(0, noisevar)
    if fct_name == "linear":
        fct =  lambda x: a*x + b + noise()
        print(f'{a}x + {b}')
    elif fct_name == "quadratic":
        fct =  lambda x: a*x**2 + b*x + c + noise()
        print(f'{a}x^2 + {b}x + {c}')
    elif fct_name == "exponential":
        fct = lambda x: a*np.exp(b*x) + c
    elif fct_name == "periodic":
        fct = lambda x: np.sin(a *x + b)*100 + c + noise()
    elif fct_name == "step":
        # step function
        def step_function(x, a, b, c):
            return np.where(x < c, a, b)
        fct = lambda x: step_function(x, a, b, c) + noise()
    else:
        raise ValueError('Function should be either linear, quadratic, periodic, step or exponential')
    return fct

def store_data(path, data):
    '''
    Args:
        path (str): path to store data
        data (dict): data to be stored
    '''
    path += '.csv'
    dir_path = path.replace('/'+path.split('/')[-1], '')
    prevdir_path = dir_path.replace('/'+dir_path.split('/')[-1], '')

    # Create directory if it does not exist
    if os.path.exists(prevdir_path) == False:
        os.mkdir(prevdir_path)
        os.mkdir(dir_path)
    elif os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

    # Create dataframe and store it in csv file
    data = pd.DataFrame.from_dict(data)
    if os.path.exists(path):
        # If already created, append to csv file
        data['subject'] = [pd.read_csv(path)['subject'].max() + 1 for _ in range(len(data))] 
        data.to_csv(path, mode='a', header=False, index=False) 
    else:
        data['subject'] = [1 for _ in range(len(data))] 
        data.to_csv(path, mode='a', index=False)

        
