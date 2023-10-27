import numpy as np
import openai
import random
from dotenv import load_dotenv
import os

def llm(engine, action_letters=['F', 'J']):
    '''
    LLM for interpolation task. Replace this function with your own LLM.
    Args:
        engine (str): engine of the LLM
        max_tokens (int): max tokens
        temp (float): temperature
    Returns:
        llm (function): LLM
    '''
    if engine == 'random':
        def random(text, action_letters=action_letters): 
            return np.random.choice(action_letters)
        return random
    #TODO: Add your own LLM here
    elif engine.startswith('text-'):
        return gpt3
    else:
        ValueError('Add your own LLM interaction function here.')

def gpt3(text, max_tokens=1, action_letters=['F', 'J']):
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key
    engine = 'text-davinci-002'
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = max_tokens,
        temperature = 0.0,
    )
    if response.choices[0].text.strip() not in action_letters:   #When answer is not part of the bandit arms
        wrong_answer = response.choices[0].text.strip()
        response = openai.Completion.create(
            engine = engine,
            prompt = text + f" {wrong_answer}" + f"\nQ: Machine {wrong_answer} is not part of this casino. Which machine do you choose between machine {action_letters[0]} and machine {action_letters[1]}?\nA: Machine",
            max_tokens = max_tokens,
            temperature = 0.0,
        )
    return response.choices[0].text.strip()

def llm_prior(engine):
    if engine == 'random':
        def random(text, arm1, max_tokens=3, suffix='points.'): 
            return np.random.normal(0, 1)
        return random
    #TODO: Add your own LLM here
    elif engine.startswith('text-'):
        return gpt3_prior
    else:
        ValueError('Add your own LLM interaction function here.')

def gpt3_prior(text,arm1, max_tokens=3, suffix='points.'):
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key
    engine = 'text-davinci-002'
    # Loop until we get a number because with temperature set to 1.0, we sometimes get non-numbers
    for i in range(5):
        openai.api_key = key
        response = openai.Completion.create(
            engine = engine,
            prompt = text,
            max_tokens = max_tokens,
            temperature = 1.0,
            suffix = suffix
        )
        try:
            output = float(response.choices[0].text.strip())
            return output
        except:
            try:
                upd_text = text + f" {response.choices[0].text.strip()} \nQ: {response.choices[0].text.strip()} is not a number of points. Let me ask again. How rewarding do you expect machine {arm1} to be?\nA: I expect machine {arm1} to deliver an average of approximately "
                response = openai.Completion.create(engine = engine,prompt = upd_text ,max_tokens = 3,temperature = 1.0)
                output = float(response.choices[0].text.strip())
                return output
            except:
                print(f"Error: {response.choices[0].text.strip()} is not a number")
                if i==3:
                    import ipdb; ipdb.set_trace()
                pass

def sample_alphabet(alphabet): 
    arm1 = random.choice(alphabet)
    alphabet = alphabet.replace(arm1, '')
    arm2 = random.choice(alphabet)
    alphabet = alphabet.replace(arm2, '')
    return arm1, arm2, alphabet

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False