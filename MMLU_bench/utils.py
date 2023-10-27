import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd

def llm(engine):
    '''
    LLM interaction function.
    Args:
        engine: LLM engine to use.
    Returns:
        A function that takes a text prompt and returns a completion.
            '''
    if engine == 'random':
        return lambda _: np.random.choice(['A', 'B', 'C', 'D'])
    #TODO: Add your own LLM here
    elif engine.startswith('text-'):
        return gpt3
    else:
        ValueError('Add your own LLM interaction function here.')


def gpt3(text, engine='text-davinci-002', max_tokens=2, temp=0):
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        max_tokens = max_tokens,
        temperature = temp,
    )
    return response.choices[0].text.strip().replace(" ",'')[0]

def get_category_from_name(name):
    name_dict = {
        "high school psychology": "Social Sciences",
        "anatomy": "STEM",
        "virology": "Other",
        "high school mathematics": "STEM",
        "college computer science": "STEM",
        "security studies": "Social Sciences",
        "formal logic":  "Humanities",
        "philosophy": "Humanities",
        "high school computer science": "STEM",
        "nutrition": "Other",
        "moral scenarios": "Humanities",
        "professional medicine": "Other",
        "human aging": "Other",
        "high school biology": "STEM",
        "professional law": "Humanities",
        "marketing": "Other",
        "high school european history": "Humanities",
        "public relations": "Social Sciences",
        "clinical knowledge": "Other",
        "logical fallacies": "Humanities",
        "prehistory": "Humanities",
        "high school geography": "Social Sciences",
        "high school world history": "Humanities",
        "econometrics": "Social Sciences",
        "global facts": "Other",
        "high school macroeconomics": "Social Sciences",
        "jurisprudence": "Humanities",
        "management": "Other",
        "college medicine": "Other",
        "college chemistry": "STEM",
        "conceptual physics": "STEM",
        "high school physics": "STEM",
        "computer security": "STEM",
        "moral disputes": "Humanities",
        "international law": "Humanities",
        "miscellaneous": "Other",
        "us foreign policy": "Social Sciences",
        "astronomy": "STEM",
        "world religions": "Humanities",
        "machine learning": "STEM",
        "high school microeconomics": "Social Sciences",
        "abstract algebra": "STEM",
        "human sexuality": "Social Sciences",
        "high school us history": "Humanities",
        "college physics": "STEM",
        "electrical engineering": "STEM",
        "high school chemistry": "STEM",
        "medical genetics": "Other",
        "professional psychology": "Social Sciences",
        "college biology": "STEM",
        "business ethics": "Other",
        "elementary mathematics": "STEM",
        "sociology": "Social Sciences",
        "high school statistics": "STEM",
        "college mathematics": "STEM",
        "high school government and politics": "Social Sciences",
        "professional accounting": "Other",
        }

    return name_dict[name]


def get_names_from_category(category):
    if category == 'STEM':
        return ['anatomy', 'high school mathematics', 'college computer science', 'high school computer science', 'high school biology', 'college chemistry', 'conceptual physics', 'high school physics', 'college mathematics', 'high school statistics', 'elementary mathematics', 'college biology', 'high school chemistry', 'electrical engineering', 'college physics', 'abstract algebra', 'machine learning', 'astronomy', 'computer security']
    if category == 'Humanities':
        return ['formal logic', 'philosophy', 'moral scenarios', 'high school european history', 'logical fallacies', 'prehistory', 'high school world history', 'jurisprudence', 'high school us history', 'world religions', 'international law', 'moral disputes']
    if category == 'Social Sciences':
        return ['high school psychology', 'security studies', 'public relations', 'high school geography', 'econometrics', 'high school macroeconomics', 'high school government and politics', 'sociology', 'professional psychology', 'human sexuality', 'high school microeconomics' , 'us foreign policy']
    if category == 'Other':
        return ['virology', 'nutrition', 'professional medicine', 'human aging', 'marketing', 'clinical knowledge', 'global facts', 'management', 'professional accounting', 'business ethics', 'medical genetics', 'miscellaneous', 'college medicine']


def format_subject(subject, offset=5):
    l = subject.rsplit('/', 1)[-1]
    l = l.rsplit('.', 1)[0]
    l = l[:-offset]
    l = l.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    choices = ["A", "B", "C", "D"]
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def ada_embeddings(text, engine, model="text-embedding-ada-002"):
    load_dotenv(); key = os.getenv("OPENAI_API_KEY")
    openai.api_key = key

    path = f'task_sims/{engine}.csv'
    # Check if CSV file exists
    if not os.path.exists(path):
        # Create empty dataframe and save to CSV file
        df = pd.DataFrame(columns=['task', 'ada_embedding'])
        df.to_csv(path, index=False)
    df = pd.read_csv(path)
    if text in df['task'].unique():
        return np.array(eval(np.array(df[df['task'] == text]['ada_embedding'])[0]))
    else:
        try: 
            embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        except:
            raise ValueError('Here you are computing the word embedding for the task using the ada model, so you need to set your own OPENAI_API_KEY in a .env file.')

        #append to csv file task and embedding
        df = df.append({'task': text, 'ada_embedding': embedding}, ignore_index=True)
        df.to_csv(path, index=False)
        return np.array(embedding)