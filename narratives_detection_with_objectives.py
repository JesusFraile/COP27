from dotenv import load_dotenv
import os
from utils import utils
from model import model
from tqdm import tqdm
import pandas as pd
import json
import random

#Constants
load_dotenv() # Load .env variables
API_KEY = os.getenv("API_KEY")
# API_KEY = os.getenv("API_KEY_2")
MODEL_NAME='gemini-2.5-flash-lite'
BATCH_SIZE=800
RUN_NAME='dipro_narrativas_with_objectives'
PATH_TO_SAVE=f'/data/jfraile/Programs/COP27/datasets_with_answer/{RUN_NAME}.pkl'



path='/data/jfraile/Programs/datasets/DIPROMATS_2024_T2/dipromats24_t2_train_en.json'
with open(path) as json_data:
    data=json.load(json_data)

narrative_example=[]
for n in data['narratives']:
    narrative_example.append(n['title'])

random.shuffle(narrative_example)


def get_prompt(tweet_batch):
    t=f"""
You are an expert analyst specialised in geopolitical discourse and diplomatic communications. Your task is to analyse a list of strategic communication objectives and determine which narratives, if any, contribute to achieving these objectives.

Strategic Communication Objectives:

OBJ1. Domination and Control
This is the central objective discussed by the Frankfurt School. Reason is used to maintain the status quo and existing power structures.

OBJ2. Persuasion and Manipulation
This involves using communication to influence people's decisions and behaviours, not through rational arguments, but through emotions and psychological techniques.

OBJ3. Legitimisation and Justification 
The instrumentalisation of reason is also used to justify actions or systems that would otherwise be unacceptable. The aim is to give an appearance of rationality to what is irrational or unjust.


Instructions:

1. Read each communication objective carefully (OBJ1, OBJ2, OBJ3).
2. Identify which narratives explicitly or implicitly support the achievement of that objective. A narrative can be political, social, economic, or ideological.
3. If no narratives contribute to the objective, return the text NONE.
4. Output the results ONLY as a JSON object, where the keys are tweet numbers (0-{len(tweet_batch)-1}) and the values are arrays of narrative strings. Do NOT include any extra text or explanation.

Example format:
{{0: ["Narrative 1", "Narrative 2"], 1: [], 2: ["Narrative 3"], ...}}

Tweet to analyse:
{utils.create_tweet_set_prompt(tweet_batch)}

Output:
"""
    return t

#Loading save dataset
if os.path.exists(PATH_TO_SAVE):
    df_answers=pd.read_pickle(PATH_TO_SAVE)
    print("Loaded DataFrame")
else:
    #Load dataset
    df=utils.load_dataset('es')

    #n rows for debugg
    df=df.sample(frac=1)[:BATCH_SIZE].reset_index(drop=True)
    
    df_answers=df.copy()
    print("Created new DataFrame")


gemini_model=model.GeminiModel(API_KEY, MODEL_NAME, 'narrative_identification')


for start_idx in tqdm(range(0, len(df_answers), BATCH_SIZE)):
    end_idx = min(start_idx + BATCH_SIZE, len(df_answers))
    
    batch_df = df_answers.iloc[start_idx:end_idx]
    batch_texts = []
    indices_to_update = []
    
    for i, row in batch_df.iterrows():
        if utils.check_nan(row['narratives']):
            batch_texts.append(row['text'])
            indices_to_update.append(i)
    
    if not batch_texts:
        continue  
    

    prompt = get_prompt(batch_texts)  
    
    answer_dict = gemini_model.do_inference(prompt)
    
    for idx in range(len(batch_texts)):
        try:
            df_answers.at[indices_to_update[idx], 'narratives'] = answer_dict[idx]
        except:
            print("Error while saving batch answers")
    
    df_answers.to_pickle(PATH_TO_SAVE, protocol=4)
    print(f"Saved batch {start_idx}-{end_idx}")




