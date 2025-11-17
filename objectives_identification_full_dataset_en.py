from dotenv import load_dotenv
import os
from utils import utils
from model import model
from tqdm import tqdm
import pandas as pd
import json
import random
import time

#Constants
load_dotenv() # Load .env variables
API_KEY = os.getenv("API_KEY_1_1")
# API_KEY = os.getenv("API_KEY_2")
MODEL_NAME='gemini-2.5-flash-lite'
BATCH_SIZE=20


DATASET_NAME='cop27_en_original_full_text'
RUN_NAME=f'objectives_cop_no_guide_{DATASET_NAME}_batch_20'
DATASET_PATH=f'/data/jfraile/Programs/datasets/COP27/{DATASET_NAME}.pkl'
PATH_TO_SAVE=f'/data/jfraile/Programs/COP27/datasets_with_answer/{RUN_NAME}.pkl'



def get_prompt(tweet_batch):
    t=f"""
You are an expert in identifying strategic communication objectives. Your task is to analyse a list of {len(tweet_batch)} tweets about the Climate Summit (COP27) and determine which communication objectives from the list provided does each tweet support?

A tweet supports a strategic communication objective only when its primary reading clearly aligns with that objective, meaning that the connection is explicit and does not require inference or the combination of multiple ideas to be understood.

The analysis must always be conducted from the perspective of the message sender.

List of Strategic Communication Objectives:

1. Reinforce one's own image, ideology, or values.
2. Legitimise decisions.
3. Seek unity against a hostile enemy.
4. Weaken the opponent's image, identity, ideology, or values.
5. Delegitimise the opponent's decisions or policies.
6. Promote social change or activate new policies.
7. Stop social change or defend the status quo.

Instructions:

1. Read each tweet carefully.

2. Determine which Strategic Communication Objectives, if any, the tweet promotes using the list above. A tweet supports a strategic communication objective only when its primary reading clearly aligns with that objective, meaning that the connection is explicit and does not require inference or the combination of multiple ideas to be understood.

3. If the tweet does not promotes any Strategic Communication Objectives, return [].

4. Output the results ONLY as a JSON object, where the keys are tweet numbers (0-{len(tweet_batch)-1}) and the values are arrays of objectives numbers. Do NOT include any extra text or explanation.

Example format:
{{0: [1, 3], 1: [2, 4], 2: [], ...}}

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
    df=utils.load_cop_dataset_full_text(DATASET_PATH)

    # #n rows for debugg
    # df=df.sample(frac=1)[:BATCH_SIZE].reset_index(drop=True)
    
    df_answers=df.copy()
    df_answers['objectives']=pd.Series()
    print("Created new DataFrame")


gemini_model=model.GeminiModel(API_KEY, MODEL_NAME, 'objectives_identification')


for start_idx in tqdm(range(0, len(df_answers), BATCH_SIZE)):
    end_idx = min(start_idx + BATCH_SIZE, len(df_answers))
    
    batch_df = df_answers.iloc[start_idx:end_idx]
    batch_texts = []
    indices_to_update = []
    
    for i, row in batch_df.iterrows():
        if utils.check_nan(row['objectives']):
            batch_texts.append(row['Text'])
            indices_to_update.append(i)
    
    if not batch_texts:
        continue  
    

    prompt = get_prompt(batch_texts)  

    answer_dict = gemini_model.do_inference(prompt)
    
    for idx in range(len(batch_texts)):
        try:
            df_answers.at[indices_to_update[idx], 'objectives'] = answer_dict[idx]
        except:
            print("Error while saving batch answers")
    
    df_answers.to_pickle(PATH_TO_SAVE, protocol=4)
    print(f"Saved batch {start_idx}-{end_idx}")




