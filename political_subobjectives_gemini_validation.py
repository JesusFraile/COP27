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
API_KEY = os.getenv("API_KEY_1_2")
# API_KEY = os.getenv("API_KEY_2")
MODEL_NAME='gemini-2.5-flash-lite'
BATCH_SIZE=20


RUN_NAME=f'political_narratives_validation_only_none_narrative'
PATH_TO_SAVE=f'/data/jfraile/Programs/COP27/datasets_with_answer/{RUN_NAME}.pkl'

df_es=pd.read_pickle('/data/jfraile/Programs/datasets/COP27/cop27_es_original_full_text.pkl')
df_en=pd.read_pickle('/data/jfraile/Programs/datasets/COP27/cop27_en_original_full_text.pkl')

n_es=int(20000*0.2)
n_en=int(20000*0.8)
df_random_es=df_es.sample(n=n_es, random_state=13).reset_index(drop=True)
df_random_en=df_en.sample(n=n_en, random_state=13).reset_index(drop=True)
df_random_en['lang']='en'
df_random_es['lang']='es'
df_to_process=pd.concat([df_random_es, df_random_en], ignore_index=True)



def get_prompt(tweet_batch):
    t=f"""
You are an expert in identifying strategic communication narratives. 
Your task is to analyse a list of {len(tweet_batch)} tweets about the Climate Summit (COP27) and determine which narratives from the list provided each tweet supports.

The analysis must always be conducted from the perspective of the message sender.

List of Narratives:

1. Sembrar la Duda sobre la Ciencia
2. Desacreditar las Instituciones Clave
3. Minimizar la Gravedad (Retardismo)
4. Explotar el Costo Económico
5. Promover la Inacción (Shifting the Blame)
6. Promover Soluciones Dilatorias (Retardismo Tecnológico)
7. Polarización y Movilización Antirregulación
8. Infiltración y Cabildeo Directo
9. Establecer la Urgencia y la Base Científica
10. Localizar el Impacto
11. Conectar Mitigación y Adaptación
12. Fomentar una Narrativa de Esperanza y Empoderamiento
13. Presionar por una Mayor Ambición (Aumento de las NDC)
14. Visibilizar el Liderazgo y la Responsabilidad
15. Asegurar la Financiación Climática
16. Promover una Transición Justa
17. Reforzar el Multilateralismo
18. Incorporar al Sector Privado
19. Fomentar la Participación Inclusiva
20. Divulgar Resultados y Herramientas

Instructions:

1. Read each tweet carefully.
2. Determine which Narratives, if any, the tweet promotes using the list above. 
3. If the tweet promotes another clear and explicit narrative that is not in the list **and does not match any existing narrative**, include it in the output as a new narrative label that is general enough to apply to multiple tweets and with the **same level of abstraction** as the existing narratives  (e.g., if different tweets promote the same idea such as "China is a great nation" and "China is a great country", 
   you must use only one unified narrative label for all of them). The generated narratives must be 
4. If the tweet does not promote any narrative, return [].
5. If a tweet supports one of the 20 existing narratives, you MUST always include its corresponding NUMBER in the output. 
   You may ONLY include text (a string) if the tweet promotes a new narrative that is not in the list.
6. Output the results ONLY as a JSON object, where the keys are tweet numbers (0–{len(tweet_batch)-1}) 
   and the values are arrays of narrative numbers and/or new narrative strings. 
   Do NOT include any extra text or explanation.

Example format:
{{
  0: [1, 3],
  1: [2, 4, "New narrative that supports"],
  2: []
}}


Tweets to analyse:
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
    df_answers=df_to_process.copy()
    df_answers['narratives']=pd.Series()
    print("Created new DataFrame")


gemini_model=model.GeminiModel(API_KEY, MODEL_NAME, 'objectives_identification')


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




