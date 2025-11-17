import pandas as pd
import json
import re
import numpy as np

def load_dipromats_dataset(lang):
    RUTE_FILES='/data/jfraile/Programs'
    df_path=f'{RUTE_FILES}/datasets/DIPROMATS_2024_T2/test_dataset_{lang}.json'
    with open(df_path) as json_data:
        data=json.load(json_data)
        df_t2=pd.json_normalize(data)
        df_t2.reset_index(inplace=True)
        df_t2['country'].replace({'Russia':'Russia', 'China':'China', 'European Union':'EU', 'USA':'USA'}, inplace=True)

        df_t2['language']=lang
        df_t2.rename(columns={'text':'Text'}, inplace=True)
        # df_t2['narratives']=pd.Series()
    return df_t2

def check_nan(value):
    try:
        # Converting value to float
        if np.isnan(float(value)):
            return True
        return False
    except (ValueError, TypeError):
        # If it cannot be converted, it is a non-NaN string.
        return False
    
def create_tweet_set_prompt(tweets):
    t=''

    for i in range(len(tweets)):
        tweet=tweets[i]
        t=f"""{t}
Tweet {i}. {tweet}
"""
    return t

def load_cop_dataset(path):
    df=pd.read_csv(path)
    return df
def load_cop_dataset_full_text(path):
    df=pd.read_pickle(path)
    df.rename(columns={'text':'Text'}, inplace=True)
    return df
