import os
import pandas as pd
import openai
import json
import re
path = 'data/main_sample'

#f = open("api_key.txt", "r")
#api_key = f.read()
#openai.api_key = api_key
#f.close()
#model = "gpt-4o-mini-2024-07-18"


def labeling():
    label = []
    print(label)
    fl = ''
    year = input('Year: ')
    cls = False
    if "clear" in year:
        year = year.split(' ')[1]
        cls = True
    for f in os.listdir(path):
        if year in f and f.endswith('label.csv'):
            fl = f
            print("Getting data from:", fl)
            break
    df = pd.read_csv(os.path.join(path, fl))
    df.fillna('', inplace=True)
    df['pset'] = df['NUMBER'].apply(
        lambda x: str(x)[:3] if (isinstance(x, str) and len(x) >= 3) else x)
    last_type = ""
    for type in df['pset'].unique():
        if type == '':
            continue
        print("Type:", type)
        df_type = df[df['pset'] == type]
        df_type = df_type[['NUMBER', 'QUESTION']]
        print(df_type)
        wait = input("Press Enter to continue...")
labeling()