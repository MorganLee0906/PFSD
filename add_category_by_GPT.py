import os
import pandas as pd
dir_path = 'data/five_years'
for i in os.listdir(dir_path):
    if i.endswith('label.csv'):
        df = pd.read_csv(dir_path + '/' + i)
        for qnum in df['NUMBER']:
