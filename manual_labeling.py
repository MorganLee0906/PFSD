# Description: Labeling the questions with GPT-4
# If the GPT-4 response is not correct, the user can input the correct label and the program will save the result to fine_tune.jsonl
# The program will also save the new label to labels.txt
import os
import pandas as pd
import openai
import json
import re
path = 'data/main_sample'



def labeling():
    # fine_tune = []
    # base_year = input('Base year: ')
    # fl = ''
    # for f in os.listdir(path):
    #     if base_year in f and f.endswith('label.csv'):
    #         fl = f
    #         print("Getting data from:", fl)
    #         break
    # base_df = pd.read_csv(os.path.join(path, fl))
    label = []
    if os.path.exists('label.txt'):
        with open('label.txt', 'r') as f:
            label = list(f.read().split('\n'))

    label = list(set(label))
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
    for pbset in df['pset'].unique():
        if pbset == '':
            continue
        # system_prompt = "輸出根據題組內容最適當的標籤，無需輸出「加掛題組」等字"
        system_prompt = f""
        print('=====================')
        print(pbset)
        print(df[df['pset'] == pbset]['QUESTION'])

        # Clear labels
        if 'LABEL' in df.columns and cls:
            df["LABEL"] = ""
            print("Clearing labels...")
            cls = False
        elif 'LABEL' not in df.columns:
            df['LABEL'] = ""

        if last_type != "":
            prompt = f"The question type before is {last_type}. Question: {df[df['pset'] == pbset]['QUESTION'].values}"
        else:
            prompt = f"Question: {df[df['pset'] == pbset]['QUESTION'].values}"
        print(prompt)
        try:
            if df[df['pset'] == pbset]['LABEL'].values[0] == '':
                this_type = ""
            else:
                this_type = df[df['pset'] == pbset]['LABEL'].values[0]
            if this_type not in label:
                print("===New label===")
            if True:
                wait = input(
                    f'This {this_type}, Last {last_type}, Correct? (y/n)')
                if 'exit' in wait:
                    break
                elif wait != '':
                    correct_label = wait
                    this_type = correct_label
                    msg = {}
                    msg["messages"] = []
                    msg["messages"].append(
                        {"role": "system", "content": f"{system_prompt}"})
                    msg["messages"].append(
                        {"role": "user", "content": f"{prompt}"})
                    msg["messages"].append(
                        {"role": "assistant", "content": f"{correct_label}"})
                    tune_json = open('fine_tune.jsonl', 'a+')
                    msg_text = json.dumps(msg, ensure_ascii=False)
                    print(msg_text)
                    tune_json.write(msg_text+'\n')
                    tune_json.close()
                

            last_type = this_type

            if last_type not in label:
                label.append(last_type)

            df.loc[df['pset'] == pbset, 'LABEL'] = last_type

        except Exception as e:
            print(e)
            df.to_csv(os.path.join(path, fl.replace(
                "label", "labeled")), index=False)
            wait = input('Press Enter to continue...')
            if wait == 'exit':
                break
            continue

    df.to_csv(os.path.join(path, fl), index=False)
    label.sort()
    w = open('labels.txt', 'w')
    w.write('\n'.join(label))
    w.close()


labeling()
