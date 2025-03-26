# Description: Labeling the questions with GPT-4
# If the GPT-4 response is not correct, the user can input the correct label and the program will save the result to fine_tune.jsonl
# The program will also save the new label to labels.txt
import os
import pandas as pd
import openai
import json
import re
path = 'data/five_years'

f = open("api_key.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()


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
    if os.path.exists('labels.txt'):
        with open('labels.txt', 'r') as f:
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
        system_prompt = f"你是問卷題目分類專家，根據題組內容提供適當標籤. 目前的標籤如下（如無適當標籤可自行生成）: {label}, 讓每個問題被分配到適合的標籤(請特別注意問題所問的情境或身份，例如：問先生/太太的工作則輸出婚姻/同居/配偶) 問題所屬的標籤可能與上一題的標籤有關係，請務必參考。針對問題輸出單一標籤並且勿輸出任何多餘文字或符號"
        print('=====================')
        print(pbset)
        print(df[df['pset'] == pbset]['QUESTION'])

        # Clear labels
        if 'LABEL' in df.columns and cls:
            df["LABEL"] = ""
            print("Clearing labels...")
            cls = False

        if 'LABEL' in df.columns and df.loc[df['pset'] == pbset, 'LABEL'].values[0] != '':
            print("Already labeled")
            last_type = df.loc[df['pset'] == pbset, 'LABEL'].values[0]
            if last_type not in label:
                label.append(last_type)
            continue
        elif 'LABEL' not in df.columns:
            df['LABEL'] = ""

        print("Asking GPT-4...")
        if last_type != "":
            prompt = f"The question type before is {last_type}. Question: {df[df['pset'] == pbset]['QUESTION'].values}"
        else:
            prompt = f"Question: {df[df['pset'] == pbset]['QUESTION'].values}"
        print(prompt)
        try:
            response = openai.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:personal:labeling:BEueWn8W",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,  # 限制回應的長度以只包含 cluster 編號
                temperature=0.1  # 設定較低的隨機性以提高準確性
            )
            print(response.choices[0].message.content.strip().split('\n')[0])
            this_type = response.choices[0].message.content.strip().split('\n')[
                0]
            this_type = re.split(r'：|︰|・|:', this_type)[0]
            if this_type not in label:
                print("===New label===")
            if this_type != last_type:
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
                else:
                    msg = {}
                    msg["messages"] = []
                    msg["messages"].append(
                        {"role": "system", "content": f"{system_prompt}"})
                    msg["messages"].append(
                        {"role": "user", "content": f"{prompt}"})
                    msg["messages"].append(
                        {"role": "assistant", "content": f"{this_type}"})
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
