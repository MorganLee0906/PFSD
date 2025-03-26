import pandas as pd
import os
import re
from scipy.spatial.distance import cosine


def get_question_group(number):
    # r1 = re.compile(r'[a-zA-Z]\d{2}[a-zA-Z]\d+$')
    # if r1.match(number):
    #    return number[:4]
    # else:
    return number[:3]


embedding_path = 'data/five_years/five_years_with_embeddings'
pkl_lst = {}
for e in os.listdir(embedding_path):
    if not e.endswith('.pkl'):
        continue
    pkl = pd.read_pickle(embedding_path + '/' + e)
    pkl_lst[e[:-4].split('_')[0]] = pkl

embedding = pd.concat([df[['QUESTION', 'Q_embedding']]
                      for df in pkl_lst.values()], ignore_index=True)
print(embedding)

folder_path = 'data/five_years'

all_time = pd.DataFrame(columns=['YEAR', 'LABEL', 'QUESTION', 'Q_embedding'])
error_row = pd.DataFrame(columns=['YEAR', 'LABEL', 'QUESTION'])
for f in os.listdir(folder_path):
    if not f.endswith('label.csv'):
        continue
    csv = pd.read_csv(os.path.join(folder_path, f)).dropna()
    for _, row in csv.iterrows():
        row['Q_embedding'] = embedding[embedding['QUESTION']
                                       == row['QUESTION']]['Q_embedding'].values[0]
        selected = row[['YEAR', 'LABEL', 'QUESTION', 'Q_embedding']]
        all_time = pd.concat(
            [all_time, pd.DataFrame([selected])], ignore_index=True)
    print(f"Finish processing {f}")
    all_time["check"] = ""
    for i, row1 in all_time.iterrows():
        for j, row2 in all_time.iterrows():
            if i >= j or row1['YEAR'] == row2['YEAR'] or row1['check'] or row2['check']:
                continue
            sim = 1 - cosine(row1['Q_embedding'], row2['Q_embedding'])
            if sim > 0.95:
                print(
                    f"Similarity between {row1['QUESTION']} and {row2['QUESTION']}: {sim}")
                if row1['LABEL'] != row2['LABEL']:
                    print(
                        f"Label mismatch: {row1['LABEL']} and {row2['LABEL']}")
                    if "：" in row1['LABEL'] and "：" in row2['LABEL'] and row1['LABEL'].split('：')[1] == row2['LABEL'].split('：')[1]:
                        print("Same sub_category. Ignore.")
                        continue
                    wait = input("Correct? (y/n)")
                    if wait == 1:
                        error_row = pd.concat(
                            [error_row, pd.DataFrame([row1])], ignore_index=True)
                        all_time.loc[i, 'LABEL'] = ""
                        all_time.loc[i, 'check'] = "N"
                    elif wait == 2:
                        error_row = pd.concat(
                            [error_row, pd.DataFrame([row2])], ignore_index=True)
                        all_time.loc[j, 'LABEL'] = ""
                        all_time.loc[j, 'check'] = "N"
                    elif wait == 3:
                        error_row = pd.concat(
                            [error_row, pd.DataFrame([row1])], ignore_index=True)
                        error_row = pd.concat(
                            [error_row, pd.DataFrame([row2])], ignore_index=True)

                        all_time.loc[i, 'LABEL'] = ""
                        all_time.loc[j, 'LABEL'] = ""
                        all_time.loc[i, 'check'] = "N"
                        all_time.loc[j, 'check'] = "N"
                    error_row.to_csv('error_row.csv', index=False)

print(error_row)
