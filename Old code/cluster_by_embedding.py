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
embedding = {}
for e in os.listdir(embedding_path):
    if not e.endswith('.pkl'):
        continue
    pkl = pd.read_pickle(embedding_path + '/' + e)
    embedding[e[:-4].split('_')[0]] = pkl
    print(pkl)
    print('Finish reading:', e[:-4].split('_')[0])

folder_path = 'data/five_years/five_years_by_type'

while type != 'exit':
    type = input('Input the type:')
    if type == 'exit':
        break
    file_path = folder_path + '/type_' + type + '.csv'
    csv = pd.read_csv(file_path)
    csv['cluster'] = 0
    csv['embedding'] = ''
    csv['similarity'] = 0
    csv['category'] = ''
    refer = csv[csv['YEAR'] == 'CV2008'].index
    csv.loc[refer, 'cluster'] = range(1, len(refer) + 1)

    for idx in csv.index:
        data = csv.loc[idx]
        num = data['NUMBER']
        year = data['YEAR']
        csv.at[idx, 'embedding'] = embedding[year][embedding[year]
                                                   ['NUMBER'] == num]['Q_embedding'].values[0]
        if csv.loc[idx]['YEAR'] == 'CV2008':
            csv.at[idx, 'category'] = get_question_group(
                csv.loc[idx]['NUMBER'])

    for idx in csv.index:
        max = 0
        max_category = ''
        this_num = get_question_group(csv.loc[idx]['NUMBER'])
        if csv.loc[idx]['YEAR'] == 'CV2008':  # Ignore reference
            continue
        for ref in refer:
            sim = 1 - cosine(csv.loc[idx]['embedding'],
                             csv.loc[ref]['embedding'])
            if sim > max:
                max = sim
                max_category = get_question_group(csv.loc[ref]['NUMBER'])
            if sim == 1:
                print(
                    f'{csv.loc[idx]['YEAR']} {get_question_group(csv.loc[idx]['NUMBER'])} == {get_question_group(csv.loc[ref]['NUMBER'])}')
                if csv.loc[idx]['category'] != '' and csv.loc[idx]['category'] != get_question_group(csv.loc[ref]['NUMBER']):
                    print("Conflict: Original:",
                          csv.loc[idx]['category'], "New:", get_question_group(csv.loc[ref]['NUMBER']))
                csv.loc[(csv['NUMBER'].str.startswith(this_num)) &
                        (csv['YEAR'] == csv.loc[idx]['YEAR']), 'category'] = get_question_group(csv.loc[ref]['NUMBER'])
                csv.loc[idx, 'similarity'] = 1
                csv.loc[idx, 'cluster'] = csv.loc[ref]['cluster']

        if max > csv.loc[(csv['NUMBER'].str.startswith(this_num)) &
                         (csv['YEAR'] == csv.loc[idx]['YEAR']), 'similarity'].max() and max > 0.88:
            csv.loc[(csv['NUMBER'].str.startswith(this_num)) &
                    (csv['YEAR'] == csv.loc[idx]['YEAR']), 'category'] = max_category
            print(
                f'{csv.loc[idx]['YEAR']} {this_num} ~= {max_category}, sim = {max}')
    print("Check conflict:")
    new_category = 1
    for idx in csv.index:
        for itr in csv.index:
            if idx == itr:
                continue
            sim = 1 - cosine(csv.loc[idx]['embedding'],
                             csv.loc[itr]['embedding'])
            if sim == 1 and csv.loc[idx]['category'] != csv.loc[itr]['category']:
                print("Conflict:")
                print(csv.loc[idx]['YEAR'], csv.loc[idx]
                      ['QUESTION'], "is", csv.loc[idx]['category'])
                print("But", csv.loc[itr]['YEAR'], csv.loc[itr]
                      ['QUESTION'], "is", csv.loc[itr]['category'])
                target = 0
                ref = 0
                if csv.loc[idx]['category'] == '':
                    target, ref = idx, itr
                elif csv.loc[itr]['category'] == '':
                    target, ref = itr, idx
                if csv.loc[idx]['category'] == '' or csv.loc[itr]['category'] == '':
                    this_num = get_question_group(csv.loc[target]['NUMBER'])
                    all_category = csv[(csv['NUMBER'].str.startswith(this_num)) & (
                        csv['YEAR'] == csv.loc[target]['YEAR'])]['category'].unique()
                    print("All category in this subset:", all_category)
                    for c in csv[(csv['NUMBER'].str.startswith(this_num)) & (csv['YEAR'] == csv.loc[target]['YEAR'])].index:
                        if csv.loc[c]['category'] == csv.loc[ref]['category']:
                            continue
                        csv.at[c, 'category'] = csv.loc[ref]['category']
                        print("Merge", csv.loc[c]['YEAR'], csv.loc[c]
                              ['QUESTION'], csv.loc[c]['category'], "to", csv.loc[ref]['category'])
                    for c in all_category:
                        if c != '':
                            csv[csv['category'] ==
                                c]['category'] = csv.loc[ref]['category']
            # Not in base year(CV2008), but has same question -> new category
            elif sim == 1 and csv.loc[idx]['category'] == csv.loc[itr]['category'] == '':
                print("New category")
                num1 = get_question_group(csv.loc[idx]['NUMBER'])
                num2 = get_question_group(csv.loc[itr]['NUMBER'])
                if len(csv[(csv['NUMBER'].str.startswith(num1)) & (csv['YEAR'] == csv.loc[idx]['YEAR'])]['category'].unique()) == 1:
                    csv.loc[(csv['NUMBER'].str.startswith(num1)) & (
                        csv['YEAR'] == csv.loc[idx]['YEAR']), 'category'] = new_category
                if len(csv[(csv['NUMBER'].str.startswith(num2)) & (csv['YEAR'] == csv.loc[itr]['YEAR'])]['category'].unique()) == 1:
                    csv.loc[(csv['NUMBER'].str.startswith(num2)) & (
                        csv['YEAR'] == csv.loc[itr]['YEAR']), 'category'] = new_category
                new_category += 1
                print(num1, "and", num2, "are new category", new_category)
    not_clustered = csv[csv['category'] == '']['NUMBER']
    num = list(set([get_question_group(n) for n in not_clustered.to_list()]))
    for n in num:
        for idx in not_clustered.index:
            if csv.loc[idx]['NUMBER'].startswith(n):
                csv.at[idx, 'category'] = new_category
                print(csv.loc[idx]['NUMBER'], csv.loc[idx]
                      ['QUESTION'], "is new category", new_category)

    # print(f'{csv.loc[idx]['QUESTION']}, {csv[csv['cluster'] == max_cluster]['QUESTION'].values[0]}, sim = {max}, cluster = {max_cluster}')
    # csv.at[idx, 'cluster'] = max_cluster
    # csv.at[idx, 'similarity'] = max
    # csv = csv.sort_values(by='cluster')
    csv = csv.drop(columns=['embedding'])
    csv.to_csv(file_path[:-4] + '_sim_embed.csv', index=False)
