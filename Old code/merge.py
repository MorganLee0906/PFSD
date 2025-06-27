import pandas as pd
import re


def keep_chinese(input_string):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5，。！？]')
    result = re.findall(chinese_pattern, input_string)
    filtered_string = ''.join(result)
    return filtered_string


def eval_option(option):
    key, value = option.split('=')
    return key.strip(), keep_chinese(value.strip())


type = input("Enter the type you want to merge: ")
cluster_file = pd.read_csv(
    'data/five_years/five_years_by_type/type_' + type + '.csv')
cluster_file = cluster_file.sort_values(by=['YEAR'])
years = set(cluster_file['YEAR'])
print('This type include years: ', years)

clusters = list(set(sorted(cluster_file['cluster'])))
unclustered = cluster_file[cluster_file['cluster'] == 0]
for idx in unclustered.index:
    cluster_file.at[idx, 'cluster'] = clusters[-1] + 1
    clusters.append(clusters[-1] + 1)
    print(clusters[-1]+1, cluster_file.at[idx, 'QUESTION'])
clusters.pop(0)
merged_dataframe = pd.DataFrame(columns=['id', *clusters])
col_header = []
for i in range(0, len(clusters)):
    col_header.append(
        cluster_file[cluster_file['cluster'] == clusters[i]]['QUESTION'].values[0])
    print('Cluster: ', clusters[i], 'Question: ', col_header[-1])
w = input("Continue? (y/n)")
for y in years:
    print('Processing year: ', y)
    option = pd.read_csv('data/five_years/' + y + '_answer.csv')
    survey = pd.read_csv('data/five_years/' + y + '_survey.csv')
    questions = cluster_file[cluster_file['YEAR'] == y]['ANSWER'].tolist()
    print('This year include questions: ', questions)
    option_dict = {}
    for q in questions:
        options = option[option['ANSWER'] == q]['OPTION'].tolist()
        option_dict[q] = {}
        for o in options:
            k, v = eval_option(o)
            option_dict[q][k] = v
    print("Finish processing options")
    for idx, data in survey.iterrows():
        data_row = [str(data.iloc[0])]
        for c in clusters:
            if cluster_file[(cluster_file['YEAR'] == y) & (cluster_file['cluster'] == c)].empty:
                data_row.append("N/A")
            else:
                num = cluster_file[(cluster_file['YEAR'] == y) & (
                    # find in the survey data
                    cluster_file['cluster'] == c)]['NUMBER'].values[0]
                ans = cluster_file[(cluster_file['YEAR'] == y) & (
                    # find in the option data
                    cluster_file['cluster'] == c)]['ANSWER'].values[0]
                try:
                    data_row.append(
                        option_dict[ans][str(int(data[num.lower()]))])
                except:
                    # print('>', ans, num, data[num.lower()])
                    data_row.append(data[num.lower()])
                    # Exception: The answer is not in the option (e.g. age/year), then we just add the original data to the row
        merged_dataframe.loc[len(merged_dataframe)] = data_row
    print("Finish year: ", y)
print("Finish all years")

# Modify column names
merged_dataframe.columns = ['id', *col_header]
print(merged_dataframe.columns)
merged_dataframe.sort_values(by=['id'], inplace=True)
merged_dataframe.to_csv(
    'data/five_years/merged_type_' + type + '.csv', index=False)
