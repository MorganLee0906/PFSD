import pandas as pd
import os
dir_path = 'data/five_years/five_years_by_type/'
output_path = 'data/five_years/question_map/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
for type in range(1, 10):
    print("Reading type " + str(type) + "...")
    df = pd.read_csv(dir_path + 'type_' + str(type) + '_sim_embed.csv')
    question_map = pd.DataFrame()
    question_map['YEAR'] = df['YEAR'].unique()
    print(question_map)
    for cluster in range(df['cluster'].min(), df['cluster'].max()+1):
        print("Cluster " + str(cluster) + "...")
        this_cluster = df[df['cluster'] == cluster]
        if not this_cluster[this_cluster['YEAR'] == 'CV2008'].empty:
            print(this_cluster[this_cluster['YEAR'] == 'CV2008'])
            base = this_cluster[this_cluster['YEAR'] == 'CV2008']
        else:
            max_sim = this_cluster['similarity'].max()
            print(this_cluster[this_cluster['similarity'] == max_sim].head(1))
            base = this_cluster[this_cluster['similarity'] == max_sim].head(1)
        col_name = str(cluster)+' '+base['QUESTION']
        question_map[col_name] = ""
        question_map[col_name].fillna("", inplace=True)
        for idx, q in this_cluster.iterrows():
            idxx = question_map[question_map['YEAR']
                                == q['YEAR']].index.values[0]
            if question_map.iloc[idxx][col_name].values[0] == "":
                question_map.loc[question_map['YEAR'] ==
                                 q['YEAR'], col_name] = q['QUESTION']
            else:
                cnt = 1
                print("This year has duplicate questions.")
                while True:
                    print(cnt)
                    if not (q['YEAR']+f"_{cnt}") in question_map['YEAR'].values:
                        new_row = pd.DataFrame({'YEAR': [q['YEAR']+f"_{cnt}"]})
                        question_map = pd.concat(
                            [question_map, new_row], ignore_index=True)
                        question_map.fillna("", inplace=True)
                    if question_map.iloc[question_map[question_map['YEAR'] ==
                                                      q['YEAR']+f"_{cnt}"].index.values[0]][col_name].values[0] == "":
                        question_map.loc[question_map['YEAR'] ==
                                         q['YEAR']+f"_{cnt}", col_name] = q['QUESTION']
                        break
                    else:
                        cnt += 1
                    print(question_map)

        print(question_map)
    question_map.sort_values(by='YEAR', inplace=True)
    question_map.to_csv(output_path + 'type_' + str(type) +
                        '_question_map.csv', index=False)
    print("Finish type " + str(type) + ".")
    wait = input("Press Enter to continue...")
