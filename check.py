import pandas as pd
import os
tp = input("Enter the type of the file: ")
csv = pd.read_csv(f'data/five_years/five_years_by_type/type_{tp}.csv')
sorted_csv = csv.sort_values(by='cluster')
sorted_csv["check"] = ""
max_cluster = int(sorted_csv['cluster'].max()) + 1
i = 1
while i < max_cluster:
    os.system('clear')
    print(f"\033[33m Cluster {i}/{max_cluster}", sorted_csv[(sorted_csv['cluster'] == i) & (
        sorted_csv['YEAR'] == 'CV2008')]['QUESTION'].values, '\033[0m ')
    idx = sorted_csv[sorted_csv['cluster'] == i].index
    for j in idx:
        print(j, sorted_csv.at[j, 'QUESTION'], sep='\t')
        similar = str(sorted_csv.at[j, 'matched'])
        similar = similar if similar != 'nan' else ''
        sim_list = list(map(float, similar.split(','))
                        ) if similar != '' else []
        for k in sim_list:
            print('\t', k, sorted_csv[(sorted_csv['cluster'] ==
                  k) & (sorted_csv['YEAR'] == 'CV2008')]['QUESTION'].values, sep='\t')
    try:
        check = input(
            'Enter the index and its correct cluster if incorrect, or leaves empty: ')
        while check != '' and check != 'exit':
            chk = check.split(' ')
            if len(chk) == 2:
                if chk[0] == 'j':
                    i = int(chk[1])
                    break
                elif chk[0] == 'm':
                    for j in idx:
                        sorted_csv.at[j, 'cluster'] = int(chk[1])
                        i = int(chk[1])
                    break
                else:
                    sorted_csv.at[int(chk[0]), 'cluster'] = int(chk[1])
            elif len(chk) == 1:
                if chk[0] == 'n':
                    max_cluster += 1
                if chk[0] == 'exit':
                    break

            idx = sorted_csv[sorted_csv['cluster'] == i].index
            os.system('clear')
            print(f"\033[33m Cluster {i}/{max_cluster}", sorted_csv[(sorted_csv['cluster'] == i) & (
                sorted_csv['YEAR'] == 'CV2008')]['QUESTION'].values, '\033[0m ')
            idx = sorted_csv[sorted_csv['cluster'] == i].index
            for j in idx:
                print(j, sorted_csv.at[j, 'QUESTION'], sep='\t')
                similar = str(sorted_csv.at[j, 'matched'])
                similar = similar if similar != 'nan' else ''
                sim_list = list(map(float, similar.split(','))
                                ) if similar != '' else []
                for k in sim_list:
                    print('\t', k, sorted_csv[(sorted_csv['cluster'] ==
                                               k) & (sorted_csv['YEAR'] == 'CV2008')]['QUESTION'].values, sep='\t')
            check = input(
                'Enter the index and its correct cluster if incorrect, or leaves empty: ')
        if check == 'exit':
            break
        i += 1
    except:
        print("Invalid input")

sorted_csv.to_csv(
    f'data/five_years/five_years_by_type/type_{tp}.csv', index=False)
print("Done")
