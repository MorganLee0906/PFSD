# !pip install -r requirements.txt
import os
import pandas as pd
import openai
from openai import OpenAI
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import csv
import sys

sys.stdout.flush()

# f = open("api_key.txt", "r")
# api_key = f.read()
# openai.api_key = api_key
# f.close()

print("Modules are already loaded.")

# -----------------------------------------------------------
# Manually preprocess the data


def preprocess_data(input_path, output_path):
    similarity_list = [[['CVIII2014', 'x'], ['CIII2004', 'X'], ['CV2008', 'x'], ['CII2002', 'x'], ['CX2018', 'x']],
                       [['CVIII2014', 'a'], ['CIII2004', 'A'], [
                           'CV2008', 'a'], ['CII2002', 'a'], ['CX2018', 'a']],
                       [['CVIII2014', 'b'], ['CIII2004', 'B'], [
                           'CV2008', 'b'], ['CII2002', 'b'], ['CX2018', 'b']],
                       [['CVIII2014', 'c'], ['CIII2004', 'C'], [
                           'CV2008', 'c'], ['CII2002', 'c'], ['CX2018', 'c']],
                       [['CVIII2014', 'd'], ['CIII2004', 'D'], [
                           'CV2008', 'd'], ['CII2002', 'd'], ['CX2018', 'd']],
                       [['CVIII2014', 'e'], ['CIII2004', 'F'], [
                           'CV2008', 'e'], ['CII2002', 'f'],  ['CX2018', 'e']],
                       [['CVIII2014', 'f'], ['CIII2004', 'G'], [
                           'CV2008', 'f'], ['CII2002', 'g'], ['CX2018', 'f']],
                       [['CIII2004', 'E'], ['CII2002', 'e']],
                       [['CX2018', 'g']]]

    # 確認輸出路徑存在
    os.makedirs(output_path, exist_ok=True)

    for i, sublist in enumerate(similarity_list):
        type_name = f"type_{i+1}"
        print("Now processing: ", type_name)
        all_rows = pd.DataFrame()
        for item in sublist:
            folder_name, label = item
            csv_file = os.path.join(input_path, f"{folder_name}_label.csv")
            if os.path.exists(csv_file):
                print("Now reading: ", csv_file)
                df = pd.read_csv(csv_file)
                df = df.dropna()
                filtered_rows = df[df['type'] == label]
                all_rows = pd.concat([all_rows, filtered_rows])
                all_rows.reset_index(drop=True, inplace=True)

            else:
                print(f"File not exists: {csv_file}")

        all_rows.reset_index(drop=True, inplace=True)

        target_pkl_file = os.path.join(output_path, f"{type_name}.pkl")
        with open(target_pkl_file, 'wb') as f:
            pickle.dump(all_rows, f)
        print(f"Saved: {target_pkl_file}")
    print('Perprocessing done!')

# Clustering with GPT-4o-mini
# Read: pkl file, which contains the questions.
# Output: csv and pkl file, which contains the questions and the assigned clusters.


def extract_year(year_string):
    match = re.search(r'\d{4}', year_string)
    return int(match.group()) if match else None


def clustering_question(input_pkl_path, output_folder_path):
    print(f'Now reading {input_pkl_path}')
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # -----------------------------------------------------------
    # 第一次分類：利用CV2008年的資料進行分類
    # 提取年份
    refer_indices = data[data['YEAR'] == 'CV2008'].index
    data.loc[refer_indices, 'cluster'] = range(1, len(refer_indices)+1)
    refer_data = data.loc[refer_indices][['QUESTION', 'cluster']]

    print("1st clustering reference:")
    reference_text = "\n".join([f"Question: {row['QUESTION']}, Cluster: {
                                row['cluster']}" for _, row in refer_data.iterrows()])
    print(reference_text)

    none_indices = data[data['cluster'].isna()].index

    # 使用 GPT-4o-mini 進行分類
    for index in none_indices:
        print(f"Question data: {data.iloc[index]['QUESTION']}")
        prompt = f"""
        Based on the previous clustering data, here is the reference:
        {reference_text}
        Now, based on the following data, please:
        1. Assign a cluster number.
        2. Provide the similarity between 0 and 1.
        Format the response as: Cluster: X, Similarity: Y
        Data: {data.iloc[index]['QUESTION']}

        If the data does not fit into any existing clusters then format the response as: Cluster: 0, Similarity: 0
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                        "content": "You are an expert in clustering survey data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,  # 限制回應的長度以只包含 cluster 編號
                temperature=0.1  # 設定較低的隨機性以提高準確性
            )

            # 從 GPT 的回應中獲取預測的 Cluster 值
            generated_text = response.choices[0].message.content.strip()
            print(f"Generated cluster for index {index}: {generated_text}")

            # 使用正則表達式從生成的文本中提取 Cluster 和 Similarity 值
            cluster_match = re.search(r'Cluster:\s*(\d+)', generated_text)
            similarity_match = re.search(
                r'Similarity:\s*([0-9]*\.?[0-9]+)', generated_text)
            predicted_cluster = cluster_match.group(
                1) if cluster_match else 0
            predicted_similarity = similarity_match.group(
                1) if similarity_match else 0

            # 更新數據
            data.at[index, 'cluster'] = predicted_cluster
            data.at[index, 'similarity'] = predicted_similarity
            print(index, f"cluster:{predicted_cluster}", f"similarity:{
                predicted_similarity}", sep='\t')
        except Exception as e:
            print(f"Error while processing index {index}: {str(e)}")

    # -----------------------------------------------------------
    # 第二次分類：對於難以分類的資料，再進行一次分類
    # 將 cluster 和 similarity 轉換為數值
    data['cluster'] = pd.to_numeric(data['cluster'], errors='coerce')
    data['similarity'] = pd.to_numeric(data['similarity'], errors='coerce')

    for year in data['YEAR'].unique():
        year_data = data[data['YEAR'] == year]
        # 獲取所有 cluster 的唯一值 (除了 cluster 0)
        clusters = year_data['cluster'].dropna().unique()
        clusters = clusters[clusters != 0]
        for cluster in clusters:
            cluster_data = year_data[year_data['cluster'] == cluster]
            if len(cluster_data) > 1:
                # 獲取具有最高相似度的記錄
                max_similarity = cluster_data['similarity'].max()
                # 獲取具有最高相似度的記錄的索引
                max_similarity_indices = cluster_data[cluster_data['similarity']
                                                      == max_similarity].index
                # 獲取其他記錄的索引
                other_indices = cluster_data[~cluster_data.index.isin(
                    max_similarity_indices)].index
                # 將相似度較低的記錄設置為 cluster 0
                data.loc[other_indices, 'cluster'] = 0
                data.loc[other_indices, 'similarity'] = 0

    data['YEAR_EXTRACTED'] = data['YEAR'].apply(extract_year)
    # 處理 cluster = 0 的資料
    cluster_0_data = data[data['cluster'] == 0]
    median_year = cluster_0_data['YEAR_EXTRACTED'].median()
    min_year = cluster_0_data.loc[cluster_0_data['YEAR_EXTRACTED']
                                  < median_year, 'YEAR_EXTRACTED'].max()
    selected_data = cluster_0_data[cluster_0_data['YEAR_EXTRACTED'] == min_year]

    if selected_data.empty:
        print("沒有符合條件年份的資料")
    original_max_cluster = data['cluster'].max()

    # 將選定的資料分配到新的 cluster
    for index in enumerate(selected_data.index):
        data.at[index, 'cluster'] = data['cluster'].max() + 1

    print('第二次分類總共有', max(data['cluster']), '個cluster')
    print("2nd clustering reference:")
    refer_data = data[data['cluster'] >
                      original_max_cluster][['QUESTION', 'cluster']]
    reference_text_2 = "\n".join([f"Question: {row['QUESTION']}, Cluster: {
        row['cluster']}" for _, row in refer_data.iterrows()])
    print(reference_text_2)
    # 使用 GPT-4o-mini 進行第二次分類
    none_indices = data[data['cluster'] == 0].index
    for index in none_indices:
        print(f"Question data: {data.iloc[index]['QUESTION']}")
        prompt = f"""
        Based on the previous clustering data, here is the reference:
        {reference_text_2}
        Now, based on the following data, please:
        1. Assign a cluster number.
        2. Provide the similarity between 0 and 1.
        Format the response as: Cluster: X, Similarity: Y
        Data: {data.iloc[index]['QUESTION']}

        If the data does not fit into any existing clusters then format the response as: Cluster: 0, Similarity: 0
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                        "content": "You are an expert in clustering survey data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )

            generated_text = response.choices[0].message.content.strip()
            cluster_match = re.search(r'Cluster:\s*(\d+)', generated_text)
            similarity_match = re.search(
                r'Similarity:\s*([0-9]*\.?[0-9]+)', generated_text)
            predicted_cluster = int(
                cluster_match.group(1)) if cluster_match else 0
            predicted_similarity = float(
                similarity_match.group(1)) if similarity_match else 0

            # 更新數據
            data.at[index, 'cluster'] = predicted_cluster
            data.at[index, 'similarity'] = predicted_similarity
            print(index, "  ;cluster:", predicted_cluster,
                  "  ;similarity:", predicted_similarity)
        except Exception as e:
            print(f"Error processing index {index}: {e}")

    # Step 3: 对相同年份但相似度较低的 cluster 进行处理
    for year in data['YEAR_EXTRACTED'].unique():
        year_data = data[data['YEAR_EXTRACTED'] == year]
        clusters = year_data['cluster'].unique()
        clusters = clusters[clusters != 0]

        for cluster in clusters:
            cluster_data = year_data[year_data['cluster'] == cluster]
            if len(cluster_data) > 1:
                max_similarity = cluster_data['similarity'].max()
                max_similarity_indices = cluster_data[cluster_data['similarity']
                                                      == max_similarity].index
                other_indices = cluster_data[~cluster_data.index.isin(
                    max_similarity_indices)].index

                # 将相似度较低的设置为 cluster 0
                data.loc[other_indices, 'cluster'] = 0
                data.loc[other_indices, 'similarity'] = 0

    # 保存更新後的.pkl文件
    updated_pkl_path = os.path.join(
        output_folder_path, input_pkl_path.split('/')[-1])
    with open(updated_pkl_path, 'wb') as f:
        pickle.dump(data, f)

    # 輸出csv文件路徑
    csv_file_name = input_pkl_path.split('/')[-1].replace('.pkl', '.csv')
    csv_path = os.path.join(output_folder_path, csv_file_name)

    output_data = data[['ANSWER', 'NUMBER',
                        'QUESTION', 'YEAR', 'cluster', 'similarity']]

    output_data.to_csv(csv_path, index=False)

    print(f"分類結果已保存至 {csv_path} 和 {updated_pkl_path}")

# -----------------------------------------------------------


def keep_chinese(input_string):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5，。！？]')
    result = re.findall(chinese_pattern, input_string)
    filtered_string = ''.join(result)
    return filtered_string


def preprocess_answer(folder_path, answer_folder_path):
    for f_type in os.listdir(folder_path):
        if f_type.endswith(".csv") and not f_type.startswith("var_map"):  # Read typeX.csv
            print(f"Processing: {f_type}")
            f_type_path = os.path.join(folder_path, f_type)
            df = pd.read_csv(f_type_path)
            print(df)
            csv_filename = f'var_map_{os.path.splitext(f_type)[0]}.csv'
            csv_path = os.path.join(folder_path, csv_filename)
            with open(csv_path, 'w', encoding='utf-8', newline='') as f_varmap:  # Write var_map_typeX.csv
                fieldnames = ['ANSWER_new', 'OPTION_dict', 'OPTION_new',
                              'OPTION_pre', 'ANSWER_pre', 'YEAR', 'NUMBER']
                writer = csv.DictWriter(f_varmap, fieldnames=fieldnames)
                writer.writeheader()

                processed = []
                for idx, row in df.iterrows():
                    cluster = row['cluster']
                    if cluster in processed:
                        continue
                    print(f"Processing row: {idx}")
                    processed.append(cluster)
                    year = [row['YEAR']]
                    number = [row['NUMBER']]
                    answer_pre = [row['ANSWER']]
                    option_pre = []
                    option_new = []
                    option = []
                    for idx2, other_row in df.iterrows():
                        if other_row['cluster'] == cluster and not row.equals(other_row):
                            answer_pre.append(other_row['ANSWER'])
                            year.append(other_row['YEAR'])
                            number.append(other_row['NUMBER'])
                    for yr, answer_num in zip(year, answer_pre):
                        answerfile = os.path.join(
                            answer_folder_path, f'{yr}_answer.csv')
                        print(f"Reading: {answerfile}")
                        print(f"Answer number: {answer_num}")
                        answerdf = pd.read_csv(answerfile)
                        answerdf = answerdf.dropna()
                        option_dict = {}
                        for answer_idx, answer_row in answerdf.loc[answerdf['ANSWER'] == answer_num].iterrows():
                            print(answer_num.upper(
                            ), answer_row['YEAR'], answer_row['ANSWER'], answer_row['OPTION'])
                            option_pre.append(answer_row['OPTION'])
                            option_text = keep_chinese(answer_row['OPTION'])
                            if option_text not in option_new:
                                option_new.append(option_text)
                            option_index = option_new.index(option_text)
                            option_dict[answer_row['OPTION'].split(
                                '=')[0].strip()] = option_index
                        option.append(str(option_dict))
                    for option_index, val in enumerate(option_new):
                        option_new[option_index] = f'{option_index}={val}'
                    print(option)
                    writer.writerow({'ANSWER_new': cluster,
                                     'OPTION_dict': ', '.join(option),
                                     'OPTION_new': ', '.join(option_new),
                                     'OPTION_pre': ', '.join(option_pre),
                                     'ANSWER_pre': ', '.join(answer_pre),
                                     'YEAR': ', '.join(year),
                                     'NUMBER': ', '.join(number)})
            print(f"Saved: {csv_path}")


def merge_surveydata(question_path, output_path):
    for question_type in os.listdir(question_path):
        if question_type.endswith('.csv') and question_type.startswith(f'type_'):
            question_file = os.path.join(question_path, question_type)
            data = pd.read_csv(question_file)
            data = data.dropna()
            selected_columns = ['YEAR', 'cluster', 'QUESTION', 'NUMBER']
            data = data[selected_columns]
            data.rename(columns={'cluster': 'CLUSTER'}, inplace=True)

            data = data.sort_values(by='YEAR')
            result_data = pd.DataFrame(
                columns=['ID', 'YEAR', 'CLUSTER', 'QUESTION', 'NUMBER'])
            last_year = None
            for index, row in data.iterrows():
                current_year = row['YEAR']
                if current_year != last_year:
                    if last_year is not None:
                        survey_file.close()

                    survey_file_path = f'./data/five_years/{
                        current_year}_survey.csv'
                    survey_file = open(survey_file_path, 'r')

                    survey_file.seek(0)
                    header = next(survey_file)
                    title_mapping = {title.lower(): idx for idx, title in enumerate(
                        header.strip().split(','))}

                survey_file.seek(0)
                next(survey_file)
                num_subjects = sum(1 for line in survey_file)
                survey_file.seek(0)
                next(survey_file)
                print('受試者個數:', num_subjects)

                for i in range(num_subjects):
                    subject_data = next(survey_file).strip().split(',')
                    data_value = subject_data[title_mapping[row['NUMBER'].lower(
                    )]]
                    result_data = pd.concat([result_data, pd.DataFrame(
                        [{'ID': subject_data[0], 'YEAR': current_year, 'CLUSTER': row['CLUSTER'], 'QUESTION': row['QUESTION'], 'NUMBER': row['NUMBER'], 'ANSWER': data_value}])], ignore_index=True)

                last_year = current_year
            if last_year is not None:
                survey_file.close()
            os.makedirs(output_path, exist_ok=True)
            sorted_data = result_data.sort_values(
                by=['ID', 'CLUSTER', 'YEAR', 'NUMBER'])
            sorted_data.reset_index(drop=True, inplace=True)

            pivot_data = pd.pivot_table(sorted_data, index=[
                                        'ID', 'YEAR'], columns='CLUSTER', values='ANSWER', aggfunc=list)
            pivot_data.reset_index(inplace=True)
            pivot_data.columns.name = None
            sorted_data = pivot_data.sort_values(by=['ID', 'YEAR'])
            sorted_data.reset_index(drop=True, inplace=True)

            result_path = os.path.join(
                output_path, f'{question_type.split(".")[0]}_survey_data_processed.csv')
            sorted_data.to_csv(result_path, index=False)
            print(f'Saved: {result_path}')


# preprocess_data('data/five_years/', 'data/five_years/five_years_by_type')
# for i in range(1, 10):
#     clustering_question(f'data/five_years/five_years_by_type/type_{i}.pkl',
#                         'data/five_years/five_years_by_type')
# preprocess_answer('data/five_years/five_years_by_type', 'data/five_years')
merge_surveydata('data/five_years/five_years_by_type',
                 'data/five_years/five_years_survey_data')
