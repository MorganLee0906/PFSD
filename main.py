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

f = open("api_test.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()

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
    w = open(f"record_{input_pkl_path.split('/')
             [-1].replace('.pkl', '.txt')}", 'w')
    w.write(f'Now reading {input_pkl_path}')
    print(f'Now reading {input_pkl_path}')
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)

    refer_indices = data[data['YEAR'] == 'CV2008'].index
    data.loc[refer_indices, 'cluster'] = range(1, len(refer_indices)+1)
    data["matched"] = None
    refer_data = data.loc[refer_indices][['QUESTION', 'cluster']]
    max_cluster = len(refer_indices)

    print("1st clustering reference:")
    w.write("1st clustering reference:")
    reference_text = "\n".join([f"Question: {row['QUESTION']}, Cluster: {
                                row['cluster']}" for _, row in refer_data.iterrows()])
    print(reference_text)
    w.write(reference_text)
    print("-----------------------------------")
    w.write("-----------------------------------\n")

    none_indices = data[data['cluster'].isna()].index

    # 使用 GPT-4o-mini 進行分類
    for index in none_indices:
        if data.iloc[index]['YEAR'] == 'CV2008':
            continue
        print("Processing index: ", index, "Year:", data.iloc[index]['YEAR'])
        w.write(f"Processing index: {index} Year: {
                data.iloc[index]['YEAR']}\n")
        prompt = f"""
        Using the reference clustering data below:
        {reference_text}

        For the given input question, follow these steps:
            1.	Analyze the question – Identify its key features and compare them to the reference clusters.
            2.	Determine the primary cluster – Assign the closest matching cluster (if similarity ≥ 0.8). If no match, create a new cluster with similarity 0.
            3.	Find similar clusters – List other clusters with moderate to high similarity (≥ 0.5).
            4.	Format the output as:
        Cluster: X, Similarity: Y, Similar with: [Z0, Z1, …]

        Input Data: {data.iloc[index]['QUESTION']}
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                        "content": '''You are an expert in data clustering, specializing in categorizing survey questions based on nuanced semantic differences. Your tasks:
                                        1.	Match questions to existing clusters based on context and intent.
                                        2.	Differentiate similar wording with distinct meanings (e.g., 'planned' vs. 'confirmed') and account for subject differences (e.g., 'father' vs. 'mother').
                                        3.	If a question fits multiple clusters, select the most appropriate one with clear reasoning.
                                        4.	If no sufficient match exists, assign a new unique cluster number.
                                        5.	Provide a similarity score (0-1) for the primary classification (1 = perfect match).
                                        6.	List other similar clusters (only their numbers, no similarity scores).
                                        7.	STRICTLY follow the required output format.'''},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=40,  # 限制回應的長度以只包含 cluster 編號
                temperature=0.1  # 設定較低的隨機性以提高準確性
            )

            # 從 GPT 的回應中獲取預測的 Cluster 值
            generated_text = response.choices[0].message.content.strip()

            # 使用正則表達式從生成的文本中提取 Cluster 和 Similarity 值
            cluster_match = re.search(r'Cluster:\s*(\d+)', generated_text)
            similarity_match = re.search(
                r'Similarity:\s*([0-9]*\.?[0-9]+)', generated_text)
            match = re.search(r'Similar with: \[(.*?)\]', generated_text)
            predicted_matched = match.group(1).split(', ') if match else []
            predicted_cluster = cluster_match.group(1) if cluster_match else 0
            predicted_similarity = similarity_match.group(
                1) if similarity_match else 0

            print(f"Reply for question {
                  data.iloc[index]['QUESTION']}:\n", generated_text)
            w.write(f"Reply for index {
                    data.iloc[index]['QUESTION']}\n: {generated_text}\n")
            print("Predicted Cluster:", predicted_cluster,
                  "Predicted Similarity:", predicted_similarity)
            w.write(f"Predicted Cluster: {predicted_cluster}, Predicted Similarity: {
                    predicted_similarity}\n")
            if predicted_matched:
                print("Predicted Matched:", predicted_matched)
                w.write(f"Predicted Matched: {predicted_matched}\n")
            # 更新數據
            data.at[index, 'cluster'] = predicted_cluster
            data.at[index, 'similarity'] = predicted_similarity
            data.at[index, 'matched'] = predicted_matched
            if int(predicted_cluster) > max_cluster:
                max_cluster = int(predicted_cluster)
                reference_text += f"\nQuestion: {
                    data.iloc[index]['QUESTION']}, Cluster: {predicted_cluster}"
                print("Updated reference clustering data:", f"Question: {
                      data.iloc[index]['QUESTION']}, Cluster: {predicted_cluster}")
                w.write("Updated reference clustering data:", f"Question: {
                        data.iloc[index]['QUESTION']}, Cluster: {predicted_cluster}\n")
        except Exception as e:
            print(f"Error while processing index {index}: {str(e)}")
            w.write(f"Error while processing index {index}: {str(e)}")

    fcsv_file_name = input_pkl_path.split(
        '/')[-1].replace('.pkl', '.csv')
    fcsv_path = os.path.join(output_folder_path, fcsv_file_name)
    foutput_data = data[['ANSWER', 'NUMBER',
                        'QUESTION', 'YEAR', 'cluster', 'similarity', 'matched']]
    foutput_data.to_csv(fcsv_path, index=False)
    print(f"Saved: {fcsv_path}")


def clustering_method2(input_csv_path, output_folder_path):
    w = open(f"record_{input_csv_path.split('/')
             [-1].replace('.csv', '.txt')}", 'w')
    w.write(f'Now reading {input_csv_path}')
    print(f'Now reading {input_csv_path}')
    data = pd.read_csv(input_csv_path)
    cnt = 1
    for c in data['category'].unique():
        base = 'CV2008'
        if base not in data[data['category'] == c]['YEAR'].values:
            print(f"Category {c} not found in CV2008")
            base = str(
                min(map(extract_year, data[data['category'] == c]['YEAR'])))
            base = data[data['YEAR'].str.contains(base)]['YEAR'].values[0]
            print("New base year:", base)
        else:
            print("Found CV2008")
        print(f"Now processing category: {c}")
        refer_indices = data[(data['YEAR'] == base) &
                             (data['category'] == c)].index
        for idx in refer_indices:
            if data.loc[idx]['cluster'] == 0:
                data.loc[idx, 'cluster'] = data['cluster'].max() + 1
            data.loc[idx, 'similarity'] = 1
        refer_data = data.loc[refer_indices][['QUESTION', 'cluster']]

        print("1st clustering reference:")
        w.write("1st clustering reference:\n")
        reference_text = "\n".join([f"Question: {row['QUESTION']}, Cluster: {
                                    row['cluster']}" for _, row in refer_data.iterrows()])
        print(reference_text)
        w.write(reference_text+'\n')
        print("-----------------------------------")
        w.write("-----------------------------------\n")

        none_indices = data[(data['YEAR'] != base) & (
            data['category'] == c) & (data['similarity'] != 1)].index
        print(data.loc[none_indices]['QUESTION'].to_list())
        # wait = input('Press Enter to continue...')
        # 使用 GPT-4o-mini 進行分類
        for index in none_indices:
            data.loc[index, 'cluster'] = 0
            data.loc[index, 'similarity'] = 0
            if data.iloc[index]['YEAR'] == base:
                continue
            print("Processing index: ", index,
                  "Year:", data.iloc[index]['YEAR'])
            w.write(f"Processing index: {index} Year: {
                    data.iloc[index]['YEAR']}\n")
            prompt = f"""
            Using the reference clustering data below:
            {reference_text}
            For the given input question, follow these steps:
                1.	Analyze the question – Identify its key features and compare them to the reference clusters.
                2.	Determine the primary cluster – Assign the closest matching cluster (if similarity ≥ 0.8). If no match, create a new cluster with similarity 0.
            Input Data: {data.iloc[index]['QUESTION']}
            """

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                            "content": '''你是一個專門合併不同年度之問卷題目的專家，你的任務是相同的問題分類在一起。你需要：
                                        1. 分析問題 - 若兩個不同的題目在問同一件事，則分類到相同的群組中。
                                        2. 確定主要群組 - 將最接近的匹配群組分配給問題（如果相似度≥ 0.8）。如果沒有匹配，則創建一個新的群組，相似度為0。
                                        3. 格式化輸出如下，不准輸出與格式無關的文字：Cluster: X, Similarity: Y
                                        4. 如果問題與多個群組匹配，請選擇最合適的群組。
                                        5. 若題目敘述不完整，請依照題目內的主詞進行分類。
                                '''},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,  # 限制回應的長度以只包含 cluster 編號
                    temperature=0.1  # 設定較低的隨機性以提高準確性
                )

                # 從 GPT 的回應中獲取預測的 Cluster 值
                generated_text = response.choices[0].message.content.strip()

                # 使用正則表達式從生成的文本中提取 Cluster 和 Similarity 值
                cluster_match = re.search(r'Cluster:\s*(\d+)', generated_text)
                similarity_match = re.search(
                    r'Similarity:\s*([0-9]*\.?[0-9]+)', generated_text)
                predicted_cluster = cluster_match.group(
                    1) if cluster_match else 0
                predicted_similarity = similarity_match.group(
                    1) if similarity_match else 0
                predicted_similarity = float(predicted_similarity)
                if predicted_similarity < 0.9:
                    predicted_cluster = 0
                    predicted_similarity = 0
                print(f"Reply for question {
                    data.loc[index]['QUESTION']}:\n", generated_text)
                w.write(
                    f"Reply for index {data.loc[index]['QUESTION']}\n: {generated_text}\n")
                print("Predicted Cluster:", predicted_cluster,
                      "Predicted Similarity:", predicted_similarity)
                w.write(
                    f"Predicted Cluster: {predicted_cluster}, Predicted Similarity: {predicted_similarity}\n")
                # 更新數據
                if int(predicted_cluster) not in data[data['category'] == c]['cluster'].values or int(predicted_cluster) == 0:
                    print(predicted_cluster, data['cluster'].values)
                    predicted_cluster = data['cluster'].max() + 1
                    reference_text += f"\nQuestion: {
                        data.loc[index]['QUESTION']}, Cluster: {predicted_cluster}"
                    print("Updated reference clustering data:", f"Question: {
                        data.loc[index]['QUESTION']}, Cluster: {predicted_cluster}")
                    w.write(
                        f"Updated reference clustering data: Question: {data.loc[index]['QUESTION']}, Cluster: {predicted_cluster}\n")
                data.at[index, 'cluster'] = int(predicted_cluster)
                data.at[index, 'similarity'] = predicted_similarity
            except Exception as e:
                print(f"Error while processing index {index}: {str(e)}")
                w.write(f"Error while processing index {index}: {str(e)}")

        fcsv_file_name = input_csv_path.split(
            '/')[-1].replace('.csv', '.csv')
        fcsv_path = os.path.join(output_folder_path, fcsv_file_name)
        foutput_data = data[['ANSWER', 'NUMBER',
                            'QUESTION', 'YEAR', 'cluster', 'similarity', 'category']]
        foutput_data.sort_values(by='category', inplace=True)
        foutput_data.to_csv(fcsv_path, index=False)
        print(f"Saved: {fcsv_path}")


def keep_chinese(input_string):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5，。！？]')
    result = re.findall(chinese_pattern, input_string)
    filtered_string = ''.join(result)
    return filtered_string


def preprocess_answer(folder_path, answer_folder_path):
    for f_type in os.listdir(folder_path):
        if f_type.endswith("3.csv") and not f_type.startswith("var_map"):  # Read typeX.csv
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
        if question_type.endswith('.csv') and question_type.startswith(f'type_3'):
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


def convert_option(survey_path, var_map_path, type):
    survey_file_path = os.path.join(
        survey_path, f'{type}_survey_data_processed.csv')
    survey_data = pd.read_csv(survey_file_path)
    var_map_file_path = os.path.join(var_map_path, f'var_map_type_{type}.csv')
    var_map_data = pd.read_csv(var_map_file_path)
    survey_data = sorted(survey_data, key=lambda x: x['ID'])


# preprocess_data('data/five_years/', 'data/five_years/five_years_by_type')
# for i in range(1, 10):
# tp = input("Please enter the type number: ")
# clustering_method2(f'data/five_years/five_years_by_type/type_{tp}_sim_embed.csv',
#                   'data/five_years/five_years_by_type')

#     clustering_question(f'data/five_years/five_years_by_type/type_{i}.pkl',
#                         'data/five_years/five_years_by_type')
# preprocess_answer('data/five_years/five_years_by_type', 'data/five_years')
merge_surveydata('data/five_years/five_years_by_type',
                 'data/five_years/five_years_survey_data')
clustering_question('data/five_years/five_years_by_type/type_2.pkl',
                    'data/five_years/five_years_by_type')
