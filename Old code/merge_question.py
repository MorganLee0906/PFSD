import os
import openai
import re
import pandas as pd
from scipy.spatial.distance import cosine

f = open("api_test.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()

embedding_path = 'data/five_years/five_years_with_embeddings'
all_embedding = pd.DataFrame()
for e in os.listdir(embedding_path):
    if not e.endswith('.pkl'):
        continue
    pkl = pd.read_pickle(embedding_path + '/' + e)
    all_embedding = pd.concat(
        [all_embedding, pkl[['YEAR', 'NUMBER', 'QUESTION', 'Q_embedding']]])
dir_path = 'data/five_years'
all_df = pd.DataFrame()
for file in os.listdir(dir_path):
    if file.endswith('_label.csv'):
        year = re.findall(r'\d+', file)[0]
        if year == '2014' or year == '2018':
            continue
        df = pd.read_csv(os.path.join(dir_path, file))
        df['year'] = year
        df['embedding'] = ''
        df.dropna(subset=['QUESTION'], inplace=True)
        for _, row in df.iterrows():
            try:
                row['embedding'] = all_embedding.loc[(
                    all_embedding['YEAR'] == row['YEAR']) & (all_embedding['QUESTION'] == row['QUESTION'])]['Q_embedding'].values[0]
            except:
                print("Error:", row['YEAR'], row['QUESTION'])
        all_df = pd.concat(
            [all_df, df[['year', 'LABEL', 'QUESTION', 'NUMBER', 'ANSWER', 'embedding']]], ignore_index=True)
        print("Read:", file)
all_df.sort_values(['year', 'NUMBER'], inplace=True, ignore_index=True)

all_df['cluster_type'] = ''
all_df['cluster_num'] = ''
all_df['similarity'] = 0
print(all_df)

categories_cnt = 0
for category in all_df['LABEL'].unique():
    print("Category:", category)
    cluster_cnt = 1
    base_year = max(all_df[all_df['LABEL'] == category]['year'].values)
    print("Base year:", base_year)
    refer_idx = all_df[(all_df['year'] == base_year) & (
        all_df['LABEL'] == category)].index.tolist()
    refer_text = ""
    for idx in refer_idx:
        all_df.loc[idx, 'cluster_type'] = chr(ord('a')+categories_cnt)
        all_df.loc[idx, 'cluster_num'] = cluster_cnt
        cluster_cnt += 1
        refer_text += f"Question: {all_df.loc[idx, 'QUESTION']}, Cluster: {
            all_df.loc[idx, 'cluster_num']}\n"
    print("Refer text:", refer_text)

    # Step 1: Use embeddings to cluster questions
    cmp_idx = all_df[(all_df['year'] != base_year) & (
        all_df['LABEL'] == category)].index.tolist()
    for idx in refer_idx:
        for cidx in cmp_idx:
            if cosine(all_df.loc[idx, 'embedding'], all_df.loc[cidx, 'embedding']) == 0:
                all_df.loc[cidx, 'cluster_type'] = all_df.loc[idx,
                                                              'cluster_type']
                all_df.loc[cidx, 'cluster_num'] = int(
                    all_df.loc[idx, 'cluster_num'])
                all_df.loc[cidx, 'similarity'] = 1

    # Step 2: Use GPT-4o to cluster questions
    for idx in cmp_idx:
        if all_df.loc[idx, 'cluster_num'] != '':
            continue
        prompt = f"""
            Using the reference clustering data below:
            {refer_text}
            For the given input question, follow these steps:
                1.	Analyze the question – Identify its key features and compare them to the reference clusters.
                2.	Determine the primary cluster – Assign the closest matching cluster (if similarity ≥ 0.8). If no match, create a new cluster with similarity 0.
                
            Input Data: {all_df.loc[idx, 'QUESTION']}
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
                                    5. 若題目敘述不完整，請依照題目內出現的資訊進行分類。
                                    6. 務必區分問題內的主詞、動詞、副詞的差異，若有不同必須分類到不同群組。
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
            if predicted_similarity < 0.8:
                predicted_cluster = 0
                predicted_similarity = 0
            print(f"Reply for question {
                all_df.loc[idx, 'QUESTION']}:\n", generated_text)
            print("Predicted Cluster:", predicted_cluster,
                  "Predicted Similarity:", predicted_similarity)

            # check whether there has same year in the cluster
            for chkidx in all_df[(all_df['cluster_type'] == chr(ord('a')+categories_cnt)) & (all_df['cluster_num'] == int(predicted_cluster))].index.tolist():
                if all_df.loc[chkidx, 'year'] == all_df.loc[idx, 'year']:
                    print("Same year in the cluster")
                    print("Question:", all_df.loc[chkidx, 'QUESTION'], "Year:",
                          all_df.loc[chkidx, 'year'], "Similarity:", all_df.loc[chkidx, 'similarity'])
                    print("Question:", all_df.loc[idx, 'QUESTION'], "Year:",
                          all_df.loc[idx, 'year'], "Similarity:", predicted_similarity)
                    if all_df.loc[chkidx, 'similarity'] > predicted_similarity:
                        predicted_cluster = 0
                        predicted_similarity = 1
                        break
            if int(predicted_cluster) not in all_df[all_df['cluster_type'] == chr(ord('a')+categories_cnt)]['cluster_num'].values or int(predicted_cluster) == 0:
                print(predicted_cluster, all_df[all_df['cluster_type'] == chr(
                    ord('a')+categories_cnt)]['cluster_num'].values)

                predicted_cluster = all_df[all_df['cluster_type'] == chr(
                    ord('a')+categories_cnt)]['cluster_num'].max() + 1
                refer_text += f"\nQuestion: {
                    all_df.loc[idx, 'QUESTION']}, Cluster: {predicted_cluster}"
                print("Updated reference clustering data:", f"Question: {
                    all_df.loc[idx, 'QUESTION']}, Cluster: {predicted_cluster}")
                predicted_similarity = 1
            all_df.loc[idx, 'cluster_type'] = chr(ord('a')+categories_cnt)
            all_df.loc[idx, 'cluster_num'] = int(predicted_cluster)
            all_df.loc[idx, 'similarity'] = predicted_similarity
        except Exception as e:
            wait = input(f"Error while processing index {idx}: {str(e)}")
    # check the result
    for cluster in all_df[all_df['cluster_type'] == chr(ord('a')+categories_cnt)]['cluster_num'].unique():
        print("Cluster:", cluster)
        print(all_df[(all_df['cluster_type'] == chr(ord('a')+categories_cnt)) & (
            all_df['cluster_num'] == cluster)]['QUESTION'])
        wait = input("=====================================")
    all_df.to_csv('data/all_clustered.csv', index=False)

    categories_cnt += 1

all_df.to_csv('data/all_clustered.csv', index=False)
