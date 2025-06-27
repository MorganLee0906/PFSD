import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import string
import openai
from merge_results import merge_results
from fetching_pickle import drop_paired

f = open("api_key.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()
model = "gpt-4o-mini-2024-07-18"

sys_prompt = """
You are an expert in survey question matching. Input is a seed group and its similar-group block, each line like "PSID: 問題文字 [TEXT]" or "[]". ONLY Identify items with “[]” across different years that are semantically identical. Output strictly in lines of format:
YEAR-NUMBER=YEAR-NUMBER
例如:
RII2000-b14c=RII2001-C03A02
如果一個 seed 子題對到多個年份，用 "=" 串接：
RII2000-b14c=RII2001-C03A02=RR2002-d14c
只輸出如上格式，多行即可，不要多餘文字。若無任何配對，不要輸出任何文字或符號。
  
# 範例輸入/範例輸出：
## 範例 1（有一對）
Input:
Year: RII2000, Pset: b14, Label: {'工作'}
b14a: 問題1 [Y]
b14b: 問題2 []
=== similar groups ===
Found 1 similar groups:
RII2001 B14 {'工作'} (similarity: 0.98)
B14A: 問題1 [Y]
B14B: 問題2 []
Output:
RII2000-b14b=RII2001-B14B

## 範例 2（多對）
Input:
Year: RII2000, Pset: x01, Label: {...}
x01a: Q1 []
x01b: Q2 []
=== similar groups ===
Found 2 similar groups:
RII2001 X01 {...}
X01A: Q1 []
X01B: Q2 []
RR2002 X02 {...}
X02A: Q1 []
X02B: Q2 []
Output:
RII2000-x01a=RII2001-X01A=RR2002-X02A
RII2000-x01b=RII2001-X01B=RR2002-X02B

## 範例 3（無配對）
Input: …（only [Y] 或無相似）
Output:
"""
 
def cluster_by_embedding(df):
    df = df.dropna(subset=["YEAR", "QUESTION", "LABEL"]).reset_index(drop=True)
    df["dup_same_year"] = ""
    df["paired"] = ""

    for year in df["YEAR"].unique():
        print(f"Processing year: {year}")
        idx_list = df.index[df["YEAR"] == year].tolist()
        if len(idx_list) < 2:         # 只有 0 / 1 題就不用比
            continue


        # 取出該年份所有 embedding
        emb_mat = np.stack(df.loc[idx_list, "embedding"].values)

        # 建最近鄰模型（cosine 距離）
        nbrs = NearestNeighbors(metric="cosine").fit(emb_mat)
        dist_mat, ind_mat = nbrs.kneighbors(emb_mat, n_neighbors=len(idx_list))

        # 逐題檢查：若與同年另一題相似度 > 0.95，雙方都標記為 "Y"
        for row_pos, global_idx in enumerate(idx_list):
            for nei_pos, d in zip(ind_mat[row_pos][1:], dist_mat[row_pos][1:]):  # [1:] 跳過自己
                sim = 1 - d
                if sim > 0.95:
                    #print(f"Found similar questions: {df.at[global_idx, 'QUESTION']} and {df.at[idx_list[nei_pos], 'QUESTION']} with similarity {sim:.4f}")
                    df.at[global_idx, "dup_same_year"] = "Y"
                    df.at[idx_list[nei_pos], "dup_same_year"] = "Y"

    # ────────────────────────────────────────────────────────────────
    # Step 2: cross‑year comparison for *non‑duplicate* questions
    # ────────────────────────────────────────────────────────────────
    print("\n=== Building cross‑year compare_table for non‑duplicate questions ===")

    # 1) 只保留沒有被標記為同年重複的題目
    clean_df = df[df["dup_same_year"] != "Y"].reset_index()      # 保留原 index 在 `index` 欄

    if clean_df.empty:
        print("No eligible questions found. Exiting.")
        exit(0)

    # 2) 準備向量與 NN 模型
    emb_mat_all = np.stack(clean_df["embedding"].values)
    nbrs_all = NearestNeighbors(metric="cosine").fit(emb_mat_all)

    visited_global_idx = set()     # 用原 df index 追蹤
    rows = []

    # 3) 逐題擴散找『跨年度相似度 > 0.95』且年份互不重複的群組
    for local_idx, row in clean_df.iterrows():
        global_idx = row["index"]          # 在 df 的原 index
        if global_idx in visited_global_idx:
            continue

        # 取鄰居 (包含自己 n_neighbors = len(clean_df) 時也 OK)
        k = len(clean_df)         # 安全上限
        dist, ind = nbrs_all.kneighbors(emb_mat_all[local_idx:local_idx+1], n_neighbors=k)
        ind, dist = ind[0][1:], dist[0][1:]     # 去掉自己

        group_local_idx = [local_idx]
        years_in_group = {row["YEAR"]}

        # 收集顯示資訊：先放 seed，本身 similarity 設 1.0
        group_info = [{
            "YEAR": row["YEAR"],
            "NUMBER": row["NUMBER"],
            "QUESTION": row["QUESTION"],
            "similarity": 1.00
        }]

        for j, d in zip(ind, dist):
            sim = 1 - d
            if sim <= 0.95:
                break                      # 之後相似度只會更低，直接跳出
            yr = clean_df.at[j, "YEAR"]
            if yr in years_in_group:
                continue                   # 年份已在群組，跳過
            group_local_idx.append(j)
            years_in_group.add(yr)
            # 記錄這個鄰居的資訊
            group_info.append({
                "YEAR": yr,
                "NUMBER": clean_df.at[j, "NUMBER"],
                "QUESTION": clean_df.at[j, "QUESTION"],
                "similarity": round(sim, 4)
            })

        if len(group_local_idx) == 1:
            continue                       # 找不到跨年相似題，略過
        #else:
            # 輸出找到的群組明細
            #print("\n--- Similarity group found ---")
            #print(pd.DataFrame(group_info)[["YEAR", "NUMBER", "QUESTION", "similarity"]])
        df.at[global_idx, "paired"] = "Y"  # 標記已配對
        # 標記 visited，避免重複分組
        visited_global_idx.update(clean_df.loc[group_local_idx, "index"])

        # --- 組成 compare_table 的 row (格式: NUMBER/QUESTION) ---
        row_dict = {"question": f"{row['QUESTION']}"}

        for k_idx in group_local_idx:
            yr_key = str(clean_df.at[k_idx, "YEAR"])
            num  = clean_df.at[k_idx, "NUMBER"]
            qtxt = clean_df.at[k_idx, "QUESTION"]
            row_dict[yr_key] = f"{num}/{qtxt}"
            orig_idx = clean_df.at[k_idx, "index"]        # 原 dataframe 的 index
            df.at[orig_idx, "paired"] = "Y"               # 正確標記已配對

        rows.append(row_dict)

    if rows:
        compare_table = pd.DataFrame(rows).dropna(axis=1, how="all")  # 去掉全空的年份列
        print(f"Before merging, compare_table has {len(compare_table)} rows and {compare_table.shape[1]} columns (years).")
        compare_table = merge_results(compare_table)  # 合併結果
        print(f"After merging, compare_table has {len(compare_table)} rows and {compare_table.shape[1]} columns (years).")
        # 儲存結果
        compare_table.to_csv("compare_table_embedding.csv", encoding="utf-8-sig", index=False)
        print(f"compare_table saved. Rows: {len(compare_table)}, Columns (years): {compare_table.shape[1]-1}")
    else:
        print("No cross‑year matches found with similarity > 0.95.")
    return df

"""
df = pd.read_pickle("questions_with_emb.pkl")
df = cluster_by_embedding(df)
df.to_pickle("first_stage.pkl")
df.drop(columns = "embedding").to_csv("first_stage.csv", encoding="utf-8-sig", index=False)
"""

"""
Second stage: Cross-year similarity matching by GPT-4o-mini
"""

print("Reading first stage data...")
df = pd.read_pickle("second_stage.pkl")
df = df.fillna('')
print(df.shape)
print(df)
# Step 2b: 找出「pset 在同一年相似度高」的題組，
#          再去其他年份尋找相似度 > 0.95 的 *pset 群*，僅列印結果
# ────────────────────────────────────────────────────────────────
print("\n=== Cross‑year similar pset groups ===")

unpaired_idx = df.index[(df["paired"] != "Y") & (df["paired"] != "NOT FOUND") & (df["paired"] != "GPT")].tolist()
visited_group = set()
wait = input(f"Unpaired Problem Set: {len(unpaired_idx)}. Press Enter to continue...")
# ---------- 準備每個 (YEAR, pset) 的索引清單與平均向量 ----------
group_idx   = {}
group_emb   = {}

for (yr, ps), sub in df.groupby(["YEAR", "pset"]):
    idx_list = sub.index.tolist()
    emb_mean = np.mean(np.stack(sub["embedding"].values), axis=0)
    group_idx[(yr, ps)] = idx_list
    group_emb[(yr, ps)] = emb_mean

for local_idx, row in df.loc[unpaired_idx].iterrows():
    year = row["YEAR"]
    pset = row["pset"]
    if (year, pset) not in visited_group:
        visited_group.add((year, pset))
        print("Adding group for year", year, "pset", pset)

visited_group = sorted(visited_group, key=lambda x: (x[0], x[1]))  # 按年份和 pset 排序
rows = []

asking_count = 0
error_count = 0

for year, pset in visited_group:
    sub_df = df[(df["YEAR"] == year) & (df["pset"] == pset)]
    prompt_lst = []
    print(f"\nYear: {year}, Pset: {pset}, Label: {set(sub_df['LABEL'].to_list())}")
    prompt_lst.append(f"Year: {year}, Pset: {pset}, Label: {set(sub_df['LABEL'].to_list())}")
    for local_idx, row in sub_df.iterrows():
        print(f"{row['NUMBER']}: {row['QUESTION']} [{row['paired']}]")
        prompt_lst.append(f"{row['NUMBER']}: {row['QUESTION']} [{row['paired']}]")
    emb = group_emb[(year, pset)]
    found = False
    sim_grp = set()
    for (other_year, other_pset), other_emb in group_emb.items():
        if other_year == year:
            continue
        sim = 1 - distance.cosine(emb, other_emb)
        if sim > 0.9:
            found = True
            sim_grp.add((other_year, other_pset, sim))
    if not found:
        print("No similar groups found.")
        sub_df = df[(df["YEAR"] == year) & (df["pset"] == pset)]
        for local_idx, row in sub_df.iterrows():
            row_dict = {
                "question": row["QUESTION"]
            }
            row_dict[year] = f"{row['NUMBER']}/{row['QUESTION']}"
            rows.append(row_dict)
            df.loc[(df["YEAR"] == row['YEAR']) & (df["NUMBER"] == row['NUMBER']), "paired"] = "NOT FOUND"        
        continue
    else:
        visited_year = set()
        sim_grp = sorted(sim_grp, key=lambda x: (x[0], -x[2]))
        print("=== similar groups ===")
        prompt_lst.append("=== similar groups ===")
        print(f"Found {len(sim_grp)} similar groups:")
        for other_year, other_pset, sim in sim_grp:
            if other_year in visited_year:
                continue
            visited_year.add(other_year)
            sub_df = df[(df["YEAR"] == other_year) & (df["pset"] == other_pset)]
            print(f"{other_year} {other_pset} {set(sub_df['LABEL'].to_list())} (similarity: {sim:.4f})")
            prompt_lst.append(f"{other_year} {other_pset} {set(sub_df['LABEL'].to_list())} (similarity: {sim:.4f})")
            for local_idx, row in sub_df.iterrows():
                print(f"{row['NUMBER']}: {row['QUESTION']} [{row['paired']}]")
                prompt_lst.append(f"{row['NUMBER']}: {row['QUESTION']} [{row['paired']}]")
        # Ask GPT-4o-mini to summarize the findings
        prompt = "\n".join(prompt_lst)
        print("\nAsking GPT-4o-mini to summarize the findings...")
        count = 1
        visited = set()
        for _ in range(count):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=700
                )
                asking_count += 1
                # Parse GPT response for mappings of form YEAR-NUMBER=YEAR-NUMBER
                content = response.choices[0].message.content.strip()
                print("GPT response:")
                print(content)
                content = content.split("\n")
                content = [line.strip() for line in content if line.strip()]
                for line in content:
                    if "=" in line:
                        year_num_pair = line.split("=")
                        if len(year_num_pair) >= 2:
                            row_dict = {}
                            for i in range(len(year_num_pair)):
                                year, num = year_num_pair[i].split("-")
                                q_text = df[(df["YEAR"] == year) & (df["NUMBER"] == num)]["QUESTION"].values[0]
                                visited.add((year, num))
                                if i == 0:
                                    print(f"Creating: {year}-{num} = {q_text}")
                                    row_dict = {"question": q_text}
                                row_dict[year] = f"{num}/{q_text}"
                            rows.append(row_dict)
                            # wait = input(f"Mapping found: {row_dict}")
                                
                        else:
                            print(f"Invalid mapping format: {line}")
                    else:
                        print(f"Skipping non-mapping line: {line}")
                break  
            except Exception as e:
                print(f"Error processing GPT response: {e}")
                print("Response content:", content)
                # wait = input("Press Enter to continue...")
                error_count += 1
        for (yr, num) in visited:
            df.loc[(df["YEAR"] == yr) & (df["NUMBER"] == num), "paired"] = "GPT"
compare_table_2 = pd.DataFrame(rows)
compare_table = pd.read_csv("compare_table.csv", encoding="utf-8-sig")
compare_table = pd.concat([compare_table, compare_table_2], axis=0, ignore_index=True)
compare_table = merge_results(compare_table)
compare_table.to_csv("compare_table.csv", encoding="utf-8-sig", index=False)
df.to_pickle("second_stage.pkl")
df.drop(columns=["embedding"]).to_csv("questions_with_emb_checked.csv", encoding="utf-8-sig", index=False)

print(f"Total questions asked: {asking_count}, Errors encountered: {error_count}")