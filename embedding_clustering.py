import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
import numpy as np
import string

'''
path = './data/main_sample/'
all_years = pd.DataFrame()
for file in os.listdir(path):
    if file.endswith('label.csv'):
        if file.startswith('RIII2001'):
            continue
        print(f"Loading {file}...")
        df = pd.read_csv(os.path.join(path, file))
        all_years = pd.concat([all_years, df[["YEAR","NUMBER","QUESTION","ANSWER","pset","LABEL"]]], ignore_index=True)
print("Data loaded successfully.")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

df = all_years.copy()

emb_tensor = model.encode(
    df["QUESTION"].tolist(),
    convert_to_tensor=True,
    batch_size=128,
    show_progress_bar=True
)
df["embedding"] = emb_tensor.cpu().numpy().tolist()
df.to_pickle("questions_with_emb.pkl")
'''

df = pd.read_pickle("questions_with_emb.pkl")
df = df.dropna(subset=["YEAR", "QUESTION", "LABEL"]).reset_index(drop=True)

print(df.shape)
emb_matrix = np.vstack(df["embedding"].values).astype("float32")  # N×384
# 先做 L2 normalize => cosine 距離就 = 1 - dot
emb_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

# ──────────────────────────────────────────────
# 3. 建立 NearestNeighbors 模型 (cosine)
# ──────────────────────────────────────────────
nn = NearestNeighbors(
        n_neighbors=10,             # 1 (自己) + 前 5 個最像
        metric="cosine",
        algorithm="auto"
     ).fit(emb_norm)

# ──────────────────────────────────────────────
# 4a. 依「索引」查相似題目
# ──────────────────────────────────────────────
def similar_by_index(idx: int, topk: int = 15):
    """列出與 df.loc[idx] 最相似的 top-k 題目（不含自身）"""
    dist, ind = nn.kneighbors(emb_norm[idx:idx+1], n_neighbors=topk+1)
    ind = ind[0][1:]      # 去掉自己
    dist = dist[0][1:]
    res = df.iloc[ind].assign(similarity=1 - dist)
    res = (
        res.loc[res.similarity > 0.95]
           .dropna(subset=["YEAR", "QUESTION", "LABEL"])
    )
    res = res.sort_values("similarity", ascending=False)
    # ── 處理同一年份重複 ──
    rows_to_drop = []
    cur_label = df.at[idx, "LABEL"]

    for yr, grp in res.groupby("YEAR"):
        if len(grp) == 1:
            continue    # 沒重複，跳過

        # 若同一年份有多筆
        if grp["LABEL"].nunique() > 1:
            # 標籤不一；捨去與當前題目不同 LABEL 的列
            drop_idx = grp.index[grp["LABEL"] != cur_label].tolist()
            rows_to_drop.extend(drop_idx)

            # 如果全部都被丟掉，就保留相似度最高的那一筆
            if len(drop_idx) == len(grp):
                keep_idx = grp.index[0]   # 已排序，相似度最高
                rows_to_drop.extend(grp.index[1:])  # 丟掉剩下
        else:
            # LABEL 都相同 → 詢問是否全部保留
            print(res)
            check = input(
                f"Year {yr} has {len(grp)} duplicates with LABEL '{cur_label}'. "
                "Mark all? (y/n): "
            )
            if check.lower() != "y":
                # 只留相似度最高的那一筆
                rows_to_drop.extend(grp.index[1:])

    if rows_to_drop:
        res = res.drop(index=rows_to_drop)


    self_row = df.loc[[idx], ["YEAR", "QUESTION", "LABEL"]].assign(similarity=1.00)
    group = pd.concat([self_row, res])

    if group["LABEL"].nunique() > 1 and group["YEAR"].nunique() == len(group):
        print("===There are multiple labels in this group===")
        print(group)
        majority = group["LABEL"].value_counts().idxmax()
        # 少數派 index
        minority_idx = group.index[group["LABEL"] != majority]
        check = input(f"Auto-merge {len(minority_idx)} rows to LABEL '{majority}'? (y/n): ")
        if check.lower() != "y":
            print("Cancelled.")

            return res.loc[res.similarity > 0.95, ["YEAR", "QUESTION", "LABEL", "similarity"]]
        # 回寫到原 dataframe
        df.loc[minority_idx, "LABEL"] = majority
        # 同步更新這次傳回的 res
        res.loc[minority_idx.intersection(res.index), "LABEL"] = majority
        # （可加 print 觀察）print(f"Auto-merged {len(minority_idx)} rows to LABEL '{majority}'")
    return res.loc[res.similarity > 0.95, ["YEAR", "QUESTION", "LABEL", "similarity"]]
# 範例：找第 10 題最像的 5 題

# ──────────────────────────────────────────────
# 4b. 依「新文字」查相似題目
# ──────────────────────────────────────────────
model_path = "./models/paraphrase-multilingual-MiniLM-L12-v2"
if not os.path.exists(model_path):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder="./models"
    )
else:
    model = SentenceTransformer(model_path)
    
def similar_by_text(text: str, topk: int = 10):
    """輸入一段題目文字，回傳資料庫中最相近的 top-k 題目"""
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    dist, ind = nn.kneighbors(vec, n_neighbors=topk)
    res = df.iloc[ind[0]].assign(similarity=1 - dist[0])
    return res.loc[res.similarity > 0.95, ["YEAR", "QUESTION", "LABEL", "similarity", "checked"]]

'''
rr2008_idx = df[df["YEAR"] == "RR2008"].index.tolist()
output_path = "result.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for idx in rr2008_idx:
        header = f"=== RR2008 第 {idx}: {df.at[idx, 'QUESTION']} ===\n"
        f.write(header)
        sims = similar_by_index(idx, topk=10)
        if sims.empty:
            f.write("  (無相似度 > 0.95 的題目)\n\n")
        else:
            # to_string 會自動加換行
            f.write(sims.to_string(index=False) + "\n\n")

print(f"所有 RR2008 題目的相似題目已輸出到 {output_path}")
'''


df["checked"] = ""
rows = []

def idx_to_prefix(idx: int) -> str:
    result = ""
    n = idx + 1
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result

label_order = df["LABEL"].drop_duplicates().tolist()
label_prefix = {lab: idx_to_prefix(i) for i, lab in enumerate(label_order)}
label_counter = {lab: 0 for lab in label_order}

w = open("record.txt", "w", encoding="utf-8")
for idx in range(len(df)):
    if df.at[idx, "YEAR"] == "nan":
        continue
    if df.at[idx, "checked"] == "":
        print(f"Processing index {idx}, year = {df.at[idx, 'YEAR']}, question = {df.at[idx, 'QUESTION']}")
        w.write(f"Processing index {idx}, year = {df.at[idx, 'YEAR']}, question = {df.at[idx, 'QUESTION']}\n")
        sims = similar_by_index(idx, topk=10)
        print(sims)
        w.write(sims.to_string(index=False) + "\n")
        if sims.empty:
            print("  (無相似度 > 0.95 的題目)")
            w.write("  (無相似度 > 0.95 的題目)\n")
            df.at[idx, "checked"] = "N"
        elif sims["LABEL"].unique().size == 1 and sims["YEAR"].unique().size == sims["YEAR"].size:
            lab = df.at[idx, "LABEL"]
            label_counter[lab] += 1
            code_num = f"{label_prefix[lab]}{label_counter[lab]:02d}"
            idx_group = set([idx] + sims.index.tolist())
            d = {"id": code_num, "question": df.at[idx, "QUESTION"]}
            for j in idx_group:
                if df.at[j, "checked"] != "":
                    continue
                df.at[j, "checked"] = code_num
                y = df.at[j, "YEAR"]
                q = df.at[j, "QUESTION"]
                d[y] = q
            rows.append(d)
            print(f"{code_num} created.")
            w.write(f"{code_num} created.\n")
        else:
            idx_group = [idx] + sims.index.tolist()
            for i in idx_group:
                df.at[i, "checked"] = "X"
comp_df = pd.DataFrame(rows).set_index("question")
print(comp_df)
comp_df.to_csv("Comparison_table.csv", encoding="utf-8-sig")
df.drop(columns = ["embedding"]).to_csv("questions_with_emb_checked.csv", encoding="utf-8-sig", index=False)