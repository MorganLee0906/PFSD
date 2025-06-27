import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import NearestNeighbors
import numpy as np
import string
import openai

f = open("api_key.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()
model = "gpt-4o-mini-2024-07-18"


df = pd.read_pickle("questions_with_emb.pkl")
df = df.dropna(subset=["YEAR", "QUESTION", "LABEL"]).reset_index(drop=True)

df["dup_same_year"] = ""          # 新欄位，空字串表示尚未標記
df["paired"] = ""                # 標記跨年已連結的題目，避免重複


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

def cluster_by_embedding(df):
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

    # ────────────────────────────────────────────────────────────────
    # Step 3: merge rows that differ only by extra years
    # ────────────────────────────────────────────────────────────────
    if rows:
        merged = {}
        for r in rows:
            # canonical key: 去掉 leading NUMBER，只看題目文字
            if "/" in r["question"]:
                _, q_txt = r["question"].split("/", 1)
            else:
                q_txt = r["question"]

            if q_txt not in merged:
                merged[q_txt] = r
            else:
                # 把新的年份資料合併進去；已有的不覆寫
                for yr, val in r.items():
                    if yr == "question":
                        continue
                    if yr not in merged[q_txt]:
                        merged[q_txt][yr] = val

        rows = list(merged.values())
    # 4) 輸出 compare_table.csv
    if rows:
        compare_table = pd.DataFrame(rows).set_index("question")
        compare_table.to_csv("compare_table.csv", encoding="utf-8-sig")
        print(f"compare_table saved. Rows: {len(compare_table)}, Columns (years): {compare_table.shape[1]-1}")
    else:
        print("No cross‑year matches found with similarity > 0.95.")
    return df


def gpt_link_subgroup(sub_df: pd.DataFrame) -> dict | None:
    """
    Given a dataframe of the same pset & sub_id across years (still unpaired),
    ask GPT‑4o to link identical items.
    Returns a dict mapping year -> 'NUMBER/QUESTION', ready for compare_table,
    or None if GPT fails/parses poorly.
    """
    # Build prompt list: YearTag-NUMBER  QUESTION
    lines = []
    for _, r in sub_df.iterrows():
        year_tag = str(r["YEAR"])
        lines.append(f"{year_tag}-{r['NUMBER']}: {r['QUESTION']}")
    prompt = (
        "Below is a set of survey sub‑questions (same theme but different years).\n"
        "Group the semantically identical ones across years.  "
        "Respond with ONE line in the exact format:\n"
        "YYYY-NUMBER=YYYY-NUMBER=...   (use the Year tag and NUMBER shown)\n"
        "Include ALL input lines that are meaningfully the same; "
        "if none match, reply with the single YYYY-NUMBER only.\n\n"
        "Questions:\n" + "\n".join(lines) + "\n\nGrouping:"
    )
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        grp_line = resp.choices[0].message.content.strip()
        if "=" not in grp_line:
            return None
        mapping = {}
        for token in grp_line.split("="):
            token = token.strip()
            if "-" not in token:
                continue
            y, num = token.split("-", 1)
            row = sub_df[(sub_df["YEAR"].astype(str) == y) & (sub_df["NUMBER"] == num)]
            if not row.empty:
                q = row.iloc[0]
                mapping[y] = (q.name, f"{q['NUMBER']}/{q['QUESTION']}")
        if len(mapping) >= 2:
            first_key = next(iter(mapping.values()))[1]   # pick question text
            row_dict = {"question": first_key}
            use_indices = []
            for ytag, (idx, txt) in mapping.items():
                row_dict[ytag] = txt
                use_indices.append(idx)
            return row_dict, use_indices
    except Exception as e:
        print("GPT linking error:", e)
    return None

df = cluster_by_embedding(df)
df[df["paired"] != "Y"].drop(columns = ["embedding"]).to_csv("unpaired.csv", encoding="utf-8-sig", index=False)

# Step 2b: 找出「pset 在同一年相似度高」的題組，
#          再去其他年份尋找相似度 > 0.95 的 *pset 群*，僅列印結果
# ────────────────────────────────────────────────────────────────
print("\n=== Cross‑year similar pset groups ===")

 # -- 保險：若還沒生成 sub_id，立即補上 --
if "sub_id" not in df.columns:
    import re
    def _split_number(n):
        m = re.match(r"([A-Za-z]+\d+)([A-Za-z]*)$", str(n))
        return m.group(2).lower() if m else ""
    df["sub_id"] = df["NUMBER"].apply(_split_number)

 # ① 取出尚未 paired 的題目
unpaired_df = df[df["paired"] != "Y"].copy()

if unpaired_df.empty:
    print("\n=== All questions are already paired. ===")
else:
    print("\n=== Cross‑year similar groups (based on YEAR+pset) ===")

    # ② 先把同一年、同 pset 的題目聚成一個 seed‑group
    groups = []
    for (yr, ps), sub in unpaired_df.groupby(["YEAR", "pset"]):
        emb_mean = np.mean(np.stack(sub["embedding"].values), axis=0)
        groups.append(
            {
                "YEAR": yr,
                "pset": ps,
                "indices": sub.index.tolist(),
                "emb": emb_mean,
            }
        )
    grp_df = pd.DataFrame(groups)
    print(grp_df)
    if len(grp_df) <= 1:
        print("Not enough groups to compare.")
    else:
        # ③ 對群組平均向量做最近鄰，比對跨年、相似度 > 0.95
        emb_mat = np.stack(grp_df["emb"].values)
        nn_grp = NearestNeighbors(metric="cosine").fit(emb_mat)
        visited = set()

        for i, g in grp_df.iterrows():
            if i in visited:
                continue

            dist, ind = nn_grp.kneighbors(
                emb_mat[i : i + 1], n_neighbors=len(grp_df)
            )
            ind, dist = ind[0][1:], dist[0][1:]  # 去掉自己

            match_ids = [i]
            years_seen = {g["YEAR"]}

            for j, d in zip(ind, dist):
                if 1 - d < 0.95:
                    break
                if grp_df.at[j, "YEAR"] in years_seen:
                    continue
                match_ids.append(j)
                years_seen.add(grp_df.at[j, "YEAR"])

            if len(match_ids) == 1:
                continue  # 單獨群組，跳過

            # ④ 列印候選群組並由使用者確認是否標記 paired
            print("\n[GROUP CANDIDATE]")
            for gid in match_ids:
                yr = grp_df.at[gid, "YEAR"]
                ps = grp_df.at[gid, "pset"]
                print(f"YEAR {yr} | pset {ps}")
                # 列出該 YEAR+pset 的所有題目（不論 paired 狀態）
                full_sub = df[(df["YEAR"] == yr) & (df["pset"] == ps)]
                for _, q in full_sub.iterrows():
                    print(f"   - {q['NUMBER']}/{q['QUESTION']}/{q['paired']}")
            ans = input("Mark ALL questions in these groups as paired? (y/n) ")
            if ans.lower().startswith("y"):
                for gid in match_ids:
                    visited.add(gid)
                    for idx in grp_df.at[gid, "indices"]:
                        df.at[idx, "paired"] = "Y"
            else:
                print("  -> skipped.")
