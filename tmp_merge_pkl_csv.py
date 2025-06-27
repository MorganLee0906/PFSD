import pandas as pd

csv_df = pd.read_csv("questions_with_emb_checked.csv")
pkl_df = pd.read_pickle("questions_with_emb.pkl")

# 以 YEAR + NUMBER 做 key，把 pkl_df 的 embedding 拿回來
merged_df = csv_df.merge(
    pkl_df[['YEAR','NUMBER','embedding']],
    on=['YEAR','NUMBER'],
    how='left'
)

# 輸出成新的 pickle
merged_df.to_pickle("second_stage.pkl")
