import pandas as pd

# fetching pkl
def drop_paired(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Initial shape: {df.shape}")
    res = pd.read_csv('compare_table_with_GPT.csv')
    print(res)
    drop = []
    for col in res.columns:
        if col == "question" or col == "conflict":
            continue
        print(f"Processing column: {col}")
        for idx, row in res.iterrows():
            if row[col] == "" or pd.isna(row[col]):
                continue
            try:
                num, question = row[col].split("/", 1)
            except:
                wait = input(row[col])
            if len(df[(df["YEAR"] == col) & (df["NUMBER"] == num)].index.tolist()):
                drop.append(df[(df["YEAR"] == col) & (df["NUMBER"] == num)].index.tolist()[0])
    print(f"Dropping {len(drop)} rows.")
    df.drop(index=drop, inplace=True)
    print(f"Shape after dropping: {df.shape}")
    return df
if __name__ == "__main__":
    df = pd.read_pickle('second_stage.pkl')
    wait = input(df)
    df = drop_paired(df)
    df.to_pickle("second_stage.pkl")
    df.drop(columns = "embedding")
    df.to_csv("questions_with_emb_checked.csv", encoding="utf-8-sig", index=False)
