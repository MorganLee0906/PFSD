import pandas as pd

def merge_results(df: pd.DataFrame) -> pd.DataFrame:
    q_counts = 0
    df["conflict"] = ""
    for col in df.columns:
        if col == "question" or col == "conflict":
            continue
        else:
            print(f"Processing column: {col}")
            df = df.sort_values(by=col, ascending=False)
            visited = set()
            to_drop = []
            new_rows = []
            for idx, row in df.iterrows():
                if row[col] in visited:
                    continue
                visited.add(row[col])
                sub_df = df[df[col] == row[col]]
                if len(sub_df) > 1:
                    try:
                        new_row = {"question": row["question"]}
                    except:
                        new_row = {"question": idx}
                    conflict = False
                    for c in sub_df.columns:
                        if c == "question" or c == "conflict":
                            continue
                        if sub_df[c].nunique() > 1:
                            print(f"Column '{c}' has multiple unique values for the same question: {row[col]}")
                            # wait = input(sub_df[c].unique())
                            conflict = True
                            df.loc[sub_df.index, "conflict"] = "V"
                        else:
                            new_row[c] = sub_df[c].iloc[0]
                    if conflict:
                        print("Conflict detected, skipping this question.")
                        continue

                    print(f"Creating new row for question: {new_row}")
                    new_rows.append(new_row)
                    print(f"Dropping {len(sub_df)} rows with question: {row[col]}")
                    to_drop.extend(sub_df.index.tolist())
            if to_drop:
                df = df.drop(index=to_drop)
                print(f"Dropped {len(to_drop)} rows.")
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_df], ignore_index=True)
                print(f"Added {len(new_rows)} new rows.")
        
    cols = df.columns.tolist()
    
    df = df[[cols[0]] + sorted(cols[1:])]
    sort_keys = df.columns[1:].tolist()
    df = df.sort_values( by=sort_keys,
                        kind="mergesort" ).reset_index(drop=True)
    return df

if __name__ == "__main__":
    fname = "compare_table_with_GPT.csv"
    df = pd.read_csv("compare_table_with_GPT.csv")
    df = merge_results(df)
    df.to_csv(f"{fname.split('.')[0]}_merged.csv", index=False)
