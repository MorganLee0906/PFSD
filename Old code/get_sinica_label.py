import pandas as pd
import os
import json
import difflib

path = "../../Downloads/From_中研院_PSFD歷年題項整理/PSFD歷年題項整理"
js = open('./fine_tune_all.jsonl', "w+", encoding="utf-8")
for f in os.listdir(path):
    if f.endswith(".xlsx") and not f.startswith("~"):
        print(f)
        df = pd.read_excel(os.path.join(path, f))
        print(df.columns)
        wait = input("Press Enter to continue...")
        selected_cols = [col for col in df.columns if "標籤" in col or "題目" in col]
        label_cols = [col for col in selected_cols if "標籤" in col]
        question_col = next((col for col in df.columns if "題目" in col), None)
        print(selected_cols)
        print(question_col)
        if label_cols == []:
            print("No label columns found, skipping...")
            continue
        
        category = {}
        for row in df[selected_cols].index:
            if f.startswith("B_RI") or f.startswith("K_"):
                labels = [str(df.at[row, col]).split('：')[0] for col in label_cols if pd.notna(df.at[row, col])]
            else:
                labels = [str(df.at[row, col]) for col in label_cols if pd.notna(df.at[row, col])]
            if len(labels) > 1 or (len(labels) and (f.startswith("B_RI") or f.startswith("K_"))):
                merged_label = f.split("_")[1].split(".")[0] + ":" + labels[0]
            else:
                merged_label = f.split("_")[1].split(".")[0]
            
            question = df.at[row, question_col]
            if pd.isna(question):
                continue
            if merged_label not in category:
                category[merged_label] = [question]
            else:
                flag = False
                for q in category[merged_label]:
                    if difflib.SequenceMatcher(None, question, q).ratio() > 0.7:
                        #wait = input(f"Duplicate question found: {question} and {q}")
                        flag = True
                        break
                if not flag:
                    category[merged_label].append(question)
        for key, value in category.items():
            try:
                for i in range(0, len(value), 5):
                    batch = value[i:i+5]
                    msg = {}
                    msg["messages"] = []
                    msg["messages"].append(
                        {"role": "system", "content": ""})
                    msg["messages"].append(
                        {"role": "user", "content": f"{', '.join(batch)}"})
                    msg["messages"].append(
                        {"role": "assistant", "content": f"{key}"})
                    msg_text = json.dumps(msg, ensure_ascii=False)
                    print(msg_text)
                    js.write(msg_text+'\n')
            except:
                print("Error in processing the message")
                print(key)
                print(value)
                wait = input("Press Enter to continue...")
        #wait = input(category.keys())
    else:
        print(f"skipping {f}")
js.close()