import re
import pandas as pd
id = input("Enter the type of the file: ")
t = open(f"record_type_{id}.txt", "r")
c = pd.read_csv(f'data/five_years/five_years_by_type/type_{id}.csv')
c["matched"] = ""
txt_content = t.read()
content = txt_content.split("Processing index: ")
for i in content:
    if i != "":
        idx, content = i.split(" Year: ")
        print(idx)
        cluster_match = re.search(r'Cluster:\s*(\d+)', content)
        similarity_match = re.search(
            r'Similarity:\s*([0-9]*\.?[0-9]+)', content)
        match = re.search(r'Similar with: \[(.*?)\]', content)
        predicted_matched = match.group(1).split(', ') if match else []
        predicted_cluster = cluster_match.group(1) if cluster_match else 0
        predicted_similarity = similarity_match.group(
            1) if similarity_match else 0
        print(predicted_cluster, predicted_similarity, predicted_matched)
        if c.at[int(idx), 'cluster'] == int(predicted_cluster) and c.at[int(idx), 'similarity'] == float(predicted_similarity):
            c.at[int(idx), 'matched'] = ','.join(predicted_matched)
        else:
            w = input("The data is not matched", idx, c.at[int(
                idx), 'cluster'], c.at[int(idx), 'similarity'],)
c.to_csv(f'data/five_years/five_years_by_type/type_{id}.csv', index=False)
t.close()
print("Done")
