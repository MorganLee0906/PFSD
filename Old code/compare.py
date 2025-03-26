import pandas as pd


def find_common_categories(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 取得共同的 Category
    common_categories = set(df1['Category']) & set(df2['Category'])
    print("Common Categories:")
    for cat in common_categories:
        print(cat)
    print("="*40)
    diff_categories = set(df1['Category']).symmetric_difference(
        set(df2['Category']))
    print("Categories in File1 but not in File2:")
    for cat in set(df1['Category']):
        if cat in diff_categories:
            print(cat)
    print("Categories in File2 but not in File1:")
    for cat in set(df2['Category']):
        if cat in diff_categories:
            print(cat)

    # 列出共同 Category 的題組
    for cat in common_categories:
        subset1 = df1[df1['Category'] == cat]
        subset2 = df2[df2['Category'] == cat]

        print(f"Category: {cat}")
        print("File1:")
        print(subset1[['NUMBER', 'QUESTION']])
        print("File2:")
        print(subset2[['NUMBER', 'QUESTION']])
        print("="*40)
        wait = input("Press Enter to continue...")


# 使用範例
if __name__ == "__main__":
    find_common_categories(
        "/Users/lcy96/Documents/PFSD/data/five_years/CX2018_label_clustered.csv",
        "/Users/lcy96/Documents/PFSD/data/five_years/CV2008_label_clustered.csv"
    )
