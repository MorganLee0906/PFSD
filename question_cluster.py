import os
import pandas as pd
import openai
f = open("api_key.txt", "r")
api_key = f.read()
openai.api_key = api_key
f.close()

question_path = 'data/five_years/'
question = {}
clustered = {}
for f in os.listdir(question_path):
    if f.endswith('2008_label.csv'):
        year = f.split('_')[0]
        question[year] = {}
        csv = pd.read_csv(question_path + f)
        csv = csv.dropna()
        csv['Category'] = ''
        print('Read:', f)
        for row in csv.index:
            num = csv.loc[row]['NUMBER'][:3]
            if num not in question[year]:
                question[year][num] = [csv.loc[row]['QUESTION']]
            else:
                question[year][num].append(csv.loc[row]['QUESTION'])
        print('Finish reading:', f)
        for i in question[year]:
            str = ', '.join(question[year][i])
            print(i, str)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                        "content": "你是一個問卷題目分類專家，給你一組以逗號分隔的題目，為該組題目下適當的標題。回覆需簡潔無贅字，避免重複字詞如：..調查等。必須按照以下格式：標題"
                     },
                    {"role": "user",
                        "content": f"{str}"
                     }
                ],
                max_tokens=20,
                temperature=0.1
            )
            print(response.choices[0].message.content.strip())
            for s in str.split(', '):
                csv.loc[csv['QUESTION'] ==
                        s, 'Category'] = response.choices[0].message.content.strip()
        csv.to_csv(question_path + f[:-4] + '_clustered.csv', index=False)
        wait = input('Press Enter to continue...')
