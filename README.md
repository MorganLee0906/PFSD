# PFSD Data Integration

## Project Overview
This project utilizes Jupyter Notebook to integrate PFSD data. The primary goal is to use the GPT-4o-mini model to cluster questionnaire items and compile the questions, answers, and individual responses into a comprehensive comparison table.
You can also check `PFSD_presentation_1030.pdf` to see the project detail. 

## Technologies Used
- **Python**: The main programming language.
- **Pandas**: For data processing and analysis.
- **GPT-4o-mini**: Utilized for its natural language processing capabilities to cluster questionnaire items.
  - Note: This project requires an `openai.api_key`, which must be purchased by the user.

## Files Used
- `sav_to_csv_survey.ipynb`
- `sas_to_csv_label.ipynb`
- `sas_to_csv_answer.ipynb`
  - These files are used for transforming SAS files into CSV files.
- `(MAIN)summary_FiveYearsData_1101.ipynb`
  - This serves as the main file.

## Execution Process
- **sav_to_csv_survey.ipynb**, **sas_to_csv_label.ipynb**, **sas_to_csv_answer.ipynb**:
  - Transform SAS files downloaded from SRDA into CSV files.
- **Using the main file for the following processes**, excluding the steps marked as "Not Required":
  - **Load Package**
  - **CKIP Word Segmentation** *(Not Required)*: Only for statistical clustering and analyzing the survey questions of each year.
  - **Call API to do Embedding** *(Not Required)*: Only for statistical clustering.
  - **Data Preprocessing**: Manually classify related topics across years.
  - **Categorize with GPT-3.5 Turbo** *(Not Required)*: A trial step using OpenAI to do clustering.
  - **Survey Questions Clustering**: Use GPT-4o-mini for classification. Output file: `type_x.csv`
  - **Using Statistical Learning to do Clustering** *(Not Required)*
  - **Answer Organizing**: Organize answers using union. Output file: `variable_map_type_x.csv`
  - **Output Survey Outcome**: Output each person's answer in each cluster.
  

## Vision
<<<<<<< HEAD
- **Python 3.12**
=======

---

## 檔案說明
---
### main.py
將原先的ipynb code 整併到同一檔案
### add_to_csv.py
將文字輸出檔更新至csv（開發時的測試code，可忽略）
### check.py
檢查cluster是否正確用，未來可開發成網頁版方便操作
### merge.py
將分類好的題目與問卷整併
（原先於notebook的整併功能僅顯示問卷選項變數，此版本自動轉換為選項文字方便檢查）

---
## 使用說明
(待補)
>>>>>>> 7f7deb8 (Modify clustering procedure and add some tool (merge.py))
