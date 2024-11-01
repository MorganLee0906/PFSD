# PFSD Data Integration Project

## Project Overview
This project utilizes Jupyter Notebook to integrate data for PFSD (please describe the full name and background of PFSD here). The primary goal is to use the GPT-4o-mini model to cluster questionnaire items and compile the questions, answers, and individual responses into a comprehensive comparison table.

## Technologies Used
- **Python**: The main programming language.
- **Pandas**: For data processing and analysis.
- **GPT-4o-mini**: Utilized for its natural language processing capabilities to cluster questionnaire items.
  - Note: This project requires an `openai.api_key`, which should be purchased by the user.

## Files Used
- `sav_to_csv_survey.ipynb`
- `sas_to_csv_label.ipynb`
- `sas_to_csv_answer.ipynb`
  - The above three files are used for transforming the SAS files into CSV files.
- `(MAIN)summary_FiveYearsData_1101.ipynb`
  - This serves as the main file.
