# ChurnGuard — Customer Churn Prediction

A complete ML pipeline to predict telecom customer churn.

## Project Structure
- `DataSet/` — Raw and cleaned CSV data
- `Project_2/` — Python scripts (Tasks 1–5)
- `Dashboard/` — Visualization outputs

## Pipeline
| Task | File | Description |
|------|------|-------------|
| 1 | Task1_load_explore.py | Load & explore raw data |
| 2 | task2_clean_data.py | Clean & save cleaned_data.csv |
| 3 | task3_train_model.py | Train LR vs Random Forest |
| 4 | task4_predict.py | Predict churn for new customer |
| 5 | task5_visualize.py | Generate dashboard figures |

## Results
- Best Model: Logistic Regression
- ROC-AUC: 0.814
- Churn Recall: 90%

## How to Run
```bash
cd Project_2
python task2_clean_data.py   # generates cleaned_data.csv
python task3_train_model.py
python task4_predict.py
python task5_visualize.py
```