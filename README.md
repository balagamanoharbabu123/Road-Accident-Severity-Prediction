# Road Accident Severity Prediction (India)

## Problem statement
Predict how severe a road accident will be (Slight Injury, Serious Injury or Fatal injury) from driver details, vehicle type, road conditions, weather and the cause of the accident, and identify which factors are most associated with severe accidents.

## Dataset
"Road Accident Severity in India" (Kaggle, s3programmer): 12,316 accident records, collected from Indian city police departments.

## Tools
Python, pandas, matplotlib, seaborn, scikit-learn, Streamlit, GitHub.

## Approach
1. Cleaned the data: removed outcome columns (data leakage), filled missing values with "Unknown", created an `Hour` feature.
2. EDA: severity distribution, top causes, causes with the most severe outcomes, accidents by hour.
3. One-hot encoded categorical features and split data 80/20 with stratification.
4. Trained Logistic Regression (baseline) and Random Forest (first try and tuned).
5. Handled class imbalance (85% Slight, 14% Serious, 1% Fatal) with class weights.
6. Evaluated with per-class precision, recall, F1 and confusion matrices, not accuracy alone.
7. Deployed a Streamlit app.

## Results (test set, 2,464 rows)
| Model | Accuracy | Recall Fatal | Recall Serious | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.46 | 0.39 | 0.39 | 0.30 |
| Random Forest (first try) | 0.84 | 0.00 | 0.10 | 0.36 |
| Random Forest (tuned) | 0.68 | 0.39 | 0.38 | 0.40 |

The tuned Random Forest is chosen because it catches severe accidents far better than the first try, even though its accuracy is lower.

## Limitations
- Small dataset; only 158 Fatal cases, so Fatal results are unstable.
- Speed, seat-belt use and vehicle damage are not recorded.
- Low precision for Fatal (many false alarms).
- Feature importance shows association, not causation.

## Run it
```
pip install -r requirements.txt
# run the notebook once to create model.pkl and options.json
streamlit run app.py
```

## Interview questions and simple answers

**What is the problem and why does it matter?**
Road accidents cause many deaths. If we understand which conditions lead to severe accidents, authorities and drivers can focus safety efforts there.

**Why is this classification and not regression?**
We predict a category (Slight, Serious, Fatal), not a number. With 3 categories it is multi-class classification.

**What is class imbalance and how did you handle it?**
85% of accidents are Slight, only 1% are Fatal. A model can score 85% accuracy by always saying "Slight". I used class weights so mistakes on rare classes count more, and I judged models by recall and F1 per class.

**Why not use accuracy?**
Because of imbalance, accuracy hides the failure on rare classes. My first Random Forest had 84% accuracy but caught 0% of Fatal cases.

**What is data leakage and did you avoid it?**
Leakage is using information that would not be available at prediction time, or that reveals the answer. I removed `Casualty_severity` and `Number_of_casualties` because they describe the outcome.

**What is one-hot encoding?**
Models need numbers. One-hot encoding turns each category into its own 0/1 column, for example Weather_Raining = 1 or 0.

**Why split data into train and test? What is stratify?**
Training data teaches the model; test data checks it on accidents it has never seen. Stratify keeps the class percentages equal in both parts, which matters for the rare Fatal class.

**How does Random Forest work?**
It builds many decision trees on random parts of the data and lets them vote. This is more stable than a single tree.

**What is overfitting and how did you see it?**
Overfitting is memorising the training data instead of learning general patterns. My first Random Forest used deep trees and ignored rare classes. Limiting `max_depth` and setting `min_samples_leaf` gave better recall on severe cases.

**Why Logistic Regression as well?**
It is a simple baseline. Comparing against it shows whether a more complex model is really needed.

**Explain precision and recall.**
Precision: of the accidents I called Fatal, how many were Fatal? Recall: of the real Fatal accidents, how many did I catch? For safety, recall matters more, but low precision means more false alarms. That is the trade-off.

**What are the limitations?**
Small data, very few Fatal cases, no speed or seat-belt data, and feature importance is not causation.

**How would you improve it?**
More data, SMOTE oversampling, hyperparameter tuning with cross-validation, XGBoost, and data with speed and vehicle damage.
