"""
Task 1: Traditional Programming vs Machine Learning using Excel input.
Run: python task1_traditional_vs_ml.py
Install once: pip install pandas openpyxl scikit-learn
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

FILE = "ml_input.xlsx"

data = pd.read_excel(FILE)
print("Excel input data:")
print(data.head())

# Traditional programming: manually defined rule
def traditional_prediction(hours, attendance, previous_score):
    if hours >= 4 and attendance >= 70 and previous_score >= 50:
        return 1
    return 0

data["Traditional_Prediction"] = data.apply(
    lambda r: traditional_prediction(
        r["Hours_Studied"], r["Attendance_Percent"], r["Previous_Score"]
    ), axis=1
)

# Machine learning: model learns patterns from labelled rows in the spreadsheet
X = data[["Hours_Studied", "Attendance_Percent", "Previous_Score"]]
y = data["Passed"]
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)
data["ML_Prediction"] = model.predict(X)

print("\nComparison (1 = likely pass, 0 = likely not pass):")
print(data[["Hours_Studied", "Attendance_Percent", "Previous_Score",
            "Passed", "Traditional_Prediction", "ML_Prediction"]].to_string(index=False))
print("\nTraining-data accuracy:", round(accuracy_score(y, model.predict(X)) * 100, 2), "%")
print("\nNote: This is a small educational example, not a real student assessment tool.")
#Individual contribution
# Student Name: [Palak Chhablani
# PRN: [1302250701]
# Task: Compare traditional programming and machine learning
# Input source: Excel file
