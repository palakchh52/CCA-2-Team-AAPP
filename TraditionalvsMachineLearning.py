# AIML Assignment: Traditional Programming vs Machine Learning
# Name: Pronita Banerjee
# PRN: 1302250278

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print("==========================================")
print(" TRADITIONAL PROGRAMMING VS MACHINE LEARNING")
print("==========================================")

# Sample student data
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("\nStudent Data:")
print(df)

# Traditional programming: fixed rule
hours = 5
if hours >= 4:
    traditional_result = "Pass"
else:
    traditional_result = "Fail"

print("\nTraditional Programming")
print("Study Hours:", hours)
print("Predicted Result:", traditional_result)

# Machine learning: train a decision tree
X = df[["Study_Hours"]]
y = df["Result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

new_student = pd.DataFrame({"Study_Hours": [5]})
prediction = model.predict(new_student)[0]

ml_result = "Pass" if prediction == 1 else "Fail"

print("\nMachine Learning")
print("Study Hours:", 5)
print("Predicted Result:", ml_result)

print("\nProgram executed successfully!")
