
# AIML Assignment
# Traditional Programming vs Machine Learning
# Name: Your Name
# PRN: Your PRN

from sklearn.tree import DecisionTreeClassifier

print("========================================")
print("TRADITIONAL PROGRAMMING VS MACHINE LEARNING")
print("========================================")

# Sample data created directly in Python
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
results = [0, 0, 0, 1, 1, 1, 1, 1]

print("\nSample Student Data:")
for i in range(len(study_hours)):
    result = "Pass" if results[i] == 1 else "Fail"
    print("Study Hours:", study_hours[i][0], "| Result:", result)

# Traditional Programming: fixed rule
hours = 5

if hours >= 4:
    traditional_result = "Pass"
else:
    traditional_result = "Fail"

print("\n--- Traditional Programming ---")
print("Study Hours:", hours)
print("Result:", traditional_result)

# Machine Learning: Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(study_hours, results)

prediction = model.predict([[hours]])[0]
ml_result = "Pass" if prediction == 1 else "Fail"

print("\n--- Machine Learning ---")
print("Study Hours:", hours)
print("Predicted Result:", ml_result)

print("\nProgram executed successfully!")
