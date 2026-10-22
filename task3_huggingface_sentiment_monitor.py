"""
Task 3: Hugging Face Sentiment Monitor
Install once: pip install transformers torch
First run may download a model and needs internet.
Run: python task3_huggingface_sentiment_monitor.py
"""
from transformers import pipeline

print("Loading Hugging Face sentiment model. First run can take a few minutes...")
sentiment_analyzer = pipeline("sentiment-analysis")

problem_statements = [
    "Monitor customer feedback to identify positive and negative experiences.",
    "Analyse student feedback about college facilities.",
    "Review product feedback to identify areas for improvement.",
    "Track service complaints and flag negative comments."
]

print("\nAdditional problem statements:")
for i, statement in enumerate(problem_statements, 1):
    print(f"{i}. {statement}")

print("\nEnter a sentence to analyse (type 'exit' to stop).")
while True:
    text = input("Feedback: ").strip()
    if text.lower() == "exit":
        break
    if not text:
        print("Please enter some text.")
        continue
    result = sentiment_analyzer(text)[0]
    print(f"Sentiment: {result['label']} | Confidence: {result['score']:.2%}")
#Individual contribution
# Student Name: [Palak Chhablani]
# PRN: [1302250701]
# Task: Compare traditional programming and machine learning
# Input source: Excel file
