
# AIML Assignment: Hugging Face Sentiment Monitor
# Name: ENTER_YOUR_NAME
# PRN: ENTER_YOUR_PRN

from transformers import pipeline

print("===================================")
print("     HUGGING FACE SENTIMENT MONITOR")
print("===================================")

# Load the sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Sample sentences
sentences = [
    "I love learning artificial intelligence!",
    "This assignment is very difficult and frustrating.",
    "The program is working perfectly."
]

print("\nSentiment Analysis Results:\n")

for sentence in sentences:
    result = sentiment_model(sentence)[0]

    print("Sentence:", sentence)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"] * 100, 2), "%")
    print("-----------------------------------")

print("Program executed successfully!")
