
# AIML Assignment: Retrieval-Augmented Generation
# Name: ENTER_YOUR_NAME
# PRN: ENTER_YOUR_PRN

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("===================================")
print("   RETRIEVAL AUGMENTED GENERATION")
print("===================================")

# Knowledge base containing sample information
documents = [
    'Artificial Intelligence enables machines to perform tasks that normally require human intelligence.',
    "Machine Learning allows computers to learn patterns from data and make predictions.",
    "Deep Learning uses neural networks with multiple layersto learn complex patterns.",
    "Natural Language Processing helps computers understandand process human language."
]

# Convert the documents into numerical vectors
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

# Ask a question
question = "How do computers learn from data?"

# Convert the question into a vector
question_vector = vectorizer.transform([question])

# Find the most relevant document
similarities = cosine_similarity(
    question_vector, document_vectors
)[0]

best_index = similarities.argmax()

print("\nQuestion:", question)
print("\nRetrieved Information:")
print(documents[best_index])

print("\nRelevance Score:",
      round(similarities[best_index], 3))

print("\nProgram executed successfully!")
