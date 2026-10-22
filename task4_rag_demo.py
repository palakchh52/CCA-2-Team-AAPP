"""
Task 4: Basic Retrieval-Augmented Generation (RAG) demonstration.
This beginner-friendly example retrieves relevant notes and forms an answer
from the retrieved context. It does not call an external generative AI API.
Install once: pip install scikit-learn
Run: python task4_rag_demo.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning allows a computer to learn patterns from example data.",
    "Traditional programming uses human-written rules to produce an output.",
    "Hugging Face provides libraries and pretrained models for natural language processing.",
    "Sentiment analysis classifies text as positive, negative, or neutral depending on the model.",
    "Retrieval-Augmented Generation retrieves relevant information and uses it as context for an answer.",
    "Excel files can store tabular data that Python can load using pandas.",
    "A GitHub repository stores project files and tracks changes through commits.",
]

# Additional rules for the RAG workflow
rules = [
    "Retrieve the two most relevant notes for each question.",
    "Answer only from the retrieved notes.",
    "If the notes do not contain the answer, say the information is not available.",
    "Do not invent facts or claim that the demo uses a live external AI service.",
    "Show the retrieved context so the answer can be checked."
]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

def answer_question(question):
    q_vector = vectorizer.transform([question])
    similarities = cosine_similarity(q_vector, document_vectors)[0]
    best_indices = similarities.argsort()[::-1][:2]
    selected = [i for i in best_indices if similarities[i] > 0]
    if not selected:
        return "The information is not available in the current notes.", []
    context = [documents[i] for i in selected]
    # Simple context-grounded response: an educational RAG prototype, not an LLM.
    answer = "Based on the retrieved notes: " + " ".join(context)
    return answer, context

print("Basic RAG Demo — ask a question (type 'exit' to stop).")
print("\nRules:")
for rule in rules:
    print("-", rule)

while True:
    question = input("\nQuestion: ").strip()
    if question.lower() == "exit":
        break
    if not question:
        print("Please enter a question.")
        continue
    answer, context = answer_question(question)
    print("\nRetrieved context:")
    for item in context:
        print("-", item)
    print("\nAnswer:", answer)
    #Individual contribution
# Student Name: [Palak Chhablani]
# PRN: [1302250701]
# Task: Compare traditional programming and machine learning
# Input source: Excel file
