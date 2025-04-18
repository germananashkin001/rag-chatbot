import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json  # To load the saved JSON file

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load the chunks and embeddings from the JSON file
def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Function to retrieve relevant chunks based on a query
def retrieve(query, chunks, top_k=5):
    query_embedding = model.encode([query])
    chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])  # Access 'embedding' key

    # Calculate cosine similarities between the query and the chunks
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]

    # Retrieve top-k chunks based on similarity
    top_chunks = [chunks[i]['text'] for i in top_k_indices]  # Access 'text' key for the chunk text
    return top_chunks

# Example usage
if __name__ == "__main__":
    # Path to the saved JSON file containing chunks and embeddings
    json_path = 'doc_chunks.json'  # Change this path as per your actual JSON file

    # Load the chunks from the JSON file
    chunks = load_json(json_path)

    # Query from the user
    query = input("Enter your query: ")

    # Retrieve the top-k relevant chunks
    relevant_chunks = retrieve(query, chunks)

    # Print the relevant chunks
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"Chunk {i}: {chunk}")

