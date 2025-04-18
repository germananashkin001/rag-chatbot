import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np  # Import numpy for array manipulation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json  # To load the saved JSON file
import re

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load the chunks and embeddings from the JSON file
def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Function to clean and improve the retrieved text chunks
def clean_chunk_text(chunk_text):
    # Remove extra spaces and line breaks
    chunk_text = ' '.join(chunk_text.splitlines())
    # Optionally, remove non-alphanumeric characters that might have slipped in
    chunk_text = re.sub(r'[^\w\s.,!?-]', '', chunk_text)
    return chunk_text.strip()

# Function to retrieve relevant chunks based on a query
def retrieve(query, chunks, top_k=5):
    query_embedding = model.encode([query])
    chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])  # Access 'embedding' key

    # Calculate cosine similarities between the query and the chunks
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]

    # Retrieve top-k chunks based on similarity
    top_chunks = [clean_chunk_text(chunks[i]['text']) for i in top_k_indices]  # Clean the chunk text
    return top_chunks

# Function to query OpenAI with the relevant chunks and the user query
def query_openai_with_chunks(query, chunks, openai_api_key, model="gpt-4", top_k=5):
    # Retrieve top-k chunks relevant to the query
    relevant_chunks = retrieve(query, chunks, top_k)

    # Format the prompt with the relevant chunks and the query
    prompt = "\n".join(relevant_chunks) + "\n\n" + f"Question: {query}\nAnswer:"

    # Set the OpenAI API key
    openai.api_key = openai_api_key
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    # Initialize OpenAI client
    openai.api_key = api_key

    # Sending the request to OpenAI with the relevant chunks and query
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an intelligent and reliable assistant."},  # System prompt
            {"role": "user", "content": f"Here are some relevant details:\n{prompt}\n\nPlease answer the question: {query}"}  # User prompt
        ]
    )

    # Extract and return the model's response
    response_content = response['choices'][0]['message']['content'].strip()
    return response_content

# Example usage
if __name__ == "__main__":
    # Path to the saved JSON file containing chunks and embeddings
    json_path = 'doc_chunks.json'  # Change this path as per your actual JSON file

    # Load the chunks from the JSON file
    chunks = load_json(json_path)
    
    # Query from the user
    while True:
        query = input("Enter your query or type STOP to close session: ")
        
        if query == 'STOP':
            print('Session closed')
            break

        relevant_chunks = retrieve(query, chunks, 5)

        # Format the prompt with the relevant chunks and the query
        prompt = "Based on relevant context from documents: " + "\n".join(relevant_chunks) + "\n\n" + f"Answer this question: {query}\nAnswer:"
        load_dotenv()

        # Get the key from environment variables
        api_key = os.getenv("API_KEY")

        # Use it, for example:
        os.environ["OPENAI_API_KEY"] = api_key
        # Initialize the OpenAI client
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,  # Optional: decide if you need to store the interaction
            messages=[
                {"role": "system", "content": "You are an intelligent and reliable assistant."},  # System prompt
                {"role": "user", "content": prompt}  # User prompt
            ])

        response = completion.choices[0].message.content.strip()
        print(response)
