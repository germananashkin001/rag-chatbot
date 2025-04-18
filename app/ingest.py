import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import pymupdf
import json
from tqdm import tqdm

# Define paths
DATA_DIR = "data"
INDEX_PATH = "faiss_index.index"
EMBEDDINGS_PATH = "doc_embeddings.npy"
CHUNKS_PATH = "doc_chunks.txt"
CHUNKS_JSON_PATH = "doc_chunks.json"  # New JSON file path
CHUNK_SIZE = 300  # Adjust chunk size as needed

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to chunk text into smaller pieces
def chunk_text(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to load PDFs and extract text
def load_pdfs(folder_path):
    chunks = []  # To store all text chunks
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                print(f"Attempting to open {filename}...")
                doc = fitz.open(os.path.join(folder_path, filename))
                for page in doc:
                    page_text = page.get_text()  # Extract text from each page
                    # Add the chunked text from the page
                    chunks += chunk_text(page_text, CHUNK_SIZE)
            except pymupdf.FileDataError as e:
                print(f"Failed to open PDF {filename}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error with {filename}: {str(e)}")
    return chunks

# Function to generate embeddings for the chunks
def embed_chunks(chunks):
    print("Generating embeddings for the chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Function to save chunks and embeddings to JSON
def save_chunks(chunks, embeddings):
    chunk_data = [{'text': chunk, 'embedding': embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]  # Convert numpy array to list

    # Debug: Print first few chunks and their embeddings to inspect
    print(f"First chunk: {chunk_data[0]['text']}")
    print(f"First embedding: {chunk_data[0]['embedding']}")

    # Save to JSON file
    with open(CHUNKS_JSON_PATH, 'w') as f:
        json.dump(chunk_data, f, indent=4)  # Save in a readable format

    print(f"Saved {len(chunk_data)} chunks and embeddings to {CHUNKS_JSON_PATH}")

# Store the embeddings and chunks in a persistent format
def store_index(embeddings, chunks, index_path='index.pkl', chunks_path='chunks.pkl'):
    # Save the index (embeddings) and chunks
    with open(index_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Index and chunks saved to {index_path} and {chunks_path}")

# Main execution flow
if __name__ == "__main__":
    print("Loading PDFs...")
    chunks = load_pdfs(DATA_DIR)
    print(f"Loaded {len(chunks)} chunks from PDF files.")

    if len(chunks) == 0:
        print("No chunks were extracted. Please check your PDF files.")
    else:
        embeddings = embed_chunks(chunks)  # Generate embeddings for the chunks
        print(f"Generated {len(embeddings)} embeddings, each of shape {embeddings[0].shape if len(embeddings) > 0 else 'N/A'}")

        # Save chunks and their embeddings
        save_chunks(chunks, embeddings)

        # Store the embeddings and chunks
        store_index(np.array(embeddings), chunks)

    print("Ingestion process completed.")
