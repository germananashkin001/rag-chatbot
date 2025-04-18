import pickle

# Path to your chunks.pkl file
chunks_path = 'chunks.pkl'

# Function to load and check the structure of the chunks
def check_chunks(chunks_path):
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    # Check the first few items in the chunks list
    if len(chunks) > 0:
        print(f"First chunk data type: {type(chunks[0])}")
        print(f"First chunk content: {chunks[0]}")
        
        # Check if the first chunk is a dictionary and contains the expected keys
        if isinstance(chunks[0], dict):
            print("First chunk is a dictionary with the following keys:")
            print(f"Keys: {chunks[0].keys()}")
            if 'text' in chunks[0] and 'embedding' in chunks[0]:
                print("First chunk contains 'text' and 'embedding' keys.")
            else:
                print("First chunk does not contain 'text' and 'embedding' keys.")
        else:
            print("First chunk is not a dictionary. It might be a string or other type.")
    else:
        print("The chunks file is empty or invalid.")

# Run the check function
if __name__ == "__main__":
    check_chunks(chunks_path)
