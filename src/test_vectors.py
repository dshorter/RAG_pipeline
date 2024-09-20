import faiss
import numpy as np
import os

# Define a function to create and populate a FAISS index, then save it
def create_and_save_faiss_index():
    # Create a FAISS index with 128-dimensional vectors
    d = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

    # Add 10 random vectors to the index with custom IDs
    vectors = np.random.rand(10, d).astype('float32')
    ids = np.arange(1001, 1011)  # Example IDs
    
    # Log vectors and IDs before adding them
    print(f"Adding vectors: {vectors}")
    print(f"With IDs: {ids}")
    
    # Add vectors to the FAISS index
    index.add_with_ids(vectors, ids)
    
    # Check how many vectors were added
    print(f"Total vectors in index after adding: {index.ntotal}")  # Should be 10
    
    # Save the index to disk
    index_path = os.path.join(os.path.dirname(__file__), 'test_faiss_index.bin')
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path} (size should be larger than 6KB)")
    
    return index_path, vectors

# Define a function to load and inspect the FAISS index
def load_and_inspect_faiss_index(index_path, original_vectors):
    # Load the FAISS index from disk
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")

    # Check number of vectors
    print(f"Number of vectors in index: {index.ntotal}")

    # Log FAISS index properties
    print(f"FAISS index dimensionality: {index.d}")
    print(f"Is FAISS index trained? {index.is_trained}")

    # Perform a search to retrieve a vector
    query_id = 1001  # Use the ID of the first vector
    query_vector = original_vectors[0].reshape(1, -1)  # Use the original vector as the query
    distances, indices = index.search(query_vector, 1)  # Search for the closest vector

    print(f"Search result for ID {query_id}:")
    print(f"Distances: {distances}")
    print(f"Indices (corresponding IDs): {indices}")

    # Retrieve the associated IDs
    try:
        ids = np.array([index.id_map.at(i) for i in range(index.ntotal)])
        print(f"Associated IDs: {ids}")  # Force print the IDs
    except AttributeError:
        print("Error: Index does not have an ID map.")

# Run the test
index_path, original_vectors = create_and_save_faiss_index()  # Step 1: Create and save the index
load_and_inspect_faiss_index(index_path, original_vectors)    # Step 2: Load and inspect the index



