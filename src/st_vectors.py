import os    
import faiss
import numpy as np
import pandas as pd
import streamlit as st

# Function to load the actual FAISS index from a file
def load_faiss_index(index_path):
    try:
        index = faiss.read_index(index_path)
        st.success(f"FAISS index loaded from {index_path}")
        return index
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# Function to perform a search on the FAISS index
def search_faiss_index(index, query_vector):
    # Perform the search to retrieve the closest vector
    distances, indices = index.search(query_vector, 1)  # Return the single closest vector
    return distances, indices

# Function to retrieve vectors by performing searches on the FAISS index
def retrieve_vectors_from_index(index, original_vectors):
    results = []
    
    for i, query_vector in enumerate(original_vectors):
        distances, indices = search_faiss_index(index, query_vector.reshape(1, -1))  # Search with each vector
        result = {
            'ID': indices[0][0],
            'Distance': distances[0][0],
            'Query Vector': query_vector  # Store the query vector itself
        }
        results.append(result)
    
    return results

# Streamlit app to load and display the FAISS index
def main():
    st.title("FAISS Index Viewer")

    project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    index_path = os.path.join(project_folder, 'data', 'faiss_index.bin')    

    # Load the FAISS index
    index = load_faiss_index(index_path)

    # Generate or load some original vectors for querying (for demonstration purposes)
    d = 1536  
    original_vectors = np.random.rand(10, d).astype('float32')  # For demo, generate random vectors

    if index is not None:
        # Retrieve vectors and their IDs/distances by performing searches
        search_results = retrieve_vectors_from_index(index, original_vectors)

        # Create a DataFrame to display the results
        data = {
            'ID': [result['ID'] for result in search_results],
            'Distance': [result['Distance'] for result in search_results],
            # Convert query vectors to string format to fit into the DataFrame
            'Query Vector': [", ".join(f"{x:.4f}" for x in result['Query Vector']) for result in search_results]
        }
        
        # Display the DataFrame in Streamlit
        st.dataframe(pd.DataFrame(data))

if __name__ == '__main__':
    main()


