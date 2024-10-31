 # Update to faiss_diagnostic.py
import faiss
import numpy as np
import os
import argparse
import logging
import json
from typing import Tuple, List, Dict, Optional

class FAISSIndexDiagnostic:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)

    def inspect_existing_index(self) -> dict:
        """Inspect an existing index for diagnostic information."""
        try:
            print(f"\nInspecting index at: {self.index_path}")
            
            if not os.path.exists(self.index_path):
                print("Index file does not exist!")
                return {}
                
            print(f"File size: {os.path.getsize(self.index_path)} bytes")
            
            index = faiss.read_index(self.index_path)
            
            # Check index type and capabilities
            is_idmap = isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2))
            has_reconstruct = hasattr(index, 'reconstruct') and callable(index.reconstruct)
            base_index = index.index if is_idmap else index
            
            # Initialize stats dictionary
            stats = {
                'file_size_bytes': os.path.getsize(self.index_path),
                'num_vectors': index.ntotal,
                'dimension': base_index.d,
                'is_trained': getattr(base_index, 'is_trained', True),
                'type': type(index).__name__,
                'id_map_type': 'IndexIDMap2' if isinstance(index, faiss.IndexIDMap2) else 
                              'IndexIDMap' if isinstance(index, faiss.IndexIDMap) else 'None',
                'has_reconstruct': has_reconstruct
            }
            
            # Try to extract vector statistics if possible
            vectors_available = False
            if index.ntotal > 0 and has_reconstruct:
                try:
                    # Try to reconstruct first vector as a test
                    test_id = index.id_map[0] if is_idmap else 0
                    test_vector = index.reconstruct(int(test_id))
                    vectors_available = True
                    
                    # Get sample statistics
                    sample_size = min(100, index.ntotal)
                    sample_ids = index.id_map[:sample_size] if is_idmap else range(sample_size)
                    
                    vectors = []
                    for idx in sample_ids:
                        try:
                            vector = index.reconstruct(int(idx))
                            vectors.append(vector)
                        except Exception as e:
                            self.logger.warning(f"Could not reconstruct vector {idx}: {str(e)}")
                    
                    if vectors:
                        vectors_array = np.array(vectors)
                        stats['vector_stats'] = {
                            'min': float(np.min(vectors_array)),
                            'max': float(np.max(vectors_array)),
                            'mean': float(np.mean(vectors_array)),
                            'std': float(np.std(vectors_array)),
                            'magnitude_mean': float(np.mean([np.linalg.norm(v) for v in vectors])),
                            'samples_analyzed': len(vectors)
                        }
                except Exception as e:
                    self.logger.warning(f"Could not analyze vectors: {str(e)}")
            
            stats['sample_vector_available'] = vectors_available
            
            # Print results in a readable format
            for key, value in stats.items():
                if key != 'vector_stats':
                    print(f"{key}: {value}")
                else:
                    if vectors_available:
                        print("\nVector Statistics:")
                        for stat_key, stat_value in value.items():
                            print(f"  {stat_key}: {stat_value}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error inspecting index: {e}")
            return {
                'error': str(e),
                'file_exists': os.path.exists(self.index_path),
                'file_size_bytes': os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0
            }

    def analyze_id_mapping(self) -> Dict:
        """Analyze the ID mapping in the index."""
        try:
            index = faiss.read_index(self.index_path)
            
            if not isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                return {'error': 'Index does not use ID mapping'}
                
            # Analyze ID map
            id_stats = {
                'total_ids': len(index.id_map),
                'unique_ids': len(set(index.id_map)),
                'min_id': int(min(index.id_map)),
                'max_id': int(max(index.id_map)),
                'has_gaps': len(set(range(min(index.id_map), max(index.id_map) + 1))) != len(index.id_map)
            }
            
            return id_stats
        except Exception as e:
            return {'error': f'Failed to analyze ID mapping: {str(e)}'}

    def verify_index(self, original_vectors: Optional[np.ndarray] = None, 
                    original_ids: Optional[np.ndarray] = None) -> bool:
        """
        Verify index content matches expected vectors or perform general verification.
        Can be used with or without original vectors for comparison.
        """
        try:
            print("\nVerifying index contents...")
            index = faiss.read_index(self.index_path)
            
            # Basic checks
            print(f"Index size: {os.path.getsize(self.index_path)} bytes")
            print(f"Number of vectors: {index.ntotal}")
            print(f"Vector dimension: {index.d}")
            
            if original_vectors is not None and original_ids is not None:
                # Verify each vector if originals provided
                for i, (vector, id_) in enumerate(zip(original_vectors, original_ids)):
                    D, I = index.search(vector.reshape(1, -1), 1)
                    if I[0][0] != id_:
                        print(f"WARNING: Vector {i} has incorrect ID mapping")
                        return False
                print("All vectors verified successfully")
            else:
                # Perform general verification
                success = True
                if index.ntotal > 0:
                    try:
                        # Test search functionality
                        test_vector = index.reconstruct(0)
                        D, I = index.search(test_vector.reshape(1, -1), 1)
                        print("Search functionality verified")
                    except Exception as e:
                        print(f"WARNING: Search verification failed: {e}")
                        success = False
                
                # Verify ID mapping if applicable
                if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                    id_stats = self.analyze_id_mapping()
                    if 'error' in id_stats:
                        print(f"WARNING: ID mapping verification failed: {id_stats['error']}")
                        success = False
                    else:
                        print("ID mapping verified")
                
                return success
            
            return True
            
        except Exception as e:
            print(f"Error verifying index: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='FAISS Index Diagnostic Tool')
    parser.add_argument('--index-path', type=str, required=True,
                      help='Path to the FAISS index file')
    parser.add_argument('--action', type=str, 
                      choices=['inspect', 'verify', 'analyze-ids', 'full'],
                      default='full', help='Diagnostic action to perform')
    
    args = parser.parse_args()
    diagnostic = FAISSIndexDiagnostic(args.index_path)
    
    if args.action in ['inspect', 'full']:
        print("\n=== Current Index Status ===")
        stats = diagnostic.inspect_existing_index()
        if 'error' in stats:
            print(f"Inspection error: {stats['error']}")
    
    if args.action in ['verify', 'full']:
        print("\n=== Index Verification ===")
        result = diagnostic.verify_index()
        print(f"Verification {'successful' if result else 'failed'}")
    
    if args.action in ['analyze-ids', 'full']:
        print("\n=== ID Mapping Analysis ===")
        id_stats = diagnostic.analyze_id_mapping()
        print(json.dumps(id_stats, indent=2))

if __name__ == "__main__":
    main()