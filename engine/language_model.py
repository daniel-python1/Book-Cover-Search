import math
from collections import Counter, defaultdict
import os
import numpy as np
from engine.utils import load_json, parse_queries


class DirichletLanguageModel:
    def __init__(self, inverted_index, doc_lengths, term_frequencies, total_docs):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.term_frequencies = term_frequencies
        self.total_docs = total_docs
        
        # Collection statistics
        self.collection_length = sum(self.doc_lengths.values())
        self.avg_doc_length = self.collection_length / len(self.doc_lengths)
        
        # Dirichlet smoothing parameter (mu)
        self.mu = 500  # Optimal value from tuning
        
        # Pre-compute collection statistics for better performance
        self.collection_probs = self._compute_collection_probs()
        
        # Cache document length ratios
        doc_ids = np.array(list(self.doc_lengths.keys()))
        lengths = np.array([self.doc_lengths[doc_id] for doc_id in doc_ids])
        self.length_ratios = dict(zip(doc_ids, lengths / self.avg_doc_length))
        
    def _compute_collection_probs(self):
        """Pre-compute collection probabilities for all terms"""
        collection_freqs = defaultdict(int)
        
        # Count total term frequencies across all documents
        for doc_terms in self.term_frequencies.values():
            for term, freq in doc_terms.items():
                collection_freqs[term] += freq
        
        # Convert to probabilities with smoothing
        total_terms = sum(collection_freqs.values())
        vocab_size = len(collection_freqs)
        
        # Add small constant for smoothing
        epsilon = 0.0001
        collection_probs = {
            term: (freq + epsilon) / (total_terms + epsilon * vocab_size)
            for term, freq in collection_freqs.items()
        }
        
        return collection_probs

    def score_document(self, query_terms, doc_id):
        """Score document using Dirichlet-smoothed language model"""
        if doc_id not in self.doc_lengths or self.doc_lengths[doc_id] == 0:
            return float('-inf')
        
        doc_length = self.doc_lengths[doc_id]
        doc_terms = self.term_frequencies.get(doc_id, {})
        score = 0.0
        
        # Score each query term
        for term in query_terms:
            # Document probability
            term_freq = doc_terms.get(term, 0)
            
            # Collection probability (smoothed)
            collection_prob = self.collection_probs.get(term, 1e-10)
            
            # Dirichlet smoothing
            numerator = term_freq + self.mu * collection_prob
            denominator = doc_length + self.mu
            
            prob = numerator / denominator
            
            # Add log probability
            if prob > 0:
                score += math.log(prob)
        
        return score

    def search(self, query_terms, top_k=100):
        """Search using Dirichlet-smoothed language model"""
        # Get candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        if not candidate_docs:
            return []
        
        # Score documents
        scores = {}
        for doc_id in candidate_docs:
            score = self.score_document(query_terms, doc_id)
            if score != float('-inf'):
                scores[doc_id] = score
        
        if not scores:
            return []
        
        # Convert to numpy arrays for faster sorting
        doc_ids = np.array(list(scores.keys()))
        score_values = np.array(list(scores.values()))
        
        # Optimize top-k selection
        if len(score_values) > top_k:
            ind = np.argpartition(score_values, -top_k)[-top_k:]
            ind = ind[np.argsort(-score_values[ind])]
        else:
            ind = np.argsort(-score_values)
            
        return list(zip(doc_ids[ind], score_values[ind]))

def run_LM(queries_path, index_dir, output_path):
    """Run the Dirichlet Language Model"""
    # Load index files
    inverted_index = load_json(os.path.join(index_dir, "inverted_index.json"))
    doc_lengths = load_json(os.path.join(index_dir, "doc_lengths.json"))
    term_frequencies = load_json(os.path.join(index_dir, "term_frequencies.json"))
    metadata = load_json(os.path.join(index_dir, "metadata.json"))

    print("Starting Language Model search...")
    
    # Use renumbered queries file if it exists
    renumbered_queries_path = os.path.join(os.path.dirname(queries_path), "cran_queries_renumbered.xml")
    if os.path.exists(renumbered_queries_path):
        print("Using renumbered queries file...")
        queries_path = renumbered_queries_path
    else:
        print("Warning: Renumbered queries file not found, using original file...")

    queries = parse_queries(queries_path)
    lm = DirichletLanguageModel(inverted_index, doc_lengths, term_frequencies, metadata["total_docs"])

    with open(output_path, "w") as f:
        result_count = 0
        for query_id, query_terms in queries.items():
            if not query_terms:
                print(f"WARNING: Query {query_id} has no terms")
                continue
                
            results = lm.search(query_terms)
            
            for rank, (doc_id, score) in enumerate(results, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} lm\n")
                result_count += 1

        print(f"Finished Language Model search. Found {result_count} results")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        queries_path = os.path.join(data_dir, "cran.qry.xml")
        index_dir = os.path.join(data_dir, "index")
        output_path = os.path.join(data_dir, "results", "results_LM.txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        run_LM(queries_path, index_dir, output_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()