import math
import os 
import numpy as np
from engine.utils import parse_queries, load_json, process_text
from collections import Counter, defaultdict



class VectorSpaceModel:
    def __init__(self, inverted_index, doc_lengths, total_docs, term_frequencies):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.total_docs = total_docs
        self.term_frequencies = term_frequencies 
        
        # Pre-compute IDF values and document vectors for better performance
        self.idf_values = self.calculate_idf_values()
        self.doc_vectors = self.precompute_doc_vectors()
        
    def calculate_idf_values(self):
        idf_values = {}
        for term, postings in self.inverted_index.items():
            doc_freq = len(postings)
            # Improved IDF formula with better smoothing
            idf_values[term] = math.log((self.total_docs + 2) / (doc_freq + 0.5))
        return idf_values
    
    def precompute_doc_vectors(self):
        """Pre-compute document vectors with improved weighting"""
        doc_vectors = defaultdict(dict)
        avg_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        for doc_id, term_freqs in self.term_frequencies.items():
            doc_len = self.doc_lengths[doc_id]
            if doc_len == 0:
                continue
                
            for term, freq in term_freqs.items():
                if term in self.idf_values:
                    # BM25-inspired TF normalization
                    k1, b = 1.5, 0.75
                    tf = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len/avg_len))
                    doc_vectors[doc_id][term] = tf * self.idf_values[term]
            
            # L2 normalization
            magnitude = math.sqrt(sum(w * w for w in doc_vectors[doc_id].values()))
            if magnitude > 0:
                doc_vectors[doc_id] = {t: w/magnitude for t, w in doc_vectors[doc_id].items()}
        
        return dict(doc_vectors)

    def calculate_query_vector(self, query_terms):
        """Calculate query vector with improved weighting"""
        query_vector = {}
        terms_count = Counter(query_terms)
        query_len = len(query_terms)
        
        # Query term weighting with BM25-inspired normalization
        k1 = 1.5
        for term, count in terms_count.items():
            if term in self.idf_values:
                tf = count * (k1 + 1) / (count + k1)
                query_vector[term] = tf * self.idf_values[term]
                
        # L2 normalization
        magnitude = math.sqrt(sum(w * w for w in query_vector.values()))
        if magnitude > 0:
            query_vector = {term: weight/magnitude for term, weight in query_vector.items()}
            
        return query_vector

    def search(self, query_terms, top_k=100):
        query_vector = self.calculate_query_vector(query_terms)
        
        if not query_vector:
            return []
        
        scores = defaultdict(float)
        
        # Improved scoring with candidate pruning
        candidate_docs = set()
        for term in query_vector:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
        
        # Calculate cosine similarity only for candidate documents
        for doc_id in candidate_docs:
            if doc_id in self.doc_vectors:
                doc_vector = self.doc_vectors[doc_id]
                score = sum(query_vector[term] * doc_vector.get(term, 0) 
                          for term in query_vector)
                if score > 0:
                    scores[doc_id] = score
        
        # Convert to numpy arrays for faster sorting
        doc_ids = np.array(list(scores.keys()))
        score_values = np.array(list(scores.values()))
        
        if len(score_values) > top_k:
            ind = np.argpartition(score_values, -top_k)[-top_k:]
            ind = ind[np.argsort(-score_values[ind])]
            sorted_scores = list(zip(doc_ids[ind], score_values[ind]))
        else:
            ind = np.argsort(-score_values)
            sorted_scores = list(zip(doc_ids[ind], score_values[ind]))
            
        return sorted_scores

def run_vsm(queries_path, index_dir, output_path):
    # Load index files
    inverted_index = load_json(os.path.join(index_dir, "inverted_index.json"))
    doc_lengths = load_json(os.path.join(index_dir, "doc_lengths.json"))
    term_frequencies = load_json(os.path.join(index_dir, "term_frequencies.json"))
    metadata = load_json(os.path.join(index_dir, "metadata.json"))

    print("Starting VSM search...")
    
    # Use renumbered queries file if it exists
    renumbered_queries_path = os.path.join(os.path.dirname(queries_path), "cran_queries_renumbered.xml")
    if os.path.exists(renumbered_queries_path):
        print("Using renumbered queries file...")
        queries_path = renumbered_queries_path
    else:
        print("Warning: Renumbered queries file not found, using original file...")

    queries = parse_queries(queries_path)
    vsm = VectorSpaceModel(inverted_index, doc_lengths, metadata["total_docs"], term_frequencies)

    with open(output_path, 'w') as f:
        result_count = 0
        for query_id, query_terms in queries.items():
            if not query_terms:
                print(f"WARNING: Query {query_id} has no terms")
                continue

            results = vsm.search(query_terms)
            
            for rank, (doc_id, score) in enumerate(results, 1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.6f} vsm\n"
                f.write(line)
                result_count += 1

        print(f"Finished VSM search. Found {result_count} results")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        queries_path = os.path.join(data_dir, "cran.qry.xml")
        index_dir = os.path.join(data_dir, "index")
        output_path = os.path.join(data_dir, "results", "vsm_results.txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        run_vsm(queries_path, index_dir, output_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
                    