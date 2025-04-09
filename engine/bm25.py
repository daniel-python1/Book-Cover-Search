import math 
from collections import defaultdict 
import os 
from engine.utils import parse_queries, load_json
import numpy as np

class BM25:
    def __init__(self, inverted_index, doc_lengths, term_frequencies, total_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = inverted_index
        self.term_frequencies = term_frequencies 
        self.doc_lengths = doc_lengths
        self.total_docs = total_docs
        
        # Add epsilon to avoid division by zero
        self.avgdl = sum(self.doc_lengths.values()) / self.total_docs + 1e-8 if self.total_docs > 0 else 1e-8
        
        # Pre-compute more values for performance
        self.k1_plus_1 = k1 + 1
        self.b_complement = 1 - b
        
        # Cache document length ratios with numpy for better performance
        doc_ids = np.array(list(self.doc_lengths.keys()))
        lengths = np.array([self.doc_lengths[doc_id] for doc_id in doc_ids])
        self.length_ratios = dict(zip(doc_ids, lengths / self.avgdl))
        
        # Pre-compute IDF scores
        self.idf = self.compute_bm25_idf()
        
        # Check if WordNet is available
        try:
            from nltk.corpus import wordnet
            self.wordnet_available = True
        except LookupError:
            print("Warning: WordNet resource not found. Query expansion will be disabled.")
            print("To enable query expansion, run: python -c 'import nltk; nltk.download(\"wordnet\")'")
            self.wordnet_available = False

    def compute_bm25_idf(self):
        idf_values = {}
        N = self.total_docs

        for term, doc_ids in self.inverted_index.items():
            n_t = len(doc_ids)
            # Add epsilon to avoid log(0)
            idf = math.log((N - n_t + 0.5) / (n_t + 0.5 + 1e-8) + 1.0)
            idf_values[term] = max(0.0, idf)
        
        return idf_values
    
    def get_synonyms(self, term):
        """Get WordNet synonyms for query expansion"""
        if not self.wordnet_available:
            return set()
        
        synonyms = set()
        try:
            from nltk.corpus import wordnet
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())
        except Exception as e:
            print(f"Warning: Error getting synonyms for '{term}': {e}")
        return synonyms

    def search(self, query_terms, top_k=100, use_query_expansion=True):
        # Use sets for faster lookup
        query_terms_set = set(query_terms)
        expanded_terms = query_terms_set.copy()
        
        if use_query_expansion and self.wordnet_available:
            # Limit synonym expansion to avoid query drift
            for term in list(query_terms_set)[:3]:  # Only expand top 3 terms
                synonyms = self.get_synonyms(term)
                expanded_terms.update(synonyms)
        
        # Optimize term weights dictionary creation
        term_weights = {term: 1.0 if term in query_terms_set else 0.3 for term in expanded_terms}
        
        # Pre-allocate scores array for better performance
        scores = defaultdict(float)
        
        # Score calculation optimization
        for term in expanded_terms:
            if term not in self.inverted_index or term not in self.idf:
                continue
                
            idf = self.idf[term]
            if idf <= 0:
                continue

            term_weight = term_weights[term]
            doc_ids = self.inverted_index[term]
            
            # Vectorized score calculation
            for doc_id in doc_ids:
                tf = self.term_frequencies[doc_id].get(term, 0)
                if tf == 0:
                    continue
                    
                length_ratio = self.length_ratios[doc_id]
                denominator = tf + self.k1 * (self.b_complement + self.b * length_ratio)
                term_score = idf * (tf * self.k1_plus_1 / denominator) * term_weight
                scores[doc_id] += term_score

        # Use numpy for faster sorting
        if not scores:
            return []
            
        doc_ids = np.array(list(scores.keys()))
        score_values = np.array(list(scores.values()))
        
        # Optimize top-k selection
        if len(score_values) > top_k:
            ind = np.argpartition(score_values, -top_k)[-top_k:]
            ind = ind[np.argsort(-score_values[ind])]
        else:
            ind = np.argsort(-score_values)
            
        return list(zip(doc_ids[ind], score_values[ind]))

def run_bm25_model(queries_path, index_dir, output_path, k1=1.5, b=0.75):
    # Load index files
    inverted_index = load_json(os.path.join(index_dir, "inverted_index.json"))
    doc_lengths = load_json(os.path.join(index_dir, "doc_lengths.json"))
    term_frequencies = load_json(os.path.join(index_dir, "term_frequencies.json"))
    metadata = load_json(os.path.join(index_dir, "metadata.json"))

    print("Starting to run the BM25 model...")
    
    # Use renumbered queries file if it exists
    renumbered_queries_path = os.path.join(os.path.dirname(queries_path), "cran_queries_renumbered.xml")
    if os.path.exists(renumbered_queries_path):
        print("Using renumbered queries file...")
        queries_path = renumbered_queries_path
    else:
        print("Warning: Renumbered queries file not found, using original file...")

    queries = parse_queries(queries_path)
    
    # Initialize BM25 with tuned parameters
    bm25 = BM25(inverted_index, doc_lengths, term_frequencies, metadata["total_docs"], k1, b)

    with open(output_path, "w") as f:
        result_count = 0
        for query_id, query_terms in queries.items():
            if not query_terms:
                print(f"WARNING: Query {query_id} has no terms")
                continue

            results = bm25.search(query_terms, top_k=100, use_query_expansion=True)

            for rank, (doc_id, score) in enumerate(results, start=1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.6f} bm25\n"
                f.write(line)
                result_count += 1

        print(f"Finished running the BM25 model. Found {result_count} results")
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        queries_path = os.path.join(data_dir, "cran.qry.xml")
        index_dir = os.path.join(data_dir, "index")
        output_path = os.path.join(data_dir, "results", "results_bm25.txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        run_bm25_model(queries_path, index_dir, output_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    