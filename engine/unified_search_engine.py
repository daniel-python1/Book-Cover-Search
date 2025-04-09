import os 
import cv2
import numpy as np
import pandas as pd 
from engine.bm25 import BM25
from engine.vsm import VectorSpaceModel
from engine.language_model import DirichletLanguageModel
from engine.utils import load_json, process_text

class UnifiedSearchEngine:
    def __init__(self, index_dir, image_dir):
        # Load index
        self.inverted_index = load_json(os.path.join(index_dir, "inverted_index.json"))
        self.doc_lengths = load_json(os.path.join(index_dir, "doc_lengths.json"))
        self.term_frequencies = load_json(os.path.join(index_dir, "term_frequencies.json"))
        self.metadata = load_json(os.path.join(index_dir, "metadata.json"))

        total_docs = self.metadata.get("total_docs", len(self.doc_lengths))

        # Initialize models
        self.bm25 = BM25(self.inverted_index, self.doc_lengths, self.term_frequencies, total_docs)
        self.vsm = VectorSpaceModel(self.inverted_index, self.doc_lengths, total_docs, self.term_frequencies)
        self.lm = DirichletLanguageModel(self.inverted_index, self.doc_lengths, self.term_frequencies, total_docs)

        # Load CSV metadata
        csv_path = os.path.join(index_dir, "enhanced_book_metadata.csv")
        self.book_data = pd.read_csv(csv_path)
        self.book_data["doc_id"] = self.book_data.index.astype(str)

        # Add image annotation (dominant color)
        self.image_dir = image_dir
        if "dominant_color" not in self.book_data.columns:
            print("Annotating images with dominant color...")
            self.book_data["dominant_color"] = self.book_data["filename"].apply(self.get_dominant_color)

    def get_dominant_color(self, filename):
        try:
            # Use os.path.basename to extract just the filename
            base_filename = os.path.basename(filename)
        
        # Create the full path using normpath to handle any slash differences
            path = os.path.normpath(os.path.join(self.image_dir, base_filename))
        
            if not os.path.exists(path):
            # Try with URL encoding the filename
                from urllib.parse import unquote
                decoded_filename = unquote(base_filename)
                path = os.path.normpath(os.path.join(self.image_dir, decoded_filename))
            
                if not os.path.exists(path):
                    return "unknown"
        
            image = cv2.imread(path)
            if image is None:
                return "error_reading"
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.reshape((-1, 3))
            image = np.float32(image)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(image, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            color = centers[0].astype(int)
            return f"rgb({color[0]}, {color[1]}, {color[2]})"
        except:
            return "error"

    def search(self, query_string, model="bm25", top_k=20):
        query_terms = process_text(query_string)
        print("🔍 Processed Query Terms:", query_terms)
        if not query_terms:
            return []
            
        if model == "bm25":
            results = self.bm25.search(query_terms, top_k=top_k, use_query_expansion=False)
        elif model == "vsm":
            results = self.vsm.search(query_terms, top_k=top_k)
        elif model == "lm":
            results = self.lm.search(query_terms, top_k=top_k)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        # Instead of using .loc which can fail, get results one by one
        ordered_results = []
        for doc_id, _ in results:
            # Convert to string for consistent comparison
            str_doc_id = str(doc_id)
            matching_rows = self.book_data[self.book_data["doc_id"].astype(str) == str_doc_id]
            if not matching_rows.empty:
                book_info = matching_rows.iloc[0].to_dict()
                ordered_results.append(book_info)
        
        return ordered_results