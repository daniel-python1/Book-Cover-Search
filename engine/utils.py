import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json 
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


#Save and load JSON files 
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    

def download_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")
        #download_nltk()

#Processing Text Function 
def process_text(text, preserve_special=False):
    if not text:
        return []

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    processed_tokens = []
    for token in tokens:
        # ✅ Keep 4-digit numbers like years
        if token.isdigit() and len(token) == 4:
            processed_tokens.append(token)
            continue

        if len(token) < 2:
            continue

        if token in stop_words:
            continue

        token = stemmer.stem(token)
        processed_tokens.append(token)

    return processed_tokens



#Parsing the document 
def parse_doc(xml_file):
    """Enhanced document parsing with better structure handling"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return {}, defaultdict(lambda: defaultdict(int))

    # Store term frequencies and positions
    doc_terms = {}  # {doc_id: {term: freq}}
    inverted_index = defaultdict(lambda: defaultdict(int))  # {term: {doc_id: freq}}
    
    skipped_docs = 0
    processed_docs = 0

    for doc in root.findall(".//doc"):
        doc_ID = doc.find("docno")
        if doc_ID is None:
            skipped_docs += 1
            continue
            
        doc_ID = doc_ID.text.strip()
        
        # Collect text from all relevant fields
        text_elements = []
        for field in ['text', 'headline', 'title', 'abstract']:
            elem = doc.find(field)
            if elem is not None and elem.text:
                text_elements.append(elem.text.strip())
        
        if not text_elements:
            skipped_docs += 1
            continue
            
        # Process text with special character preservation
        text = " ".join(text_elements)
        tokens = process_text(text, preserve_special=True)
        
        if not tokens:
            skipped_docs += 1
            continue

        # Store term frequencies
        term_freqs = Counter(tokens)
        doc_terms[doc_ID] = dict(term_freqs)
        
        # Update inverted index with frequencies
        for term, freq in term_freqs.items():
            inverted_index[term][doc_ID] = freq
        
        processed_docs += 1

    print(f"Processed {processed_docs} documents")
    print(f"Skipped {skipped_docs} documents")
    return doc_terms, dict(inverted_index)


#Parse Queries
def parse_queries(queries_path):
    """Enhanced query parsing with better term handling"""
    queries = {}
    try:
        tree = ET.parse(queries_path)
        root = tree.getroot()
        
        for topic in root.findall('.//top'):
            query_id = topic.find('num').text.strip()
            
            # Collect text from all relevant fields
            query_parts = []
            for field in ['title', 'desc', 'narr']:
                elem = topic.find(field)
                if elem is not None and elem.text:
                    query_parts.append(elem.text.strip())
            
            # Process query text with special character preservation
            query_text = " ".join(query_parts)
            tokens = process_text(query_text, preserve_special=True)
            
            if tokens:
                queries[query_id] = tokens
                
    except Exception as e:
        print(f"Error parsing queries: {e}")
        return {}
        
    return queries

def preprocess_text(self, text):
    # Input validation
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid input text type: {type(text)}")
        return []

    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords if they exist
        if hasattr(self, 'stopwords'):
            tokens = [token for token in tokens if token not in self.stopwords]
            
        # Stemming if stemmer exists
        if hasattr(self, 'stemmer'):
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        # Final validation
        tokens = [token for token in tokens if token and len(token) > 1]
        
        if not tokens:
            print(f"Warning: No valid tokens produced for text")
            return []
            
        return tokens
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return []

def read_file(file_path):
    """Simple file reading with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""