from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import sys
from urllib.parse import quote, unquote
from spellchecker import SpellChecker
import re

# Import the search engine and genre_dict from the unified module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine.unified_search_engine import UnifiedSearchEngine, genre_dict

def correct_query(query):
    print(f"RAW BEFORE SANITIZATION: {repr(query)}")
    query = re.sub(r'\s+', ' ', query).strip().replace('\n', '').replace('\r', '')
    spell = SpellChecker()

    # Get all genre keywords
    genre_keywords = set()
    for syn_list in genre_dict.values():
        for syn in syn_list:
            genre_keywords.add(syn.lower())

    # First protect multi-word genre terms to prevent spell-checking them individually
    protected_terms = {}
    for phrase in sorted(genre_keywords, key=lambda x: -len(x)):
        if " " in phrase:
            # Create a unique placeholder for each phrase
            placeholder = f"__GENRE_{len(protected_terms)}__"
            protected_terms[placeholder] = phrase
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            query = pattern.sub(placeholder, query)

    words = query.split()
    corrected_words = []

    for word in words:
        # Skip correcting placeholders for genre terms
        if word in protected_terms:
            corrected_words.append(protected_terms[word])
            continue

        # Skip correcting years and numbers
        if re.match(r'^(18|19|20)\d{2}s?$', word.lower()) or word.lower().isdigit():
            print(f"Skipping year/number: {word}")
            corrected_words.append(word)
            continue

        # Skip correcting single-word genre terms
        if word.lower() in genre_keywords:
            corrected_words.append(word)
            continue

        # Handle "science fiction" or "sci fi" formats specially  
        if word.lower() in ['science', 'sci']:
            next_index = words.index(word) + 1
            if next_index < len(words) and words[next_index].lower() in ['fiction', 'fi']:
                corrected_words.append(word)
                continue

        # Skip correcting colors
        color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'brown', 'grey', 'gray']
        if word.lower() in color_keywords:
            corrected_words.append(word)
            continue

        # Reduce repeated characters (like "boooook" to "book")
        if word.isalpha():
            word = re.sub(r'(.)\1{2,}', r'\1\1', word)

        # Apply spell correction for non-genre, non-year, non-color terms
        corrected = spell.correction(word)
        if corrected and corrected != word:
            print(f"Correcting: {word} → {corrected}")
        corrected_words.append(corrected if corrected else word)

    final = ' '.join(corrected_words)
    print(f"FINAL QUERY: {final}")
    return final

app = Flask(__name__)

index_dir = "data/index"
image_dir = "book_covers"
search_engine = UnifiedSearchEngine(index_dir, image_dir)

@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path) if path else ''

@app.template_filter('urlencode')
def urlencode_filter(s):
    if s is None:
        return ''
    return quote(str(s))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    print(f"RAW QUERY RECEIVED: '{query}'")
    term_mappings = {
        "90s": "1990s",
        "80s": "1980s",
        "70s": "1970s",
        "60s": "1960s"
    }
    
    # Apply mappings
    for old_term, new_term in term_mappings.items():
        # Only replace full words, not parts of words
        query = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, query, flags=re.IGNORECASE)
    
    if query != request.args.get('query', ''):
        print(f"QUERY MAPPED TO: '{query}'")
    
    model = request.args.get('model', 'bm25')
    page = int(request.args.get('page', 1))
    results_per_page = 30
    if not query:
        return render_template('results.html', books=[], query='')
    model = request.args.get('model', 'bm25')
    page = int(request.args.get('page', 1))
    results_per_page = 30
    if not query:
        return render_template('results.html', books=[], query='')

    # Process the query with spell checking
    corrected_query = correct_query(query)
    
    # Get results using the updated search engine
    results = search_engine.search(corrected_query, model=model, top_k=results_per_page*page)
    
    # If no results were found and the query was corrected, try the original query as a fallback
    if not results and corrected_query != query:
        print(f"No results with corrected query. Trying original query: '{query}'")
        results = search_engine.search(query, model=model, top_k=results_per_page*page)

    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    page_results = results[start_index:end_index]

    if request.args.get('format') == 'json':
        return jsonify(results)

    return render_template('results.html', books=page_results, query=query, model=model, page=page)

@app.route('/book_covers/<path:filename>')
def serve_book_cover(filename):
    try:
        return send_from_directory('book_covers', filename)
    except:
        try:
            decoded_filename = unquote(filename)
            return send_from_directory('book_covers', decoded_filename)
        except:
            return send_from_directory('static', 'placeholder.jpg')

@app.route('/api')
def api_docs():
    return render_template('api.html')

if __name__ == '__main__':
    app.run(debug=True)