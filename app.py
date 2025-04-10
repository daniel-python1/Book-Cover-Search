from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import sys
from urllib.parse import quote, unquote
from spellchecker import SpellChecker
import re

#Mapping similar words 
genre_dict = {
    "fantasy": ["fantasy", "fantasy books", "magical", "magic", "wizards", "dragons", "fairy tale", "mythology", "elves", "mystical", "sorcery"],
    "science_fiction": ["science fiction", "sci fi", "space opera", "futuristic", "aliens", "robots", "space exploration", "time travel", "cyberpunk", "dystopian", "post-apocalyptic", "artificial intelligence"],
    "romance": ["romance", "romantic", "love", "love story", "relationship", "romantic books", "romantic fiction", "passion", "couple", "dating", "affair", "heartbreak"],
    "mystery": ["mystery", "mystery books", "detective", "crime", "whodunit", "suspense", "murder", "investigation", "crime novel", "detective story", "mystery thriller"],
    "horror": ["horror", "horror books", "scary", "spooky", "thriller", "haunted", "ghost", "paranormal", "supernatural", "psychological horror", "dark", "monsters"],
    "historical_fiction": ["historical fiction", "historical", "period drama", "historical books", "historical novel", "historical romance", "war novels", "medieval", "victorian", "renaissance", "ancient"],
    "young_adult": ["young adult", "ya", "teen", "teen fiction", "teen books", "adolescent", "young readers", "coming of age", "teen drama", "young adult fiction"],
    "adventure": ["adventure", "adventure books", "action", "exploration", "journey", "quest", "survival", "explorer", "expedition", "adventure novels", "action books"],
    "thriller": ["thriller", "thriller books", "suspense", "action", "crime thriller", "psychological thriller", "spy", "espionage", "political thriller", "mystery thriller", "action thriller"],
    "poetry": ["poetry", "poem", "poems", "verse", "rhymes", "poet", "spoken word", "rhyme", "lyrics"],
    "biography": ["biography", "biographies", "memoir", "autobiography", "life story", "life history", "personal story", "real life", "real life stories", "true story"],
    "autobiography": ["autobiography", "self biography", "memoir", "personal story", "self-written biography", "life story"],
    "children": ["children", "kids", "children's books", "children books", "kids books", "juvenile", "kids fiction", "children fiction", "story books", "picture books"],
    "drama": ["drama", "plays", "stage", "theater", "performance", "dramatic", "drama books", "acting", "stage play"],
    "classic_literature": ["classic literature", "classics", "classic books", "timeless books", "vintage literature", "literary classics", "old literature"],
    "sports": ["sports", "sports books", "athletics", "sport books", "team sports", "individual sports", "sports fiction", "extreme sports", "soccer books", "basketball books"],
    "technology": ["technology", "tech", "science and technology", "tech books", "innovation", "gadgets", "electronics", "computing", "IT", "technology books", "future tech"],
    "philosophy": ["philosophy", "philosophical", "ethics", "logic", "philosophical books", "thinking", "metaphysics", "existentialism", "plato", "socratic", "philosophy books"],
    "psychology": ["psychology", "psychological", "mental health", "behavior", "psychology books", "cognitive science", "neuroscience", "therapy", "human behavior"],
    "religion": ["religion", "spirituality", "faith", "bible", "holy books", "christianity", "buddhism", "islam", "religious", "spiritual", "catholic", "holy scripture"],
    "education": ["education", "learning", "teaching", "study", "academic", "school books", "educational books", "college", "university", "educational resources", "education books"]
}

def normalize_genre(query, genre_dict):
    """
    Normalizes the query by mapping it to a genre if a known genre term is found.
    Returns the genre name if found, or None if no genre matches.
    """
    query = query.lower()

    # Check if the query matches any genre terms
    for genre, synonyms in genre_dict.items():
        for synonym in synonyms:
            if synonym.lower() == query:
                return genre
            elif synonym.lower() in query:
                return genre  # Return the genre if a part of the query matches
    return None  # Return None if no genre matches



def correct_query(query):
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'(.)\1{2,}', r'\1\1', query)
    spell = SpellChecker()
    words = query.split()
    corrected_words = [spell.correction(word) for word in words]

    # Combine the corrected words back into a string
    corrected_query = ' '.join(corrected_words)

    return corrected_query


# Add the project root to the Python path to import the search engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine.unified_search_engine import UnifiedSearchEngine

app = Flask(__name__)

# Initialize the search engine
index_dir = "data/index"
image_dir = "book_covers"
search_engine = UnifiedSearchEngine(index_dir, image_dir)

# Load enhanced metadata CSV for reference
df = pd.read_csv('C:/Users/comer/OneDrive - Dublin City University/Desktop/Web Scrape/data/index/enhanced_book_metadata.csv')

# Add basename filter for templates
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
    model = request.args.get('model', 'bm25')
    
    # Get the current page from the URL query parameters, default to 1 if not provided
    page = int(request.args.get('page', 1))  # Default to page 1
    results_per_page = 30  # Number of results per page
    
    if not query:
        return render_template('results.html', books=[], query='')

    # Correct spelling before using the query in search
    genre = normalize_genre(query, genre_dict)

    if not genre:
        corrected_query = correct_query(query)
    else:
        corrected_query = query 
    # Step 1: Calculate `top_k` based on the page number
    top_k = results_per_page * page  # This fetches the appropriate number of results for the page

    if genre:
        results = search_engine.search(genre, model=model, top_k=top_k)
    else:
        results = search_engine.search(corrected_query, model=model, top_k=top_k)

    
    # Pagination: Only show results for the current page
    start_index = (page - 1) * results_per_page
    end_index = start_index + results_per_page
    page_results = results[start_index:end_index]  # Get the results for the current page
    
    # Return JSON if requested (for AJAX)
    if request.args.get('format') == 'json':
        return jsonify(results)
    
    # Otherwise render the results page
    return render_template('results.html', books=page_results, query=corrected_query, model=model, page=page)

# Serve book cover images
@app.route('/book_covers/<path:filename>')
def serve_book_cover(filename):
    try:
        # First try the direct filename
        return send_from_directory('book_covers', filename)
    except:
        # If that fails, try URL decoding the filename
        try:
            decoded_filename = unquote(filename)
            return send_from_directory('book_covers', decoded_filename)
        except:
            # If that still fails, return a placeholder
            return send_from_directory('static', 'placeholder.jpg')

# Add a route for API documentation
@app.route('/api')
def api_docs():
    return render_template('api.html')

if __name__ == '__main__':
    app.run(debug=True)