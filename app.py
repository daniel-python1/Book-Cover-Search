from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import sys
from urllib.parse import quote, unquote

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
    
    if not query:
        return render_template('results.html', books=[], query='')
    
    # Determine appropriate top_k based on query type
    if len(query.split()) <= 1:  # Single word queries might be more specific
        base_top_k = 20
    elif any(word.lower() in query.lower() for word in ['title', 'book']):
        base_top_k = 10  # More specific searches get fewer results
    else:
        base_top_k = 30  # Broader searches get more results
    
    # Allow user override with explicit top_k parameter
    top_k = int(request.args.get('top_k', base_top_k))
    
    # Use the search engine to find relevant books
    results = search_engine.search(query, model=model, top_k=top_k)
    
    # Return JSON if requested (for AJAX)
    if request.args.get('format') == 'json':
        return jsonify(results)
    
    # Otherwise render the results page
    return render_template('results.html', books=results, query=query, model=model)

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