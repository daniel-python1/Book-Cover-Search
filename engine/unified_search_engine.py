import os 
import re
import cv2
import numpy as np
import pandas as pd 
from engine.bm25 import BM25
from engine.vsm import VectorSpaceModel
from engine.language_model import DirichletLanguageModel
from engine.utils import load_json, process_text

genre_dict = {
    "fantasy": ["fantasy", "fantasy books", "magical", "magic", "wizards", "dragons", "fairy tale", "mythology", "elves", "mystical", "sorcery"],
    "science_fiction": ["science fiction", "sci fi", "science_fiction", "sciencefiction", "space opera", "futuristic", "aliens", "robots", "space exploration", "time travel", "cyberpunk", "dystopian", "post-apocalyptic", "artificial intelligence"],
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

class UnifiedSearchEngine:
    def __init__(self, index_dir, image_dir):
        print("Initializing UnifiedSearchEngine...")
        self.genre_dict = genre_dict
        self.index_dir = index_dir
        self.image_dir = image_dir

        # Step 1: Enhance metadata if not already done
        csv_path = os.path.join(index_dir, "enhanced_book_metadata.csv")

        if not os.path.exists(csv_path):
            print("Enhancing metadata with dominant color and surrogate text...")

            original_csv_path = os.path.join(index_dir, "book_metadata.csv")
            df = pd.read_csv(original_csv_path)

            df['doc_id'] = df.index.astype(str)

            def get_dominant_color_internal(filename):
                try:
                    base_filename = os.path.basename(filename)
                    full_path = os.path.normpath(os.path.join(self.image_dir, base_filename))
                    print(f"Attempting to process: {full_path}")

                    if not os.path.exists(full_path):
                        from urllib.parse import unquote
                        decoded_filename = unquote(base_filename)
                        full_path = os.path.normpath(os.path.join(self.image_dir, decoded_filename))
                        if not os.path.exists(full_path):
                            print(f"Image not found: {full_path}")
                            return "unknown"

                    image = cv2.imread(full_path)
                    if image is None:
                        print(f"Failed to read image: {full_path}")
                        return "error_reading"

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.reshape((-1, 3))
                    image = np.float32(image)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(image, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    color = centers[0].astype(int)
                    return f"rgb({color[0]}, {color[1]}, {color[2]})"
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                    return "error"

            df['dominant_color'] = df['filename'].apply(get_dominant_color_internal)

            df['surrogate_text'] = df.apply(
                lambda row: f"{row['title']} {row['authors']} {row['subject']} {row['year']} {row['dominant_color']}",
                axis=1
            )

            df.to_csv(csv_path, index=False)

            documents = {}
            for _, row in df.iterrows():
                doc_id = row['doc_id']
                text = row['surrogate_text']
                documents[doc_id] = process_text(text)

            with open(os.path.join(index_dir, 'documents.json'), 'w') as f:
                json.dump(documents, f)

            print(f"Enhanced metadata and processed {len(documents)} documents for indexing.")

        # Proceed with standard engine init
        self.inverted_index = load_json(os.path.join(index_dir, "inverted_index.json"))
        self.doc_lengths = load_json(os.path.join(index_dir, "doc_lengths.json"))
        self.term_frequencies = load_json(os.path.join(index_dir, "term_frequencies.json"))
        self.metadata = load_json(os.path.join(index_dir, "metadata.json"))

        self.color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'brown', 'grey', 'gray']

        self.common_words = ['book', 'books', 'cover', 'covers', 'with', 'a', 'the', 'of', 'in', 'on', 'and', 'or', 'has', 'have', 'find', 'me', 'show', 'get', 'color', 'colors', 'colored']

        self.decade_patterns = {
            r'90s\b': (1990, 1999),
            r'80s\b': (1980, 1989),
            r'70s\b': (1970, 1979),
            r'60s\b': (1960, 1969),
            r'50s\b': (1950, 1959),
            r'2000s\b': (2000, 2009),
            r'2010s\b': (2010, 2019),
            r'2020s\b': (2020, 2029),
            r'\bnineties\b': (1990, 1999),
            r'\beighties\b': (1980, 1989),
            r'\bseventies\b': (1970, 1979),
            r'\bsixties\b': (1960, 1969),
            r'\bfifties\b': (1950, 1959)
        }

        total_docs = self.metadata.get("total_docs", len(self.doc_lengths))
        print(f"Total documents: {total_docs}")

        print("Initializing BM25 model...")
        self.bm25 = BM25(self.inverted_index, self.doc_lengths, self.term_frequencies, total_docs)

        self._vsm = None
        self._lm = None

        print("Loading book metadata...")
        self.book_data = pd.read_csv(csv_path)
        self.book_data["doc_id"] = self.book_data.index.astype(str)

        if "dominant_color" not in self.book_data.columns:
            print("Annotating images with dominant color...")
            self.book_data["dominant_color"] = self.book_data["filename"].apply(self.get_dominant_color)

        print("Normalizing genre data...")
        self.book_data["normalized_subject"] = self.book_data["subject"].apply(self._normalize_genre)

        print("UnifiedSearchEngine initialized successfully.")
        
        # Define color keywords for later use
        self.color_keywords = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'pink', 'brown', 'grey', 'gray']
        
        # Common words that should be ignored when determining if a query is filter-only
        self.common_words = ['book', 'books', 'cover', 'covers', 'with', 'a', 'the', 'of', 'in', 'on', 'and', 'or', 'has', 'have', 'find', 'me', 'show', 'get', 'color', 'colors', 'colored']

        # Define decade patterns and their year ranges
        self.decade_patterns = {
            r'90s\b': (1990, 1999),
            r'80s\b': (1980, 1989),
            r'70s\b': (1970, 1979),
            r'60s\b': (1960, 1969),
            r'50s\b': (1950, 1959),
            r'2000s\b': (2000, 2009),
            r'2010s\b': (2010, 2019),
            r'2020s\b': (2020, 2029),
            r'\bnineties\b': (1990, 1999),
            r'\beighties\b': (1980, 1989),
            r'\bseventies\b': (1970, 1979),
            r'\bsixties\b': (1960, 1969),
            r'\bfifties\b': (1950, 1959)
        }

        total_docs = self.metadata.get("total_docs", len(self.doc_lengths))
        print(f"Total documents: {total_docs}")

        # Initialize only BM25 by default (as it's the most commonly used)
        # Other models will be lazily initialized when needed
        print("Initializing BM25 model...")
        self.bm25 = BM25(self.inverted_index, self.doc_lengths, self.term_frequencies, total_docs)
        
        # Create placeholders for other models that will be lazily initialized
        self._vsm = None
        self._lm = None

        print("Loading book metadata...")
        csv_path = os.path.join(index_dir, "enhanced_book_metadata.csv")
        self.book_data = pd.read_csv(csv_path)
        self.book_data["doc_id"] = self.book_data.index.astype(str)

        self.image_dir = image_dir
        if "dominant_color" not in self.book_data.columns:
            print("Annotating images with dominant color...")
            self.book_data["dominant_color"] = self.book_data["filename"].apply(self.get_dominant_color)
        
        # Add normalized subject for better genre matching
        print("Normalizing genre data...")
        self.book_data["normalized_subject"] = self.book_data["subject"].apply(self._normalize_genre)
        
        print("UnifiedSearchEngine initialized successfully.")
    
    @property
    def vsm(self):
        """Lazy initialization of Vector Space Model"""
        if self._vsm is None:
            print("Initializing Vector Space Model (this may take a moment)...")
            total_docs = self.metadata.get("total_docs", len(self.doc_lengths))
            self._vsm = VectorSpaceModel(self.inverted_index, self.doc_lengths, total_docs, self.term_frequencies)
        return self._vsm
    
    @property
    def lm(self):
        """Lazy initialization of Language Model"""
        if self._lm is None:
            print("Initializing Language Model (this may take a moment)...")
            total_docs = self.metadata.get("total_docs", len(self.doc_lengths))
            self._lm = DirichletLanguageModel(self.inverted_index, self.doc_lengths, self.term_frequencies, total_docs)
        return self._lm
        
    def _normalize_genre(self, genre):
        """Normalize genre string to a standard format for comparison"""
        if not isinstance(genre, str):
            return ""
        # Convert to lowercase, remove extra spaces, replace hyphens with underscores
        norm = genre.lower().strip().replace("-", "_").replace(".", "")
        return norm

    def rgb_to_basic_color(self, rgb_str):
        """Convert RGB values to basic color names with improved thresholds"""
        try:
            # Handle different RGB string formats
            if not rgb_str or not isinstance(rgb_str, str):
                return "unknown"
                
            # Extract RGB values with regex to handle various formats
            match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_str)
            if not match:
                return "unknown"
                
            r, g, b = map(int, match.groups())
            
            # Improved color detection with more forgiving thresholds
            # RED detection - various ways to detect red
            if (r > 150 and g < 120 and b < 120) or (r > 180 and r > g*1.5 and r > b*1.5):
                return "red"
            # GREEN detection
            elif (g > 150 and r < 120 and b < 120) or (g > 180 and g > r*1.5 and g > b*1.5):
                return "green"
            # BLUE detection
            elif (b > 120 and r < 120 and g < 120) or (b > 180 and b > r*1.5 and b > g*1.5) or (b > r and b > g and b > 150):
                return "blue"
            # YELLOW detection 
            elif r > 180 and g > 180 and b < 100:
                return "yellow"
            # PURPLE detection
            elif r > 130 and b > 130 and g < 100:
                return "purple"
            # BLACK detection
            elif r < 60 and g < 60 and b < 60:
                return "black"
            # WHITE detection
            elif r > 200 and g > 200 and b > 200:
                return "white"
            # ORANGE detection
            elif r > 200 and g > 100 and g < 180 and b < 100:
                return "orange"
            # PINK detection
            elif r > 200 and g < 180 and b > 150:
                return "pink"
            # BROWN detection
            elif r > 120 and r < 200 and g > 60 and g < 150 and b < 100:
                return "brown"
            # GREY detection
            elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30 and r > 60 and r < 200:
                return "grey"
            else:
                return "other"
        except Exception as e:
            print(f"Error in color detection: {e}")
            return "unknown"

    def get_dominant_color(self, filename):
        try:
            base_filename = os.path.basename(filename)
            path = os.path.normpath(os.path.join(self.image_dir, base_filename))
            if not os.path.exists(path):
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

    def _check_for_decade(self, query_string):
        """Check if the query contains decade references and return the decade range if found"""
        lower_query = query_string.lower()
        for pattern, year_range in self.decade_patterns.items():
            if re.search(pattern, lower_query):
                return year_range
        return None

    def parse_filters(self, query_string):
        lower_query = query_string.lower()
        
        # Find color terms
        found_colors = [color for color in self.color_keywords if re.search(r'\b' + re.escape(color) + r'\b', lower_query)]
        
        # Find year - check for exact years and decades
        year_info = None
        
        # First check for decade references like "90s", "2000s", etc.
        decade_range = self._check_for_decade(lower_query)
        if decade_range:
            start_year, end_year = decade_range
            print(f"Decade range detected: {start_year}-{end_year}")
            year_info = {"decade": decade_range}
        
        # Then check for exact years (4-digit numbers starting with 18, 19, or 20)
        year_match = re.search(r'\b(18|19|20)\d{2}\b', lower_query)
        if year_match:
            exact_year = int(year_match.group())
            print(f"Exact year detected: {exact_year}")
            # If we found both a decade and an exact year, prioritize the exact year
            year_info = {"exact_year": exact_year}
        
        # Improved genre matching - check for exact matches first, then partial matches
        genre = None
        for key, synonyms in self.genre_dict.items():
            for syn in synonyms:
                # Look for exact matches with word boundaries first
                if re.search(r'\b' + re.escape(syn) + r'\b', lower_query):
                    genre = key
                    print(f"Genre matched (exact): {syn} → {key}")
                    break
            if genre:
                break
                
        # If no exact match found, try more flexible matching (for phrases like "scifi" without space)
        if not genre:
            for key, synonyms in self.genre_dict.items():
                for syn in synonyms:
                    if syn in lower_query:
                        genre = key
                        print(f"Genre matched (flexible): {syn} → {key}")
                        break
                if genre:
                    break
                    
        print(f"FILTER DEBUG — colors: {found_colors}, year: {year_info}, genre: {genre}")
        return found_colors, year_info, genre

    def _is_filters_only_query(self, query_string, parsed_genre, colors, year_info):
        """Determine if a query consists only of filter terms (genre, color, year) with no other search terms"""
        # If we don't have any filters, it's definitely not a filter-only query
        if not (parsed_genre or colors or year_info):
            return False
            
        # Start by removing genre terms from the query
        cleaned_query = query_string.lower()
        if parsed_genre:
            for syn in self.genre_dict.get(parsed_genre, []):
                cleaned_query = re.sub(r'\b' + re.escape(syn) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        
        # Remove color terms
        for color in self.color_keywords:
            cleaned_query = re.sub(r'\b' + re.escape(color) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
            
        # Remove year and decade references
        if year_info:
            # Remove exact years
            if "exact_year" in year_info:
                year_pattern = r'\b(18|19|20)\d{2}\b'
                cleaned_query = re.sub(year_pattern, '', cleaned_query)
            
            # Remove decade references
            for pattern in self.decade_patterns.keys():
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Remove common words
        for word in self.common_words:
            cleaned_query = re.sub(r'\b' + re.escape(word) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        
        # Check if anything meaningful is left
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        is_filters_only = len(cleaned_query) <= 2  # Just spaces or very short
        
        print(f"Query '{query_string}' after removing filters and common words: '{cleaned_query}'")
        print(f"Is filters-only query: {is_filters_only}")
        
        return is_filters_only
    
    def _direct_filter_search(self, genre, colors=None, year_info=None, top_k=20):
        """Directly return books matching filters without using the search model"""
        print(f"Using direct filter search - Genre: '{genre}', Colors: {colors}, Year info: {year_info}")
        
        # Create a DataFrame to work with
        df = self.book_data.copy()
        
        # Special handling for decade searches: ALWAYS use direct filtering
        if year_info and "decade" in year_info:
            start_year, end_year = year_info["decade"]
            
            # Print all the years in our dataset for debugging
            all_years = sorted(df["year"].dropna().unique())
            print(f"Years available in dataset: {all_years[:20]}...")
            
            # Use pandas filtering for better performance
            # Convert year column to numeric, replacing non-numeric values with NaN
            df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
            
            # Filter by the decade range
            df = df[(df["year_numeric"] >= start_year) & (df["year_numeric"] <= end_year)]
            
            print(f"Found {len(df)} books from {start_year}-{end_year}")
            
            # If empty result set, return early
            if len(df) == 0:
                print(f"No books found from {start_year}-{end_year}")
                return []
            
            # Print a sample of years to verify filtering
            years_in_results = sorted(df["year_numeric"].dropna().unique())
            print(f"Years in filtered results: {years_in_results}")
            
            # Print some book titles and years for verification
            sample_books = df.head(5)
            for _, book in sample_books.iterrows():
                print(f"Sample book: {book.get('title')} - Year: {book.get('year')}")
        
        # Apply genre filter if specified
        if genre:
            normalized_genre = self._normalize_genre(genre)
            genre_filtered = df[df["normalized_subject"] == normalized_genre]
            print(f"Found {len(genre_filtered)} books with genre '{normalized_genre}'")
            
            # If no books match the genre, we still want to continue with other filters
            if len(genre_filtered) > 0:
                df = genre_filtered
            else:
                print(f"No books found with genre '{genre}'. Continuing with other filters.")
        
        # Apply color filter if specified
        if colors and len(colors) > 0:
            # Create a new dataframe for color matches to avoid modifying df during iteration
            color_matches = []
            
            # Process in batches to improve performance for large datasets
            batch_size = 500
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                for _, book in batch.iterrows():
                    rgb_color = book.get("dominant_color", "")
                    color_label = self.rgb_to_basic_color(rgb_color)
                    
                    # Debug print for small result sets
                    if len(df) < 100:
                        print(f"Book: {book.get('title')} - Color: {rgb_color} → {color_label}")
                    
                    if any(color_label == user_color for user_color in colors):
                        color_matches.append(book)
            
            # If we found color matches, update our dataframe
            if color_matches:
                df = pd.DataFrame(color_matches)
                print(f"After color filter: {len(df)} books match colors {colors}")
            else:
                print(f"No books matched the specified colors: {colors}")
                return []  # No books match the color filter
        
        # Apply exact year filter if specified
        if year_info and "exact_year" in year_info:
            exact_year = year_info["exact_year"]
            print(f"Looking for books from exactly year {exact_year}")
            
            # Use pandas filtering for better performance
            df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year_numeric"] == exact_year]
            
            print(f"Found {len(df)} books from exactly {exact_year}")
            if len(df) == 0:
                return []  # No books match the exact year
            
        # Convert results to list of dictionaries
        results = [row.to_dict() for _, row in df.iterrows()]
        
        # Print sample results for debugging
        if results:
            print(f"Found {len(results)} matching books")
            if len(results) > 0:
                print(f"Sample results: {results[0]['title']} ({results[0].get('year', 'Unknown')}), {results[-1]['title'] if len(results) > 1 else ''} ({results[-1].get('year', 'Unknown') if len(results) > 1 else ''})")
        else:
            print("No matching books found")
        
        # Limit to top_k
        return results[:top_k]

    def search(self, query_string, model="bm25", top_k=20):
        # Special case for "90s" - always use direct decade filtering
        lower_query = query_string.lower().strip()
        if lower_query == "90s":
            print(f"Special handling for '90s' decade search")
            return self._direct_filter_search(None, None, {"decade": (1990, 1999)}, top_k)
        
        # Special case for other single-word decade queries
        for decade_pattern, decade_range in self.decade_patterns.items():
            cleaned_pattern = decade_pattern.replace(r'\b', '')
            if lower_query == cleaned_pattern:
                print(f"Single decade query detected: '{lower_query}' → {decade_range}")
                return self._direct_filter_search(None, None, {"decade": decade_range}, top_k)
        
        # Special case: check if the query consists of just a color name with no other terms
        if lower_query in self.color_keywords:
            print(f"Single color query detected: '{lower_query}'")
            return self._direct_filter_search(None, [lower_query], None, top_k)
        
        # Special case: check for just a year (like "2000" or "1995")
        if re.fullmatch(r'(18|19|20)\d{2}', lower_query):
            exact_year = int(lower_query)
            print(f"Single year query detected: {exact_year}")
            return self._direct_filter_search(None, None, {"exact_year": exact_year}, top_k)
        
        # 1. Parse filters - extract colors, year, and genre
        colors, year_info, parsed_genre = self.parse_filters(query_string)
        
        # Always use direct filtering for decade searches
        if year_info and "decade" in year_info:
            print("Decade search detected - using direct filtering")
            return self._direct_filter_search(parsed_genre, colors, year_info, top_k)
        
        # 2. Check if this is a filters-only query (just colors, genre, year with no other search terms)
        # For any query with at least one filter, let's use direct filtering if there are no meaningful terms left
        if colors or parsed_genre or year_info:
            if self._is_filters_only_query(query_string, parsed_genre, colors, year_info):
                return self._direct_filter_search(parsed_genre, colors, year_info, top_k)
        
        # 3. For queries with search terms plus filters, use the search model
        # First remove filter terms from the query to get the actual search terms
        cleaned_query = query_string
        
        # Remove genre terms
        if parsed_genre:
            for syn in self.genre_dict.get(parsed_genre, []):
                cleaned_query = re.sub(r'\b' + re.escape(syn) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        
        # Remove color terms
        for color in colors:
            cleaned_query = re.sub(r'\b' + re.escape(color) + r'\b', '', cleaned_query, flags=re.IGNORECASE)
        
        # Remove year and decade references
        if year_info:
            # Remove exact years
            if "exact_year" in year_info:
                year_pattern = r'\b(18|19|20)\d{2}\b'
                cleaned_query = re.sub(year_pattern, '', cleaned_query)
            
            # Remove decade references
            for pattern in self.decade_patterns.keys():
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
        
        # Clean up the query
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        print(f"Cleaned Query for Model: '{cleaned_query}'")
        
        # Process the cleaned query
        query_terms = process_text(cleaned_query)
        print("Processed Query Terms:", query_terms)
        
        # If after removing filters there are no search terms left, use direct filtering
        if not query_terms:
            print("No search terms left after removing filters. Using direct filter search.")
            return self._direct_filter_search(parsed_genre, colors, year_info, top_k)

        # Search using selected model
        try:
            if model == "bm25":
                results = self.bm25.search(query_terms, top_k=top_k*3, use_query_expansion=False)
            elif model == "vsm":
                results = self.vsm.search(query_terms, top_k=top_k*3)
            elif model == "lm":
                results = self.lm.search(query_terms, top_k=top_k*3)
            else:
                print(f"Unsupported model: {model}, falling back to BM25")
                results = self.bm25.search(query_terms, top_k=top_k*3, use_query_expansion=False)
        except Exception as e:
            print(f"Error in search model {model}: {e}")
            print("Falling back to BM25 model")
            results = self.bm25.search(query_terms, top_k=top_k*3, use_query_expansion=False)

        # Apply filters to the search results
        ordered_results = []
        for doc_id, _ in results:
            str_doc_id = str(doc_id)
            matching_rows = self.book_data[self.book_data["doc_id"].astype(str) == str_doc_id]
            if matching_rows.empty:
                continue

            book_info = matching_rows.iloc[0].to_dict()

            # Apply color filter
            if colors:
                rgb_color = book_info.get("dominant_color", "")
                color_label = self.rgb_to_basic_color(rgb_color)
                
                if len(results) < 30:  # Debug output for small result sets
                    print(f"Color detection for '{book_info.get('title')}': {rgb_color} → {color_label}")
                
                if not any(color_label == user_color for user_color in colors):
                    continue

            # Apply year filter
            if year_info:
                try:
                    book_year = int(book_info.get("year", 0))
                    
                    if "exact_year" in year_info:
                        # For exact year matching, only include books from that exact year
                        exact_year = year_info["exact_year"]
                        if book_year != exact_year:
                            continue
                            
                    elif "decade" in year_info:
                        # For decade matching, include books within the decade range
                        start_year, end_year = year_info["decade"]
                        if not (start_year <= book_year <= end_year):
                            continue
                except:
                    continue

            # Apply genre filter
            if parsed_genre:
                normalized_genre = self._normalize_genre(parsed_genre)
                subject_normalized = book_info.get("normalized_subject", "")
                
                if normalized_genre != subject_normalized:
                    continue
                
                print(f"✅ GENRE MATCHED BOOK: {book_info['title']} [{book_info.get('subject')}]")

            # This book passed all filters
            print(f"FINAL MATCH: {book_info['title']} | {book_info.get('year')} | {book_info.get('subject')}")
            ordered_results.append(book_info)

        # If no results were found with both search and filters, try direct filtering instead
        if not ordered_results and (colors or parsed_genre or year_info):
            print("No results found with search model and filters. Trying direct filter search.")
            return self._direct_filter_search(parsed_genre, colors, year_info, top_k)
            
        # Return results
        return ordered_results[:top_k]