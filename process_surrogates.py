import pandas as pd
import os
import cv2
import numpy as np
from engine.utils import process_text  # Assuming this is from your search engine

# Load your existing metadata
metadata_path = "C:/Users/comer/OneDrive - Dublin City University/Desktop/Web Scrape/data/index/book_metadata.csv"
image_folder = "C:/Users/comer/OneDrive - Dublin City University/Desktop/Web Scrape/book_covers"
df = pd.read_csv(metadata_path)

# Add dominant color to each image
def get_dominant_color(filename):
    try:
        # Use os.path.basename to extract just the filename
        base_filename = os.path.basename(filename)
        
        # Create the full path using normpath to handle any slash differences
        full_path = os.path.normpath(os.path.join(image_folder, base_filename))
        
        print(f"Attempting to process: {full_path}")
        
        if not os.path.exists(full_path):
            # Try with URL encoding the filename
            from urllib.parse import unquote
            decoded_filename = unquote(base_filename)
            full_path = os.path.normpath(os.path.join(image_folder, decoded_filename))
            
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
        color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
        return color_str
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        return "error"

# Add document ID and dominant color
df['doc_id'] = df.index.astype(str)
df['dominant_color'] = df['filename'].apply(get_dominant_color)

# Create a text surrogate by combining all metadata
df['surrogate_text'] = df.apply(
    lambda row: f"{row['title']} {row['authors']} {row['subject']} {row['year']} {row['dominant_color']}",
    axis=1
)

# Save enhanced metadata
df.to_csv("enhanced_book_metadata.csv", index=False)

# Create documents for indexing
documents = {}
for _, row in df.iterrows():
    doc_id = row['doc_id']
    text = row['surrogate_text']
    # Process the text using your engine's text processing function
    documents[doc_id] = process_text(text)

# Save documents for indexing
import json
with open('documents.json', 'w') as f:
    json.dump(documents, f)

print(f"Enhanced metadata for {len(df)} books and prepared documents for indexing")