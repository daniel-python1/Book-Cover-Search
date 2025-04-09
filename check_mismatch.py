import pandas as pd

# Load your metadata
df = pd.read_csv('C:/Users/comer/OneDrive - Dublin City University/Desktop/Web Scrape/enhanced_book_metadata.csv')

# Check for the specific ID
print(f"Document ID '3953' exists in metadata: {any(df['doc_id'].astype(str) == '3953')}")

# Check for books with 'mother' in the title
mother_books = df[df['title'].str.contains('mother', case=False, na=False)]
print(f"Found {len(mother_books)} books with 'mother' in the title")
if len(mother_books) > 0:
    print("Sample of books with 'mother' in the title:")
    print(mother_books[['doc_id', 'title']].head())