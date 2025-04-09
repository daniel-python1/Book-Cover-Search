import requests
import os
import csv
import time

# Subjects to scrape
SUBJECTS = [
    "fantasy", "science_fiction", "romance", "mystery", "horror",
    "historical_fiction", "young_adult", "adventure", "thriller",
    "poetry", "biography", "autobiography", "children", "drama",
    "classic_literature", "sports", "technology", "philosophy",
    "psychology", "religion", "education"
]

# Configuration
MAX_BOOKS_PER_SUBJECT = 200
OUTPUT_FOLDER = "book_covers"
CSV_FILE = "book_metadata.csv"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_books_from_subject(subject, limit=MAX_BOOKS_PER_SUBJECT):
    url = f"https://openlibrary.org/subjects/{subject}.json?limit={limit}"
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"Failed to fetch subject: {subject}")
            return []
        data = r.json()
        return data.get("works", [])
    except Exception as e:
        print(f"Error fetching books for {subject}: {e}")
        return []

def download_cover(cover_id, filename):
    url = f"http://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    try:
        img_data = requests.get(url, headers=HEADERS).content
        with open(filename, 'wb') as f:
            f.write(img_data)
        return url
    except Exception as e:
        print(f"Couldn't download image {url}: {e}")
        return None

def main():
    metadata = []
    count = 0

    for subject in SUBJECTS:
        print(f"Fetching subject: {subject}")
        books = get_books_from_subject(subject)

        for book in books:
            title = book.get("title", "N/A")
            authors = ", ".join([a['name'] for a in book.get("authors", [])]) if "authors" in book else "N/A"
            year = book.get("first_publish_year", "N/A")
            cover_id = book.get("cover_id")

            if not cover_id:
                continue

            filename = os.path.join(OUTPUT_FOLDER, f"{count}_{title.replace(' ', '_')}.jpg")
            cover_url = download_cover(cover_id, filename)

            if cover_url:
                metadata.append([title, authors, year, subject, cover_url, filename])
                count += 1
                time.sleep(0.1)

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["title", "authors", "year", "subject", "cover_url", "filename"])
        writer.writerows(metadata)

    print(f"Done! {count} images downloaded and saved.")

if __name__ == "__main__":
    main()
