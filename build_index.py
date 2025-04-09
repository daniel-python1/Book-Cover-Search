import os
from engine.utils import load_json, save_json
from engine.bm25 import BM25
from engine.vsm import VectorSpaceModel
from engine.language_model import DirichletLanguageModel

def build_indexes(documents):
    inverted_index = {}
    doc_lengths = {}
    term_frequencies = {}

    total_docs = len(documents)

    for doc_id, tokens in documents.items():
        term_freqs = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        doc_lengths[doc_id] = len(tokens)

        for term, freq in term_freqs.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = freq

        term_frequencies[doc_id] = term_freqs

    return inverted_index, doc_lengths, term_frequencies, total_docs

def main():
    index_dir = "data/index"  # Change this if you're saving somewhere else
    os.makedirs(index_dir, exist_ok=True)

    # Load the documents
    documents = load_json("data/index/documents.json")  # Should already be tokenized
    print(f"📄 Loaded {len(documents)} documents.")

    # Build index structures
    inverted_index, doc_lengths, term_frequencies, total_docs = build_indexes(documents)
    print("🔧 Indexes built.")

    # Save each component
    save_json(inverted_index, os.path.join(index_dir, "inverted_index.json"))
    save_json(doc_lengths, os.path.join(index_dir, "doc_lengths.json"))
    save_json(term_frequencies, os.path.join(index_dir, "term_frequencies.json"))
    save_json({"total_docs": total_docs}, os.path.join(index_dir, "metadata.json"))

    print(f"✅ Indexes saved to '{index_dir}'")

if __name__ == "__main__":
    main()
