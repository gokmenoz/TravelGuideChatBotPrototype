import os

import faiss

from constants import QUESTIONS
from utils import (
    build_faiss_index,
    chunk_text,
    embedder,
    extract_location,
    get_wikivoyage_page,
    load_index,
    save_index,
)

INDEX_DIR = "faiss_index"

# Load or initialize index and chunks
if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    index, all_chunks = load_index(INDEX_DIR)
else:
    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    all_chunks = []

# Track already indexed locations
indexed_locations = set([c["location"].lower() for c in all_chunks if "location" in c])

# Process new locations
locations = [extract_location(q) for q in QUESTIONS if extract_location(q)]
for location in locations:
    if location.lower() in indexed_locations:
        print(f"✅ Skipping already indexed location: {location}")
        continue

    print(f"🔍 Processing: {location}")
    content = get_wikivoyage_page(location)
    chunks = chunk_text(content, location)

    # Encode and add to existing index
    new_index, new_chunks = build_faiss_index(chunks, embedder)
    index.add(new_index.reconstruct_n(0, new_index.ntotal))
    all_chunks.extend(new_chunks)

# Save updated index and chunks
save_index(index, all_chunks, outdir=INDEX_DIR)
print("✅ Indexing complete.")
