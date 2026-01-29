import json
import numpy as np
from src.rag.parse_politifact import load_politifact_corpus
from src.rag.chunking import chunk_corpus
from src.rag.embed import embed_texts

OUTPUT_DIR = "data/rag_index"
EMBED_FILE = f"{OUTPUT_DIR}/embeddings.npy"
META_FILE = f"{OUTPUT_DIR}/chunks.json"

def build_rag_index():
    print("Loading Politifact corpus...")
    corpus = load_politifact_corpus()

    print("Chunking corpus...")
    chunks = chunk_corpus(corpus)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts([c["text"] for c in chunks])

    print("Saving index...")
    np.save(EMBED_FILE, embeddings)

    with open(META_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

    print("RAG index built successfully.")

if __name__ == "__main__":
    build_rag_index()