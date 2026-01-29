import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query_embedding, corpus_embeddings, corpus_chunks, k=5):
    similarities = cosine_similarity(
        [query_embedding], corpus_embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]

    return [corpus_chunks[i] for i in top_indices]