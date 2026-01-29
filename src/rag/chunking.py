def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 30:
            chunks.append(chunk)

    return chunks


def chunk_corpus(corpus):
    chunked_docs = []

    for doc in corpus:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            chunked_docs.append({
                "text": chunk,
                "source": doc["source"],
                "label": doc["label"],
                "url": doc["url"]
            })

    return chunked_docs