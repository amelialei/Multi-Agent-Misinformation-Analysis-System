import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts):
    response = client.models.embed_content(
        model="text-embedding-004",
        content=texts
    )
    return [e.values for e in response.embeddings]