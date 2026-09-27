# embeddings.py
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from database import ATTRACTIONS_DB

# Force reloading environment variables
load_dotenv(override=True)

# Initialize the authentic OpenAI client engine
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(text: str) -> list:
    """Generates vectors using text-embedding-3-small model."""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(a, b) -> float:
    """Calculates directional similarity between vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def fetch_top_matches(query: str, top_k: int = 3) -> list:
    """Vector search loop matching user text to our collection."""
    query_vector = get_embedding(query)
    scored_attractions = []
    
    for attraction in ATTRACTIONS_DB:
        payload = f"{attraction['name']} - {attraction['description']}"
        item_vector = get_embedding(payload)
        
        score = cosine_similarity(query_vector, item_vector)
        scored_attractions.append((score, attraction))
        
    scored_attractions.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_attractions[:top_k]]
