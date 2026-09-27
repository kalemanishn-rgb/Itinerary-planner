# approach_1.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from embeddings import fetch_top_matches

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def run_vanilla_rag(user_prompt: str, weather_status: str) -> str:
    """
    Approach 1: This fetches attractions purely on keywords, merges static weather state,
    and directly prompts the LLM to format a sequential schedule structure.
    """
    matched_venues = fetch_top_matches(user_prompt, top_k=3)
    
    context_str = ""
    for v in matched_venues:
        context_str += f"- {v['name']} ({v['type']}). Hours: {v['hours']}. Info: {v['description']}\n"
        
    system_prompt = (
        "You are a basic single-day travel itinerary assistant. "
        "Use ONLY the provided context venues and the weather condition to build a day layout.\n\n"
        f"CURRENT WEATHER CONDITION: {weather_status}\n\n"
        f"AVAILABLE CONTEXT ATTRACTIONS:\n{context_str}\n"
        "Output an itinerary scheduling these places. Do not run any advanced loop modifications."
    )
    
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Build an itinerary based on: {user_prompt}"}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content
