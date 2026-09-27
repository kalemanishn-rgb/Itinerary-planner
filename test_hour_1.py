# test_hour_1.py
from embeddings import fetch_top_matches
from approach_1 import run_vanilla_rag

print("==================================================")
print("🚀 TESTING LIVE OPENAPI KEY INTEGRATION")
print("==================================================")

try:
    print("Running Test 1: Fetching Vector Search Database...")
    test_query = "paintings and modern galleries"
    matches = fetch_top_matches(test_query, top_k=2)
    for venue in matches:
        print(f"✅ Found: {venue['name']} ({venue['type']})")
        
    print("\nRunning Test 2: Generating Vanilla RAG Itinerary via gpt-5.4-mini...")
    rainy_preference = "Show me famous outdoor parks and landmarks"
    weather_condition = "Heavy Torrential Rain Storm"
    
    rag_output = run_vanilla_rag(rainy_preference, weather_condition)
    print("\nSYSTEM LLM OUTPUT:")
    print(rag_output)

except Exception as e:
    print(f"\n❌ Error encountered during execution: {e}")
