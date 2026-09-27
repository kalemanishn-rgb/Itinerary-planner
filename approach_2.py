# approach_2.py
import os
from typing import Dict, Any, List, TypedDict
from openai import OpenAI
from dotenv import load_dotenv
from database import ATTRACTIONS_DB
from embeddings import fetch_top_matches
from langgraph.graph import StateGraph, END

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# 1. Define the Global State Schema using TypedDict
class ItineraryState(TypedDict):
    user_prompt: str
    weather: str
    current_venues: List[Dict[str, Any]]
    itinerary_layout: str
    iterations: int
    conflict_found: bool

# 2. Agent 1 Node: RouterAgent
def router_agent_node(state: ItineraryState) -> Dict[str, Any]:
    print("🤖 [RouterAgent] Initializing vector lookups for target criteria...")
    top_matches = fetch_top_matches(state["user_prompt"], top_k=4)
    
    return {
        "current_venues": top_matches, 
        "iterations": state.get("iterations", 0) + 1,
        "conflict_found": False
    }

# 3. Agent 2 Node: ResearchAgent (Validates weather compliance rules)
def research_agent_node(state: ItineraryState) -> Dict[str, Any]:
    print("🔬 [ResearchAgent] Auditing operational schedules and weather compliance...")
    weather = state["weather"].lower()
    venues = state["current_venues"]
    conflict = False
    updated_venues = list(venues)
    
    # Structural rule evaluation: Is an outdoor venue scheduled during a heavy storm?
    if "rain" in weather or "storm" in weather:
        for idx, venue in enumerate(venues):
            if venue["type"] == "outdoor":
                print(f"⚠️ Conflict found: Cannot visit outdoor '{venue['name']}' in rainy weather!")
                conflict = True
                
                # Filter down available alternative indoor options that aren't already included
                all_indoors = [v for v in ATTRACTIONS_DB if v["type"] == "indoor" and v not in updated_venues]
                if all_indoors:
                    replacement_venue = all_indoors[0]  # Fixed: Select the first dictionary item cleanly
                    print(f"🔄 Swapping outdoor layout for indoor option: {replacement_venue['name']}")
                    updated_venues[idx] = replacement_venue
                    break  # Break loop to evaluate the updated layout sequentially
                    
    return {"current_venues": updated_venues, "conflict_found": conflict}

# 4. Agent 3 Node: RouteAgent (Sequences final itinerary structure via gpt-5.4-mini)
def route_agent_node(state: ItineraryState) -> Dict[str, Any]:
    print("🗺️ [RouteAgent] Sequencing time slots and drafting final layout...")
    context_str = ""
    for idx, v in enumerate(state["current_venues"]):
        context_str += f"- Slot {idx+1}: {v['name']} ({v['type']}). Hours: {v['hours']}. Info: {v['description']}\n"
        
    system_prompt = (
        "You are an expert travel coordinator. Build a strict chronological itinerary.\n"
        f"Weather Profile: {state['weather']}\n"
        f"Final Confirmed Venues:\n{context_str}\n"
        "Generate a structured daily layout confirming why these selections are safe for the current weather."
    )
    
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Provide the final optimized sequencing layout."}
        ],
        temperature=0.1
    )
    return {"itinerary_layout": response.choices[0].message.content}
    #return {"itinerary_layout": response.choices.message.content}

# 5. Cyclic Logic Controller Node
def routing_condition_check(state: ItineraryState) -> str:
    if state["conflict_found"] and state["iterations"] < 3:
        print("🔁 Loop Rule Triggered: Re-routing execution path back to Research node.")
        return "research"
    print("✅ Architecture Verified: State constraints verified clean. Transitioning to output.")
    return "route"

# 6. Assemble the Orchestration StateGraph Pipeline
workflow = StateGraph(ItineraryState)  # Fixed: Use TypedDict type hint mapping

# Add functional nodes
workflow.add_node("router", router_agent_node)
workflow.add_node("research", research_agent_node)
workflow.add_node("route", route_agent_node)

# Map edge relationships
workflow.set_entry_point("router")
workflow.add_edge("router", "research")

# Inject dynamic cyclic route condition logic
workflow.add_conditional_edges(
    "research",
    routing_condition_check,
    {
        "research": "research",
        "route": "route"
    }
)
workflow.add_edge("route", END)

# Compile functional system graph blueprint
compiled_graph = workflow.compile()

def run_advanced_agent_planner(prompt: str, weather: str) -> str:
    """Executes state pipeline mapping architecture outputs smoothly."""
    initial_state = {
        "user_prompt": prompt,
        "weather": weather,
        "current_venues": [],
        "itinerary_layout": "",
        "iterations": 0,
        "conflict_found": False
    }
    final_output = compiled_graph.invoke(initial_state)
    return final_output["itinerary_layout"]

if __name__ == "__main__":
    sample_preference = "Show me famous outdoor parks and landmarks"
    sample_weather = "Heavy Torrential Rain Storm"
    print("\n--- Executing Hour 2: LangGraph Orchestration Pipeline ---")
    print(run_advanced_agent_planner(sample_preference, sample_weather))
