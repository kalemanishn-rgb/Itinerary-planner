# evals.py
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from approach_1 import run_vanilla_rag
from approach_2 import run_advanced_agent_planner

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Test suite containing distinct architectural scenarios
TEST_SUITE = [
    {
        "scenario": "Severe Rainy Day Storm",
        "preference": "Show me iconic parks, fountains, and beautiful viewpoints walk.",
        "weather": "Heavy Torrential Rain Storming"
    },
    {
        "scenario": "Blistering Heatwave Alert",
        "preference": "I want a long outdoor hiking and relaxation promenade walk.",
        "weather": "Extreme Heat Wave Warning (42°C Outside)"
    }
]

def llm_judge_score(itinerary_text: str, weather: str) -> dict:
    """Uses gpt-5.4-mini as an LLM-as-a-judge to grade the pipeline output."""
    judge_prompt = (
        "You are an objective Capstone Evaluation Grading Auditor.\n"
        "Evaluate the following travel itinerary against the provided weather condition.\n\n"
        f"WEATHER CONDITION TO ENFORCE: {weather}\n"
        f"GENERATED ITINERARY TEXT:\n\"\"\"\n{itinerary_text}\n\"\"\"\n\n"
        "Provide your scoring strictly in the following bracketed format:\n"
        "ADHERENCE_RATE: [percentage between 0 and 100 representing how many slots are safe for the weather]\n"
        "FEASIBILITY_SCORE: [integer between 1 and 5 scoring chronological flow logic]"
    )
    
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "system", "content": judge_prompt}],
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    
    # Parse numerical metrics out of the judge's block
    adherence = 0
    feasibility = 0
    try:
        adherence_match = re.search(r"ADHERENCE_RATE:\s*\[?(\d+)\]?", content)
        feasibility_match = re.search(r"FEASIBILITY_SCORE:\s*\[?(\d+)\]?", content)
        if adherence_match: adherence = int(adherence_match.group(1))
        if feasibility_match: feasibility = int(feasibility_match.group(1))
    except Exception:
        pass
        
    return {"adherence": adherence, "feasibility": feasibility}

def execute_comparative_evaluation():
    print("==================================================================")
    print("🚀 INITIALIZING AUTOMATED CAPSTONE EVALUATION RUNNER")
    print("==================================================================")
    
    a1_adherence_total, a1_feasibility_total = 0, 0
    a2_adherence_total, a2_feasibility_total = 0, 0
    
    for idx, case in enumerate(TEST_SUITE):
        print(f"\n🎬 Running Scenario {idx+1}: {case['scenario']}")
        print(f"   User Preference: {case['preference']}")
        print(f"   Weather Context: {case['weather']}")
        
        # 1. Evaluate Approach 1 (Vanilla RAG)
        print("   -> Executing Approach 1 (Vanilla RAG)...")
        a1_output = run_vanilla_rag(case["preference"], case["weather"])
        a1_scores = llm_judge_score(a1_output, case["weather"])
        
        # 2. Evaluate Approach 2 (LangGraph Advanced Agent)
        print("   -> Executing Approach 2 (LangGraph Multi-Agent Loop)...")
        a2_output = run_advanced_agent_planner(case["preference"], case["weather"])
        a2_scores = llm_judge_score(a2_output, case["weather"])
        
        # Aggregate totals
        a1_adherence_total += a1_scores["adherence"]
        a1_feasibility_total += a1_scores["feasibility"]
        a2_adherence_total += a2_scores["adherence"]
        a2_feasibility_total += a2_scores["feasibility"]
        
        print(f"   📊 [Approach 1 Scores] Adherence: {a1_scores['adherence']}% | Feasibility: {a1_scores['feasibility']}/5")
        print(f"   📊 [Approach 2 Scores] Adherence: {a2_scores['adherence']}% | Feasibility: {a2_scores['feasibility']}/5")

    # Calculate final average scores
    num_cases = len(TEST_SUITE)
    print("\n==================================================================")
    print("🏆 FINAL COMPARATIVE PERFORMANCE REPORT (SUBMISSION SUMMARY)")
    print("==================================================================")
    print(f"Approach 1 (Vanilla RAG Baseline):")
    print(f"  - Avg Constraint Adherence Rate : {a1_adherence_total / num_cases:.1f}%")
    print(f"  - Avg Qualitative Feasibility  : {a1_feasibility_total / num_cases:.1f}/5.0")
    print(f"\nApproach 2 (LangGraph Multi-Agent Orchestration):")
    print(f"  - Avg Constraint Adherence Rate : {a2_adherence_total / num_cases:.1f}%")
    print(f"  - Avg Qualitative Feasibility  : {a2_feasibility_total / num_cases:.1f}/5.0")
    print("==================================================================")

if __name__ == "__main__":
    execute_comparative_evaluation()
