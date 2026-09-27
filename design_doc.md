# Capstone Design Document: Real-Time Dynamic City Itinerary Planner
**AI Engineering Cohort Capstone Proposal**

## 1. Problem Statement
Manual synthesis of fragmented travel data (transit, operating hours, live weather) results in broken, static day schedules. Existing tools cannot adapt when dynamic parameters shift. This system solves the specific, narrow challenge of automatically generating and self-correcting a single-day city itinerary optimized by live constraint rules, vector-based user preferences, and geographical sequencing.

## 2. High-Level Architecture
The system consists of a two-tier knowledge surface: a local vector store built on `text-embedding-3-small` and an orchestration network. We evaluated two distinct implementation core flows:
*   **Approach 1 (Base Baseline):** A linear Vanilla RAG pipeline. It vector-searches attractions based on user keywords and feeds the context directly to `gpt-5.4-mini` without validation loops.
*   **Approach 2 (Advanced Graph):** A Multi-Agent cyclic orchestration network managed via LangGraph consisting of three nodes:
    1.  `RouterAgent`: Deconstructs text queries into semantic database lookup matrices.
    2.  `ResearchAgent`: Audits weather safety rules against attraction attributes (`indoor`/`outdoor`).
    3.  `RouteAgent`: Generates the finalized chronological sequencing via `gpt-5.4-mini`.

## 3. Evaluation Criteria
We executed a quantitative comparative evaluation across standard test suites checking two primary success metrics:
*   **Quantitative Constraint Adherence Rate:** The percentage of scheduled time slots that safely respect weather alert states.
*   **Qualitative Feasibility Score:** An LLM-as-a-judge score (scale 1-5) evaluating chronological layout logic.

## 4. Framework Justification
LangGraph was selected over sequential pipelines because dynamic itinerary mapping requires cyclical state checking. When the `ResearchAgent` uncovers a constraint violation, it must execute a graph loop back to rewrite the state array with an alternative asset. This state preservation pattern makes LangGraph ideal for complex agent networks.
