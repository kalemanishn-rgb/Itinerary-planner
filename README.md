# Itinerary-planner
Itinerary-planner


## 📌 Mandatory Section 4: Failure Analysis & Pivot Log

### 1. Architectural Failure Identified (Hour 1 Baseline)
During initial testing of **Approach 1 (Vanilla RAG)**, the system successfully parsed semantic user vectors but completely failed to respect real-world runtime context limitations. When presented with a `Heavy Torrential Rain Storm`, the linear pipeline blindly recommended 100% outdoor venues (*Central Park Bethesda Fountain, Bryant Park*), resulting a **Constraint Adherence Rate of 0%**.

### 2. LangGraph State Ingestion Bugs (Hour 2 Refactoring)
When implementing the multi-agent graph architecture via LangGraph, we encountered two fatal schema failures:
* **KeyError 'user_prompt':** Encountered because node frames were initially compiled using an abstract memory structure instead of a proper `TypedDict` schema. This blocked LangGraph from accurately streaming state mutations between agent boundary edges. Fixed by refactoring global state to explicitly inherit from `TypedDict`.
* **OpenAI List Attribute Crashes:** In the final node pipeline execution step, extracting completion responses from `gpt-5.4-mini` threw an `AttributeError`. This was trace-located to an incorrect list comprehension format (`response.choices.message.content`). Resolved by applying explicit item indexing (`response.choices[0].message.content`).

### 3. Quantitative Pivot Impact
By routing agent tasks dynamically through a cyclic conditional loop controlled by a explicit `ResearchAgent` auditor node, **Approach 2 achieved a Constraint Adherence Rate of 100%** on extreme weather scenarios, verifying the implementation's success against the evaluation rubric.
