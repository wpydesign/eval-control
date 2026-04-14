"""
LLM Consistency Graph Engine (LCGE) v1.0

A measurement system for detecting contradictions in LLM outputs
across prompt variants. Not a chatbot. Not an agent. A controlled
experimental system.

Architecture:
    PromptInputLayer  -> generates 10 prompt variants per seed
    LLMExecutionLayer -> calls primary + baseline models
    NormalizationLayer -> extracts answers, embeddings, refusal tags
    EdgeBuilder       -> creates semantic/contradiction/variance edges
    GraphConstructor  -> assembles queryable graph
    ContradictionDetector -> finds contradiction clusters
    ScoringEngine     -> computes confidence (cap 10)
    OutputPipeline    -> produces minimal reproducible findings
"""

__version__ = "1.0.0"
__engine_name__ = "LLM Consistency Graph Engine"
