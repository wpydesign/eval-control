"""
engine.py — LCGE Main Engine Orchestrator

The central pipeline that coordinates all layers:
    1. Prompt Input Layer     -> generate variants
    2. LLM Execution Layer   -> call models
    3. Normalization Layer   -> extract answers, embeddings
    4. Edge Builder          -> create edges
    5. Graph Construction    -> assemble graph
    6. Contradiction Detection -> find clusters
    7. Scoring               -> compute confidence
    8. Output Pipeline       -> produce report

This is a MEASUREMENT SYSTEM, not an agent.
No conversation history. No adaptive behavior. No heuristics.
"""

import json
import logging
from typing import Optional

from .prompt_input_layer import generate_variants, PromptVariant
from .llm_execution_layer import execute_all_variants, LLMResponse
from .normalization_layer import normalize_and_embed, EmbeddingEngine, NormalizedResponse
from .edge_builder import EdgeBuilder
from .graph_constructor import build_graph, ConsistencyGraph
from .contradiction_detector import ContradictionDetector, ContradictionCluster
from .scoring_engine import ScoringEngine, ScoredCluster, compute_reproducibility
from .output_pipeline import generate_report, OutputReport

logger = logging.getLogger("lcge")


class LCGEEngine:
    """
    LLM Consistency Graph Engine v1.0

    A controlled experimental system for detecting contradictions
    in LLM outputs across prompt variants.

    Usage:
        engine = LCGEEngine()
        report = engine.run(
            task="Is it ethical to lie to protect someone?",
            seed_prompt="Is it ethical to lie to protect someone from harm?",
        )
        print(report.to_json())
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._setup_logging()
        self.edge_builder = EdgeBuilder()
        self.detector = ContradictionDetector()
        self.scorer = ScoringEngine()

    def _setup_logging(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def run(
        self,
        task: str,
        seed_prompt: str,
        num_variants: int = 10,
        models: Optional[list] = None,
        reproducibility_runs: int = 1,
    ) -> OutputReport:
        """
        Execute the full LCGE pipeline.

        Args:
            task: The underlying task/intent being tested.
            seed_prompt: The original prompt text.
            num_variants: Number of prompt variants to generate (default: 10).
            models: List of model roles to test (default: ["primary"]).
            reproducibility_runs: Number of times to repeat for reproducibility (default: 1).

        Returns:
            OutputReport with full findings.
        """
        logger.info("=" * 60)
        logger.info("LLM Consistency Graph Engine v1.0 — Pipeline Start")
        logger.info(f"Task: {task}")
        logger.info(f"Seed: {seed_prompt[:80]}...")
        logger.info("=" * 60)

        # ---- Step 1: Prompt Input Layer ----
        logger.info("[Step 1] Generating prompt variants...")
        variants = generate_variants(task, seed_prompt, num_variants)
        logger.info(f"  Generated {len(variants)} variants")
        for v in variants:
            logger.debug(f"    {v.node_id[:8]} | {v.strategy:20s} | {v.prompt[:60]}...")

        # Compute prompt embeddings for similarity
        from .normalization_layer import EmbeddingEngine as PromptEmbeddingEngine
        prompt_engine = PromptEmbeddingEngine()
        prompt_texts = [v.prompt for v in variants]
        prompt_engine.embed(prompt_texts)
        self.edge_builder.set_prompt_engine(prompt_engine)

        # ---- Step 2: LLM Execution Layer ----
        logger.info("[Step 2] Executing LLM calls...")
        if models is None:
            models = ["primary"]
        llm_responses = execute_all_variants(variants, models)
        logger.info(f"  Received {len(llm_responses)} responses")
        for r in llm_responses:
            status = "REFUSED" if any(
                p in r.content.lower() for p in ["i cannot", "i can't", "as an ai"]
            ) else "OK"
            logger.debug(f"    {r.node_id[:8]} | {r.model_role:10s} | {status:8s} | {r.token_count} tokens")

        # ---- Step 3: Normalization Layer ----
        logger.info("[Step 3] Normalizing responses...")
        normalized, response_engine = normalize_and_embed(llm_responses)
        self.edge_builder.set_response_engine(response_engine)
        refusals = sum(1 for n in normalized if n.refusal_flag)
        logger.info(f"  Normalized {len(normalized)} responses ({refusals} refusals detected)")

        # ---- Step 4: Edge Builder ----
        logger.info("[Step 4] Building edges...")
        edges = self.edge_builder.build_all_edges(variants, normalized)
        security_edges = [e for e in edges if e.weight > 0.0]
        logger.info(f"  Built {len(edges)} edges ({len(security_edges)} security edges)")
        for e in edges:
            if e.weight > 0:
                logger.debug(f"    {e.source_id[:8]} -> {e.target_id[:8]} | {e.edge_type:20s} | w={e.weight:.1f}")

        # ---- Step 5: Graph Construction ----
        logger.info("[Step 5] Constructing graph...")
        graph = build_graph(variants, normalized, edges)
        logger.info(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
        logger.info(f"  Connected components: {len(graph.get_connected_components())}")

        # ---- Step 6: Contradiction Detection ----
        logger.info("[Step 6] Detecting contradictions...")
        clusters = self.detector.detect(graph)
        logger.info(f"  Found {len(clusters)} contradiction clusters")
        for c in clusters:
            logger.debug(f"    {c.cluster_id} | {c.cluster_type:25s} | nodes={len(c.nodes_involved)} | raw_conf={c.confidence:.1f}")

        # ---- Step 7: Scoring ----
        logger.info("[Step 7] Scoring clusters...")
        scored_clusters = self.scorer.score_all(clusters, graph)
        submittable = [s for s in scored_clusters if s.is_submittable]
        logger.info(f"  Scored {len(scored_clusters)} clusters ({len(submittable)} submittable)")
        for s in scored_clusters:
            flag = "SUBMIT" if s.is_submittable else "REJECT"
            logger.info(f"    {s.cluster.cluster_id} | {flag:6s} | conf={s.confidence:.1f} | div={s.diversity}")
            if s.rejection_reason:
                logger.debug(f"           Reason: {s.rejection_reason}")

        # ---- Reproducibility ----
        reproducibility = None
        if reproducibility_runs > 1:
            logger.info(f"[Reproducibility] Running {reproducibility_runs - 1} additional runs...")
            all_run_results = [scored_clusters]
            for run_idx in range(1, reproducibility_runs):
                logger.info(f"  Run {run_idx + 1}/{reproducibility_runs}...")
                run_clusters = self.detector.detect(graph)
                run_scored = self.scorer.score_all(run_clusters, graph)
                all_run_results.append(run_scored)
            reproducibility = compute_reproducibility(all_run_results)
            logger.info(f"  Overall reproducibility: {reproducibility['overall_reproducibility']}")

        # ---- Step 8: Output Pipeline ----
        logger.info("[Step 8] Generating report...")
        report = generate_report(
            task=task,
            seed_prompt=seed_prompt,
            graph=graph,
            scored_clusters=scored_clusters,
            reproducibility=reproducibility,
            raw_clusters=clusters,
        )

        logger.info("=" * 60)
        logger.info(f"Pipeline complete. Contradictions: {len(scored_clusters)}, "
                     f"Submittable: {len(submittable)}")
        logger.info("=" * 60)

        return report
