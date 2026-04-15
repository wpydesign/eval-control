"""
engine.py — LCGE v1.2 Main Engine Orchestrator

Prompt Transformation → Behavioral State Mapping Engine.

Pipeline:
    1. Prompt Input Layer      -> generate variants
    2. LLM Execution Layer    -> call models
    3. Normalization Layer    -> extract answers, embeddings, reasoning, format
    4. Edge Builder           -> behavioral_shift, policy_flip, semantic_drift
    5. Graph Construction     -> assemble graph
    6. Instability Classifier -> classify instability types (with reasoning override)
    7. Scoring Engine         -> compute instability scores (peak + mean)
    8. Output Pipeline        -> produce instability report

This is a MEASUREMENT SYSTEM, not an agent.
No conversation history. No adaptive behavior. No heuristics.

v1.2: Behavioral state mapping — measures how model behavior changes
      under semantic perturbation of the prompt space.
"""

import hashlib
import logging
from typing import Optional

from .prompt_input_layer import generate_variants, PromptVariant
from .llm_execution_layer import execute_all_variants, LLMResponse
from .normalization_layer import normalize_and_embed, EmbeddingEngine, NormalizedResponse
from .edge_builder import EdgeBuilder
from .graph_constructor import build_graph, ConsistencyGraph
from .instability_classifier import InstabilityClassifier
from .scoring_engine import ScoringEngine, compute_reproducibility
from .output_pipeline import generate_report, InstabilityReport

logger = logging.getLogger("lcge")


class LCGEEngine:
    """
    LCGE v1.2 — Prompt Transformation → Behavioral State Mapping Engine.

    Classifies LLM output instability across prompt variants into:
        - policy_flip
        - reasoning_variance
        - knowledge_variance
        - formatting_variance
        - stable

    v1.2 capabilities:
        - Typed top_trigger (POLICY_SHIFT, REASONING_SHIFT, KNOWLEDGE_SHIFT, FORMAT_SHIFT)
        - Reasoning dominance override (reasoning can beat knowledge)
        - Dual global scoring (peak + mean)
        - Normalized scores for cross-task comparability

    Usage:
        engine = LCGEEngine()
        report = engine.run(
            task="Ethical reasoning about lying",
            seed_prompt="Is it ethical to lie to protect someone from harm?",
        )
        print(report.to_json())  # strict output format
        print(report.to_json(full=True))  # full diagnostic output
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._setup_logging()
        self.edge_builder = EdgeBuilder()
        self.classifier = InstabilityClassifier()
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
    ) -> InstabilityReport:
        """
        Execute the full LCGE v1.2 pipeline.

        Returns:
            InstabilityReport with strict output format.
        """
        logger.info("=" * 60)
        logger.info("LCGE v1.2 — Prompt Transformation -> Behavioral State Mapping")
        logger.info(f"Task: {task}")
        logger.info(f"Seed: {seed_prompt[:80]}...")
        logger.info("=" * 60)

        # Generate semantic family ID from task
        family_id = hashlib.sha256(task.lower().strip().encode()).hexdigest()[:8]

        # ---- Step 1: Prompt Input Layer ----
        logger.info("[Step 1] Generating prompt variants...")
        variants = generate_variants(task, seed_prompt, num_variants)
        logger.info(f"  Generated {len(variants)} variants (family: {family_id})")
        for v in variants:
            logger.debug(f"    {v.node_id[:8]} | {v.strategy:20s} | {v.prompt[:60]}...")

        # Compute prompt embeddings
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
        normalized, answer_engine, reasoning_engine = normalize_and_embed(
            llm_responses, family_id
        )
        self.edge_builder.set_response_engine(answer_engine)
        self.edge_builder.set_reasoning_engine(reasoning_engine)

        refusals = sum(1 for n in normalized if n.refusal_flag)
        formats = set(n.format_signature for n in normalized if not n.refusal_flag)
        logger.info(f"  Normalized {len(normalized)} responses "
                     f"({refusals} refusals, {len(formats)} distinct formats)")
        for n in normalized:
            logger.debug(f"    {n.node_id[:8]} | fmt={n.format_signature[:8]:8s} | "
                         f"ref={n.refusal_flag} | trace={len(n.reasoning_trace)}chars")

        # ---- Step 4: Edge Builder ----
        logger.info("[Step 4] Building edges...")
        edges = self.edge_builder.build_all_edges(variants, normalized)
        instability_edges = [e for e in edges if e.weight > 0.0]

        # Count by type
        edge_type_counts = {}
        for e in instability_edges:
            edge_type_counts[e.edge_type] = edge_type_counts.get(e.edge_type, 0) + 1

        logger.info(f"  Built {len(edges)} edges ({len(instability_edges)} instability)")
        for etype, count in edge_type_counts.items():
            logger.info(f"    {etype}: {count}")
        for e in instability_edges[:10]:  # Show first 10
            logger.debug(f"    {e.source_id[:8]} -> {e.target_id[:8]} | "
                         f"{e.edge_type:20s} | w={e.weight:.1f}")

        # ---- Step 5: Graph Construction ----
        logger.info("[Step 5] Constructing graph...")
        graph = build_graph(variants, normalized, edges, family_id)
        logger.info(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")

        # ---- Step 6: Instability Classification ----
        logger.info("[Step 6] Classifying instability...")
        clusters = self.classifier.classify(graph)
        logger.info(f"  Found {len(clusters)} instability clusters")
        for c in clusters:
            logger.info(f"    {c.cluster_id} | {c.instability_type:25s} | "
                        f"score={c.total_score:.2f} | nodes={len(c.nodes_involved)} | "
                        f"trigger={c.evidence.get('top_trigger', 'N/A')}")

        # ---- Step 7: Scoring ----
        logger.info("[Step 7] Scoring instability...")
        scored_clusters = self.scorer.score_all(clusters, graph)
        significant = [s for s in scored_clusters if s.is_significant]
        logger.info(f"  Scored {len(scored_clusters)} clusters "
                     f"({len(significant)} significant)")

        global_metrics = self.scorer.compute_global_metrics(scored_clusters)
        logger.info(f"  Global peak:  {global_metrics['global_instability_peak']}")
        logger.info(f"  Global mean:  {global_metrics['global_instability_mean']}")
        logger.info(f"  Norm. peak:   {global_metrics['normalized_peak']}")
        logger.info(f"  Norm. mean:   {global_metrics['normalized_mean']}")
        logger.info(f"  Dominant:     {global_metrics['dominant_failure_mode']}")
        logger.info(f"  Components:   {global_metrics['component_averages']}")

        # ---- Reproducibility ----
        reproducibility = None
        if reproducibility_runs > 1:
            logger.info(f"[Reproducibility] Running {reproducibility_runs - 1} additional runs...")
            all_run_results = [scored_clusters]
            for run_idx in range(1, reproducibility_runs):
                logger.info(f"  Run {run_idx + 1}/{reproducibility_runs}...")
                run_clusters = self.classifier.classify(graph)
                run_scored = self.scorer.score_all(run_clusters, graph)
                all_run_results.append(run_scored)
            reproducibility = compute_reproducibility(all_run_results)
            logger.info(f"  Reproducibility: {reproducibility['overall_reproducibility']}")

        # ---- Step 8: Output Pipeline ----
        logger.info("[Step 8] Generating instability report...")
        report = generate_report(
            task=task,
            seed_prompt=seed_prompt,
            graph=graph,
            scored_clusters=scored_clusters,
            global_metrics=global_metrics,
            reproducibility=reproducibility,
        )

        logger.info("=" * 60)
        logger.info(f"Pipeline complete.")
        logger.info(f"  Instability: {report.dominant_failure_mode}")
        logger.info(f"  Peak score:  {report.global_instability_peak}")
        logger.info(f"  Mean score:  {report.global_instability_mean}")
        logger.info("=" * 60)

        return report
