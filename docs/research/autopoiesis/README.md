# Autopoiesis and Organizational Autonomy Research

This directory contains the research specification and operational model for AOBench.

## Start here

1. [AOBench specification](AOBENCH_SPEC_2026-11-13.md) — benchmark tracks, metrics, baselines, protocol and staged implementation.
2. [Organizational closure model](ORGANIZATIONAL_CLOSURE_MODEL_2026-11-13.md) — definitions, system boundaries, runtime evidence, causal interventions and claim limits.
3. [Organizational reconfiguration specification](AOBENCH_ORGANIZATIONAL_RECONFIGURATION_2026-11-14.md) — capability-preserving structural evolution compared with parameter-only and adaptive-topology baselines.
4. [AOBENCH-007 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/12113) — constrained organizational reconfiguration and capability preservation.
5. [AOBENCH-001 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
6. [MEL-AUTO-001 runtime regeneration work](https://github.com/Luminous-Dynamics/symthaea/issues/10098), which is the implementation foundation to reuse.

## Current evidence status

- The existing `AutopoieticMonitor` is a heuristic diagnostic, not a formal measure of autopoiesis.
- The AOBench documents are a research specification, not an implemented benchmark.
- No claim of biological life, sentience, subjective experience, consciousness, or strict biological autopoiesis is made.
- Benchmark targets are proposed gates, not measured results.
- The initial implementation should be deterministic, CPU-only and offline. Do not depend on queued CI to begin.

## Intended architecture

AOBench will separate declared organization from observed execution. A candidate system may declare components, processes and dependencies, but it only receives runtime credit when an independent verifier confirms exact input/output versions, recomputed content digests, causal lineage and explicit external dependency receipts.

The benchmark then measures perturbation recovery and pre-registered causal interventions. Open-world relevance discovery is a separate track and is not folded into a single “life score”.

## Contribution rule

Keep theoretical claims, executable behavior, structural checks, benchmark measurements and external replication clearly separated. Null and adverse results are publishable outcomes.
