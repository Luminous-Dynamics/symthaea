# Complementary Cognitive Architectures: Research and Evaluation Plan

**Status:** Proposal for experimentally falsifiable prototypes; no architecture is declared superior yet.  
**Date:** 2026-10-15  
**Scope:** Public Symthaea research, with evidence and simulation responsibilities for Mycelix and Symtropy.

## Executive recommendation

Do not replace Symthaea's HDC + liquid-dynamics foundation with another fashionable monolith. Build **complementary, swappable architectures around a shared task/evidence contract**, then compare them under equal budgets. The leading candidate to prototype is a three-part design:

1. **Semantic-compositional representation:** retain HDC/VSA for binding, bundling, similarity, and compact associative cues.
2. **Explicit predictive world state:** maintain typed variables, temporal state, and causal/relational structures where exact relations or counterfactuals matter; use CfC/LTC dynamics for suitable sequence state, not as a universal memory.
3. **Verifier and resource-aware coordinator:** route work among memory, prediction, causal analysis, planning, and exact tools; track uncertainty and ask for independent verification when the evidence is weak.

This is a testable engineering hypothesis, not a claim that the architecture is generally superior or that its internals establish consciousness.

## Why this direction

Symthaea already has substantial proposals and implementations around a unified HDC substrate, a cognitive workspace, causal reasoning, temporal dynamics, predictive self-modeling, and substrate independence. The main opportunity is to test their boundaries and interactions rather than create a competing all-encompassing design.

Useful local starting points:

- [Unified architecture proposal](../architecture/UNIFIED_ARCHITECTURE_PROPOSAL.md)
- [Cognitive self-model design](../architecture/COGNITIVE_SELF_MODEL.md)
- [Unified causal-understanding benchmark](../../src/benchmarks/unified_causal_understanding.rs)
- [HDC substrate-independence module](../../crates/core/symthaea-core/src/hdc/substrate_independence.rs)

The Cognitive Self-Model document specifically recommends extending existing `PredictiveSelfModel` machinery instead of creating a duplicative parallel abstraction. Follow that principle across the whole experiment: inventory first, adapt interfaces second, add a new component only when the existing one demonstrably cannot satisfy the required contract.

## Candidate architecture families

These are complementary hypotheses, not mutually exclusive products. Initially, compare each as a separate configuration against the current baseline. Combination experiments come only after each individual contribution is measurable.

| Candidate | Core idea | Best-fit hypothesis | Main failure mode to test |
|---|---|---|---|
| **A. Hierarchical multi-timescale loop** | Fast reflex/monitoring path, medium-timescale HDC + CfC state, slower planning and consolidation; events can trigger extra computation | Improves responsiveness on irregular streams and long tasks without running every subsystem at full rate | Stale slow-state, oscillation between levels, missed events |
| **B. Dual representation: HDC + typed relational/causal state** | HDC handles compositional cues and associative recall; typed graphs/tables carry exact entities, relations, units, and causal assumptions | Improves multi-hop relational and counterfactual tasks without sacrificing useful similarity-based retrieval | Representation disagreement; lossy or untraceable conversion between spaces |
| **C. Three-store memory** | Separate episodic traces, consolidated semantic knowledge, and procedural skills; every stored item has source/time/confidence metadata | Improves long-horizon recall, adaptation, and provenance-sensitive retrieval | Memory pollution, catastrophic interference, stale or contradictory facts |
| **D. Sparse specialist / workspace routing** | Existing modules submit structured bids; a coordinator selects the smallest sufficient set of specialists under latency and energy budgets | Improves cost per successful task and limits irrelevant module work | Router starvation, hidden coordination costs, unstable specialist monopolies |
| **E. Predictive self-model + calibrated uncertainty** | Predict expected outcome/confidence, score prediction errors, calibrate reliability, and trigger review or abstention | Improves error detection, confidence calibration, and adaptation after regime shifts | Self-referential feedback loops; apparent calibration without external truth |
| **F. Verified symbolic/tool sidecar** | For exact domains, convert typed claims to executable constraints, solvers, tests, or proof checkers; retain source-bound output receipts | Improves exact math, code, units, engineering constraints, and safety-critical calculations | Invalid translation from semantic intent to formal problem; checker coverage gaps |
| **G. Multi-hypothesis causal planner** | Generate a bounded set of competing explanations/actions, test counterfactual predictions, then select using an explicit objective and constraints | Improves ambiguous diagnosis, scientific design, and planning under uncertainty | Correlated hypotheses, combinatorial blow-up, shared blind spots |
| **H. Event-driven hybrid runtime** | Run cheap monitors continuously; wake heavier components on change, novelty, uncertainty, deadlines, or explicit requests | Reduces average energy and tail latency on sparse or mostly quiet workloads | Missed wake-up conditions, starvation, noisy events causing thrashing |

### What may be superior, and where

A plausible **superior candidate for long-horizon, tool-using, embodied tasks** is the combination of A + B + C + D + E + F: multiscale control, dual representations, tiered memory, sparse routing, calibrated uncertainty, and independent verification.

A smaller **superior candidate for embedded devices** may instead be A + B + H with bounded memory and a local verifier. A scientific-reasoning profile may emphasize B + C + E + F + G. We should not assume a single configuration wins across all workloads.

The design principle is: **share protocols and evidence, not necessarily internal representations**. Forcing all facts, causal structures, plans, motor state, and uncertainty into one undifferentiated hypervector may make interoperability convenient while obscuring types, units, provenance, and exact relational structure.

## Proposed component contracts

Keep the interfaces small enough for existing Rust modules to adopt incrementally. These names are conceptual until matched to the real codebase; they are not a request to duplicate existing types.

- **InputEnvelope:** payload reference, modality, event time, received time, source/provenance, sensitivity/permission label.
- **StateSnapshot:** versioned state plus the time horizon and assumptions it covers.
- **Prediction:** target, predicted value/distribution, horizon, assumptions, uncertainty, and expected verification method.
- **EvidenceItem:** observation or result, source identity, content hash where appropriate, timestamp, transformations, and confidence basis.
- **ActionProposal:** intended effect, preconditions, predicted outcome, resource estimate, reversibility, and required checks.
- **VerificationResult:** pass/fail/inconclusive, checker identity/version, exact inputs, diagnostics, and limitations.
- **Budget:** deadline, compute/energy/memory ceilings, and permitted specialist calls.

All module calls should expose failures explicitly. A missing prediction, failed tool, timeout, numerical error, or incomplete evidence must not silently become a confident answer. Each output should preserve the distinction between observation, inference, hypothesis, and verified result.

## Evaluation design

### Baselines and controls

Use the current integrated HDC + liquid-dynamics pipeline as baseline (B0). Add the simplest alternative needed for each task (for example, HDC-only retrieval for memory tasks or a conventional recurrent/temporal baseline for sequence prediction). Then run B1–B8 as independent ablations before testing combinations.

Preserve negative controls:

- **Fixed router vs. resource-aware router**
- **No uncertainty feedback vs. calibrated uncertainty**
- **One representation vs. dual representation**
- **Single-store vs. tiered memory**
- **No verifier vs. independent verifier**
- **Fixed-rate vs. event-triggered scheduling**
- **Shuffled or source-mismatched evidence** to confirm provenance checks detect the mismatch
- **Budget-matched simpler baseline** to prevent attributing gains to merely spending more compute

Freeze dataset splits, seeds, baseline configurations, scoring rules, and stopping conditions before interpreting results. Keep development and held-out sets separate. Do not use the same scenarios for design iteration and final generalization claims.

### Workload suite

1. **Associative retrieval:** noisy cues, distractors, compositional bindings, and memory growth. Score precision/recall and latency.
2. **Irregular temporal prediction:** variable time gaps, missing observations, regime shifts, and long gaps. Score predictive error and calibration by horizon.
3. **Causal and counterfactual reasoning:** interventions distinguishable from correlations, plus known ground-truth causal graphs. Score intervention/counterfactual accuracy.
4. **Relational and unit-sensitive engineering:** multi-step constraints, dimensional analysis, exact arithmetic, and inconsistent inputs. Score exact correctness and checker acceptance.
5. **Long-horizon memory:** episodic recall, conflicting updates, delayed queries, and provenance-sensitive questions. Score retrieval quality, stale-memory errors, and source attribution.
6. **Uncertainty under distribution shift:** held-out regimes, ambiguous cases, out-of-distribution inputs, and abstention opportunities. Score Brier/log loss when appropriate, expected calibration error with bin definitions frozen in advance, selective risk, and coverage.
7. **Resource-bounded embodied control:** deterministic simulated tasks with deadlines, sensor delays, and actuator constraints. Score goal completion, constraint violations, latency, memory, and energy where hardware measurements are available.
8. **Fault injection:** corrupted messages, unavailable specialists, timeout, inconsistent state versions, and checker failure. Score safe fallback behavior and evidence preservation.

Report task metrics separately. Do not collapse correctness, uncertainty, speed, energy, and safety into one convenient scalar without showing the underlying trade-offs.

### Performance and evidence metrics

For each exact commit/configuration, record:

- Task-level quality and per-case outcomes, not just averages.
- Median and p95 end-to-end latency, throughput, and deadline misses.
- Peak resident memory and memory traffic where measurable.
- Energy per successful task/cognitive cycle on physical hardware; identify the measurement method and uncertainty.
- Confidence calibration, abstention coverage, and selective error rate.
- Recovery and degradation under injected faults.
- Component call counts, data movement, and overhead introduced by orchestration.
- Reproducibility identifiers: commit SHA, configuration hash, seeds, data/split checksums, tool versions, machine/accelerator identity, and raw result paths.

A faster kernel is not an end-to-end improvement if transfers, conversions, or scheduling make the complete task slower. Simulated energy is a model result, not measured energy. Queued, skipped, cancelled, or stale CI results are not passing evidence.

## Cross-project division of work

### Symthaea

Own task semantics, cognitive-module adapters, benchmark tasks, reference implementations, ablations, runtime routing, and final interpretation. Reuse existing causal, temporal, memory, workspace, and predictive-self-model code after an implementation-level inventory.

### Symtropy

Use its physical/dynamical modelling capabilities to explore schedule, energy, thermal, latency, and embodied-constraint scenarios. Clearly label assumptions and model versions. Before treating simulation output as predictive, calibrate it against measurements and report residual error. Do not let a simulated advantage override the actual hardware benchmark.

### Mycelix

Provide optional experiment provenance and collaboration coordination: immutable experiment identity, source/data permissions, contributor attribution, signed or otherwise verifiable result receipts where implemented, and reproducible links to artifacts. Mycelix's governance layer should not be in the hot path of every local cognitive operation, and a stored receipt is not itself proof that a scientific result is correct.

Keep these responsibilities separate so that each system can be tested independently. Cross-repository integration should be introduced through versioned, narrow contracts rather than direct dependence on internal types.

## Research basis and cautions

- Hasani et al., **“Closed-form continuous-time neural networks,”** *Nature Machine Intelligence* (2022): [paper](https://www.nature.com/articles/s42256-022-00556-7). Provides evidence that CfC-style models can model time-series dynamics efficiently; it does not show that CfC should own all memory or reasoning.
- Vergés et al., **“Classification using hyperdimensional computing: a review with comparative analysis,”** *Artificial Intelligence Review* (2025): [paper](https://link.springer.com/article/10.1007/s10462-025-11181-2). Reviews HDC/VSA composition and classification; benchmark results remain task-dependent.
- **“Hyper-dimensional computing architectures for AI applications,”** *Results in Engineering* (2026): [paper](https://doi.org/10.1016/j.rineng.2026.109589). Reviews architecture and hardware options, including scalability and implementation trade-offs.
- Hodson et al., **“The empirical status of predictive coding and active inference,”** *Neuroscience & Biobehavioral Reviews* (2024): [PubMed record](https://pubmed.ncbi.nlm.nih.gov/38030100/). Finds promising results but emphasizes that stronger comparisons against alternative models are required.
- van Zwol et al., **“Predictive Coding Networks and Inference Learning: Tutorial and Survey,”** *ACM Computing Surveys* (2026): [DOI](https://doi.org/10.1145/3797870). Useful for implementation variants; not evidence that a specific predictive-coding integration will improve Symthaea.

Neuroscience-inspired architectures are design inspiration, not proof of biological equivalence, sentience, or causal adequacy. Predictive coding, active inference, integrated information, or self-model scores must not serve as substitutes for task-level validation.

## Execution order

1. **Inventory:** map current implementations and contracts for workspace/attention, HDC, CfC/LTC, causal reasoning, memory, predictive self-model, uncertainty, and verifiers. Mark each as implemented, proposed, tested, or unknown.
2. **Select three discriminating tasks:** associative retrieval, irregular temporal prediction, and counterfactual reasoning. Add exact-tool verification as a fourth task only if the repository already has a runnable path.
3. **Freeze baseline and protocol:** store a machine-readable task/config schema and a result format before tuning.
4. **Add one ablation at a time:** demonstrate the individual contribution of dual representations, multi-timescale routing, and uncertainty-triggered verification.
5. **Combine only proven components:** use a fixed compute budget and include orchestration overhead.
6. **Independent verification:** have a separate checker validate manifests, split identity, metric calculations, and exact-head provenance before results are described as passing.
7. **Promote cautiously:** integrate a configuration only when it beats the relevant baseline on its predeclared target without unacceptable regressions on latency, resource use, calibration, or safety constraints.

**First deliverable:** a source-derived architecture inventory plus a small, deterministic comparative harness. Do not start with new hardware, new consciousness metrics, or a wholesale rewrite. The initial aim is to learn which architectural differences change observable capability, and under what cost.
