## Transfer admission gate — research update 2026-11-10

This is an extension to MACE #9516, not a second memory engine.

### New external evidence

EvoAgentBench (Findings of EMNLP 2026) evaluates procedural ability transfer over 528 training and 267 test tasks across web research, algorithmic reasoning, software engineering and knowledge work. Its published findings say curated Ability content produces positive gains across the evaluated configurations, while automatic methods are inconsistent. Its live results page includes a severe negative-transfer cell: Memento on Nanobot / 27B / Software Engineering goes from 45.8 Vanilla to 9.5 after method injection (−36.3 points). Anchor Skill† is a curator-routed diagnostic reference, not a deployable automatic method.

Sources:
- https://aclanthology.org/2026.findings-emnlp.2344/
- https://evermind-ai.github.io/EvoAgentBench/
- https://github.com/EverMind-AI/EvoAgentBench/tree/main/benchmark

### Paradigm refinement

Memory needs **typed transfer semantics**. A factual memory, an episodic observation, a causal transition and a procedural ability are different objects and must not be admitted by one generic similarity score.

For procedure transfer:
1. Discover candidates using lexical, dense/HDC or graph retrieval.
2. Check immutable identity and evidence state.
3. Enforce domain, regime, objective, model, tool-version, capability and context preconditions.
4. Reject explicit contradictions; abstain on missing applicability evidence.
5. Only then use relevance to rank admitted candidates.

An empty objective or model scope is not a universal wildcard. Universal applicability needs an explicit agnostic declaration. Similarity is a discovery score, never transfer authority.

Causal transition knowledge has a separate contract: observational correlation does not identify an intervention effect; a qualified effect remains regime-scoped and must be reconciled against environment-side outcomes. A procedure can be useful without establishing a causal model, and a causal effect does not prove an operational procedure is executable.

### Executed synthetic reference

A standalone local package, broca-transfer-gate-v0.1, implements the typed applicability gate. It is a structured-input research oracle, not a production integration or ML model benchmark.

- 14/14 top-level tests passed.
- 10/10 hand-authored admission/reject/abstain cases matched.
- 256/256 deterministic irrelevant-context mutation controls preserved the decision.
- Race detector, vet and formatting passed.
- Statement coverage: 77.6%.
- Similarity-only selection was wrong or unjustified in 9/10 fixtures.
- Always-abstain / no-injection avoided unsafe exposure but missed both valid-transfer opportunities.

These are designed fixture counts, not sampled performance rates. The declared independent-root counts are not authenticated; digest consistency is not issuer authentication. No external EvoAgentBench runner, LLM, Rust integration or production test was run.

The accompanying offline archive is supplied separately in the conversation. SHA-256: bebd1c713b96b80e10a39131b41fc975426d354e0e5eb05e80e1abfd2888730a.

### Next qualification

Use the official EvoAgentBench runner and frozen 528/267 split. Compare Vanilla, Memento, ReasoningBank, GEPA, similarity-only injection, typed gate, and gate + automatic routing. Preserve Anchor Skill† only as a curator-routed diagnostic ceiling. Freeze model, agent scaffold, budget and evaluator; report every domain/scaffold/model cell.

Add metrics for negative-transfer rate, transfer gain, worst-cell regret, selective transfer risk/coverage, ability-hit precision, and end-to-end cost. Proposed goal: positive paired aggregate gain with a 95% interval above zero, with no domain/scaffold cell suffering a statistically supported loss greater than one percentage point. This is a project target, not an achieved result.