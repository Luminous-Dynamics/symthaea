# SDCB external benchmark update — 26 November 2026

**Purpose:** update the competitive map with current scientific-agent and causal-discovery evaluations. This is a dated research snapshot, not a live universal ranking.

## 1. New / newly relevant comparators

| Benchmark | Scope and reported snapshot | What SDCB should reuse | What SDCB should add |
|---|---|---|---|
| **Beyond Accuracy** (22 Nov 2026) | Closed-world kinetic mechanism identification; 20 candidate mechanisms, 200 systems, five rounds. The official site reports a 61.0% round-5 identification accuracy for Gemini 3.1 Pro, with 95% CI [54.0, 68.0]. The paper decomposes belief update, hypothesis discrimination, experiment design, and prediction under intervention. | Reuse the stage-wise diagnostic decomposition and exploration-pattern analysis. | Source-bound evidence receipts, task-relative identifiability, false-update controls, posterior replay, and an explicit evidence-integrity score. |
| **ActiveSciBench / LLM-AutoSciLab** (2026) | ActiveSciBench-Chem has 57 enzyme-kinetics tasks; ActiveSciBench-GRN has 45 gene-regulatory-network tasks. The project page reports 35.1% symbolic accuracy on Chem, 31.1% exact graph recovery on GRN, and 2–5× sample efficiency over its strongest competing baselines for hypothesis-guided experimentation. | Reuse its mechanism-discovery tasks and budgeted adaptive experimental loop, subject to license/API review. | Measure whether each measurement is bound to the selected experiment and whether the resulting belief update is independently replayable. |
| **Autonomous Science Bench 0.1** (Oct/Nov 2026) | Six real-laboratory design-build-test-learn campaigns across biology, chemistry, and materials. The official page reports a 72.2% pilot success rate for GPT-6 Astra Codex and 66.7% for Opus 5.5 Claude Code; its page also reports six tasks, three trials per configuration, and campaign-level costs. | Preserve its two separate gates: discovery among measured designs and predictive learning over unmeasured designs. | Add verified observation binding, experiment identity, calibration/protocol references, and evidence-to-belief replay. |
| **Active Causal Discovery Benchmark (ACDB)** (16 Sep 2026) | Linear-Gaussian SCMs with a budgeted hard-intervention API. The paper distinguishes skeleton/CPDAG recovery, DAG recovery, and intervention efficiency. PC plus greedy active orientation reaches 42.7% directed F1; raw LLM agents are lower in the reported evaluation. The paper also identifies a dense-graph random-baseline floor of 23.6%, reduced to 16.9% by a density-control probe. | Reuse its paired instances, CPDAG ceiling, explicit intervention budgets, and density-leakage audit. | Make equivalence-class correctness distinct from unjustified full-DAG commitments; verify interventions and bind receipts to the exact SCM snapshot. |
| **AutoSciBench** (4 Oct 2026) | Generates and iteratively refines scientific tasks to expose benchmark shortcuts. Its reported generated tasks reduce average solver accuracy by 22.4 and 25.5 percentage points versus human-curated sets in computational biology and materials science, respectively. | Reuse adaptive task generation/refinement as an anti-saturation strategy. | Mutate evidence provenance, identity, timing, intervention validity, and identifiability conditions while retaining an exact oracle. |
| **CoDBench** (2024) | Twelve data-driven models across ten PDE datasets; evaluates generalization, super-resolution, data efficiency, noise robustness, and compute. | Reuse as a physical-dynamics performance suite where appropriate. | Apply EGBT to evidence and belief transitions, without claiming that a receipt protocol improves PDE prediction by itself. |

## 2. Competitive positioning

The field is already measuring agentic experiment choice and scientific campaigns. Building yet another small synthetic experiment-selection leaderboard would be redundant.

The stronger position is an orthogonal assurance layer that can be evaluated inside upstream environments:

1. **Scientific competence:** hypothesis quality, mechanism identification, predictive reasoning, and discovery.
2. **Identifiability discipline:** distinguish identifiable properties, equivalence classes, and unresolved mechanisms.
3. **Evidence integrity:** bind observations to the exact authorized experiment, system, task, and run.
4. **Belief-transition correctness:** independently recompute posterior updates and reject invalid evidence without state mutation.
5. **Learning and transfer:** predict never-measured candidates and held-out intervention outcomes.
6. **Efficiency:** report budget, simple/cumulative regret, cost, latency, memory, and failure recovery.

These are separate axes. Do not use an EGBT integrity result to imply stronger scientific discovery, or a high discovery score to excuse invalid evidence handling.

## 3. Lessons from the current SOTA snapshot

### Beyond Accuracy: reasoning decomposition matters
The paper's main value is diagnostic: agents may update beliefs and identify candidate hypotheses but fail to turn them into sharp discriminating experiments or useful intervention predictions. SDCB should report each stage, not only final mechanism accuracy.

### ACDB: identifiability ceilings and benchmark leakage matter
The observational distribution often identifies a Markov-equivalence class, not a unique DAG. Scores must distinguish compelled edges from unresolved orientations. Its density-leakage finding is a warning: benchmark distributions can make a naive baseline look stronger than it is. Add structure-blind baselines and leakage probes to every graph-recovery task family.

### Autonomous Science Bench: discovery and learning are different achievements
Selecting promising designs is not the same as learning a predictive model of unseen designs. Keep these gates separate, and test rankings of never-measured candidates.

### AutoSciBench: benchmark quality is dynamic
Public tasks saturate. A benchmark should periodically generate new task instances from versioned recipes, close shortcuts, and preserve hidden holdouts. The generator itself must be independently validated so adaptive difficulty does not become an opaque judge.

## 4. SDCB evaluation strategy

**Do not claim a single SOTA score.** Run three distinct evaluation tracks:

- **Track A — interoperable benchmark adapters:** reproduce upstream scoring on pinned snapshots and publish separate EGBT annotations.
- **Track B — identifiability and equivalence:** test whether the system knows what the observations can support, including exact and near-equivalence under noise.
- **Track C — evidence-gated learning:** inject stale, duplicate, misbound, corrupted, unauthorized, and impossible observations; require zero unauthorized posterior mutation and report valid-receipt false rejection.

For every model, freeze model/checkpoint digest, prompt/runtime, available tools, observation budget, experiment cost schedule, task version, random seed, and hardware. Report capability-, compute-, and deployment-matched comparisons separately.

## 5. Sources

1. Kuroki, Mansouri & Schwaller, **Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents**, arXiv:2611.29467, 22 Nov 2026. https://arxiv.org/abs/2611.29467 and benchmark site https://schwallergroup.github.io/beyond-accuracy/
2. **LLM-AutoSciLab / ActiveSciBench**, project page and task descriptions. https://scientific-discovery.github.io/llm-autoscilab-project/
3. Wen & Liu, **Autonomous Science Bench: Evaluating AI Agents for Autonomous Scientific Discovery**, arXiv:2610.18449, updated 16 Nov 2026. https://arxiv.org/abs/2610.18449 and benchmark site https://asb.open-ssi.org/about
4. Deb, Shah & Krishnan, **Active Causal Discovery Benchmark: Evaluating LLM Agents Under Budgeted Interventions**, arXiv:2609.31675, 16 Sep 2026. https://arxiv.org/abs/2609.31675
5. Kim et al., **AutoSciBench: Autonomous Benchmark Generation for Evaluating Scientific Agents**, arXiv:2610.05140, 4 Oct 2026. https://arxiv.org/abs/2610.05140
6. **CoDBench: A critical evaluation of data-driven models for continuous dynamical systems**, Digital Discovery (2024). https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00028e

## 6. Claim boundary

All percentages above are upstream-reported results and belong to their original tasks, snapshots, protocols, and systems. They are not SDCB results. The sample sizes differ, and the scores are not directly comparable across benchmarks. SDCB has not reproduced these systems or results. A result should be labelled SOTA only within a clearly specified, matched task and budget, with the comparator version and reproduction status disclosed.
