# MACE causal-transfer benchmark matrix — proposed, not measured

All external benchmark results must use official splits/evaluators where permitted. Scores from different backbones, scaffolds, datasets, and judge protocols are not directly comparable.

## Research position

Existing strong work already covers evidence-grounded skill co-evolution (MSCE), systematic skill extraction/consumption and meta-skills (SkillLens), multi-domain procedural transfer (EvoAgentBench), long-horizon agentic trajectories (AMA-Bench), and interdependent multi-session agent-environment loops (MemoryArena). MACE should not claim novelty for causal graphs, evidence links, skills, or applicability tags alone.

MACE's proposed differentiation is narrower: a skill transfer decision should preserve the distinction between observed association and identified causal effect; carry objective and regime context; reconcile claimed outcomes with environment observations; and retain an explicit identified set when the effect is not point identified.

## External tracks

| Track | Benchmark / method | Primary question |
|---|---|---|
| Procedural transfer | EvoAgentBench | Does a learned ability improve performance in held-out tasks, and where does it cause negative transfer? |
| Skill lifecycle | MSCE | Does evidence-grounded co-evolution of traces, policies, world cognition and callable skills improve task performance? |
| Skill generation / consumption | SkillLens | Which extractor produces useful skills for which target agent, and can a meta-skill reduce negative transfer? |
| Multi-session action transfer | MemoryArena | Does memory acquired in one environment session improve later dependent tasks? |
| Long-horizon trajectories | AMA-Bench | Does causal/objective-aware retrieval improve memory over agent-environment traces? |
| Tool use | Mem2ActBench | Does remembered knowledge yield correct tool choice and argument grounding? |
| Causal identification | CLeaR 2026 partial-identification work; NeurIPS 2025 data-fusion bounds | Are conclusions robust under unmeasured confounding or explicit assumption violations? |
| Synthetic exact oracle | MACE-PI-001 | Are finite-cohort bounds and threshold decisions exact under the declared assumptions? |

## Matched baseline ladder

- B0: no persistent memory.
- B1: full-history / full-context diagnostic ceiling.
- B2: BM25.
- B3: dense / hybrid / HDC retrieval under equal context budgets.
- B4: similarity-only model-generated skills, without applicability or causal checks.
- B5: MSCE reproduction (including co-evolving memory, policy, world model, skill and feedback update).
- B6: SkillLens extractor/consumer and meta-skill baselines.
- B7: typed procedural admission gate, without causal partial identification.
- B8: MACE causal experience memory with objective, regime, pre-action prediction and outcome reconciliation.
- B9: MACE + finite-cohort partial-identification gate.
- B10: curated or oracle skill and causal-model references (diagnostic ceilings only; not deployable automatic baselines).

Freeze the exact backbone, agent scaffold, prompt, context/token budget, action/tool budget, data split, environment seed and evaluator within each paired comparison. Report results for each backbone/domain/scaffold cell.

## Metrics

### Transfer and decision quality
- paired transfer gain versus the identical no-memory agent;
- negative-transfer incidence and magnitude;
- valid-transfer recall / ability-hit precision;
- worst-cell regret and worst-domain performance;
- downstream task-chain success and delayed utility;
- objective-change and regime-shift recovery;
- selective transfer risk/coverage and abstention quality.

### Causal validity
- false causal promotion rate;
- outcome-reconciliation failure rate;
- identification interval width and coverage on synthetic known-truth data;
- decision sign/threshold stability as assumptions relax;
- explicit rates of NOT_IDENTIFIED, cross-regime rejection and unsupported causal transfer.

### Memory lifecycle and efficiency
- memory bytes including index, traces, tombstones and proof metadata;
- write and consolidation cost;
- retrieval and verification latency p50/p95/p99;
- tokens, model/tool calls, CPU/GPU time and energy/cost;
- cost per completed dependent task and cost per valid transfer.

## Proposed gates — targets, not results

1. 100% oracle agreement on deterministic finite-cohort and admission fixtures.
2. Zero robust intervention outputs when the compatible effect set crosses the declared action threshold.
3. Zero implicit cross-regime transfer.
4. No external transfer claim unless paired transfer gain's 95% confidence interval excludes zero under a preregistered analysis.
5. No statistically supported domain/scaffold deterioration greater than one percentage point; correct multiplicity and clustering treatment.
6. Reduce negative-transfer rate versus the strongest matched automatic skill baseline without achieving the target merely by abstaining on every case.
7. At no more than one point task-quality non-inferiority, target 25% lower full lifecycle cost; alternatively target at least three points gain at equal declared cost. These are project decision rules, not industry standards.

Keep identification regions, sampling confidence intervals, and calibrated predictive intervals as separate statistical objects.
