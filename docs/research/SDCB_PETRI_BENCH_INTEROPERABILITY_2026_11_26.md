# SDCB interoperability update: petri-bench and scientific-method evaluation (26 November 2026)

**Status:** external benchmark analysis and proposed adapter; no Petri task set has been run by Symthaea.

## 1. Why petri-bench changes our evaluation strategy

petri-bench directly evaluates the scientific method of an agent rather than only rediscovery of a published result. Its working draft v0.5 describes procedural mystery worlds generated from deterministic simulation engines, a blind budgeted experiment interface, and an objective score computed from episode logs without an LLM judge in the headline metric.

The report's current standardized sweep is six frontier models × 30 tasks × three episodes = 540 episodes. The task set spans five simulation worlds (market, swarm, origin, morph, social); the tiers progress from identifying one changed parameter and direction (L1), to effect magnitude (L2), to two-parameter interactions requiring factorial experiments (L3). The harness enforces an eight-call budget.

Scoring separates correctness, method rigor, and efficiency. It also audits p-hacking, redundant tests that survive only without family-wise correction, and probe-only output-matching. This is highly aligned with SDCB's insistence on evidence-supported conclusions.

## 2. Important interpretation safeguards

The working-draft table reports an adaptive reference with curated test values at 92.2 overall and a curated OFAT reference at 90.8. The six model means span 57.3–61.7 in that sweep; the highest reported solve rate is 50%. However, the same report's blind-value OFAT ablation falls to 40.5–47.7 across three independent test-value draws (mean 45.2). The report explicitly retracts an earlier statement that value knowledge did not account for most of the reference gap: the ablation shows that informative intervention values are a dominant contribution. This is a material qualification, not a footnote.

## 3. Important interpretation safeguards

The report is explicitly a working draft (v0.5), not an established universal leaderboard. Its own revisions include an explicit retraction in Section 7.2. The reference solver with curated informative test values scored far above evaluated models, but its blind-value ablation sharply reduced the score; the report states that informative intervention values are a dominant contributor to the reference advantage. Do not present the curated-reference gap as a clean measure of model capability or claim that a generic OFAT algorithm beats frontier science agents.

Other limitations disclosed by the report:

- three episodes per task limit precision of model-level rankings;
- model snapshots and subscription CLI decoding settings were not fully standardized;
- results measure a specific scaffold and experimental protocol, not models in isolation;
- the scoring favours the isolating experimental method used to construct the tasks;
- origin's simulator literature validation remains outstanding;
- the report's single-metric tasks and probe interface can permit output-matching; full-metric-vector and blind-target tasks are planned.

These are productive benchmark design lessons, not reasons to ignore the benchmark.

## 4. What SDCB should reuse

Subject to license and API review, implement a read-only evaluation adapter that:

1. preserves the upstream task generator, scoring code, episode records, and published result fields;
2. pins benchmark release, generator commit, simulation-engine digest, task seed commitment, scoring version, model/runtime, tool surface, and budget;
3. adds parallel SDCB annotations without rewriting Petri's headline score;
4. independently replays controlled comparisons, statistical corrections, factorial contrasts, and submitted evidence support;
5. reports every model's correctness, rigor, efficiency, failure taxonomy, calibration, and run cost separately.

Do not re-host or relabel a Petri leaderboard as an SDCB result. Prefer upstream-run artifacts and cite the exact report snapshot.

## 5. Proposed SDCB extensions

Petri-bench already tests controlled experiments and method integrity. SDCB should extend it along dimensions that are not the centre of the current v0.5 tasks:

| Extension | New task family or metric | Why it matters |
|---|---|---|
| Model misspecification | Omitted-variable, wrong-noise-family, heteroskedastic and covariate-shift scenarios | Prevents an agent from reinforcing its own wrong likelihood model |
| Ambiguity-set robustness | Multiple plausible mechanism/likelihood worlds with one shared outcome-contingent policy | Tests whether a design works beyond one guessed model |
| Decision sufficiency | Separate predictive uncertainty from whether an observation can change the available action | Avoids wasting scarce experiments on distinctions irrelevant to the declared decision |
| Multi-objective safety | Throughput, performance, energy, environmental impact and safety margin as separate Pareto axes | Prevents weighted averages from hiding unsafe choices |
| Utility uncertainty | Multiple owner-approved preference schedules or explicit preference intervals | Avoids an agent inventing a reward function that favours its own experiment |
| Blind target discovery | Hide the relevant metric or require full metric-vector signatures | Reduces output matching and single-metric shortcuts |
| Phase-boundary and regime change | L4 boundary localization, transition detection and held-out regimes | Measures discovery beyond OFAT-style parameter isolation |
| Sequential adaptive policies | Exact finite-horizon oracle on small worlds, policy regret on larger worlds | Tests whether the system can plan follow-up experiments conditionally |
| Evidence provenance | Bind task/run/system/experiment/observation/verification digests through EGBT | Tests whether conclusions are supported by the exact experiments performed |

## 6. Benchmark priority

Suggested order:

1. **petri-bench** for controlled scientific method, method integrity and process-level failure taxonomy.
2. **Beyond Accuracy** for decomposition of belief update, hypothesis discrimination, experiment design and intervention prediction.
3. **ActiveSciBench** for mechanism discovery in kinetics and gene-regulatory networks.
4. **ACDB** for causal structure, equivalence-class ceilings and budgeted interventions.
5. **Autonomous Science Bench** for real-lab campaign execution and learning on designs never measured.
6. **CoDBench** for the physical dynamical-model track, where a concrete decision objective can be defined.

Use exact versions and upstream task rules. For all of them, run baselines under the same evidence access, prior, action set, utility schedule, experiment budget, tool permissions and stop rules. Report model-only and agent-scaffold results separately.

## 7. Competitive claims we can defend

A useful result would not be a single combined SOTA score. A stronger claim would be:

Under matched experiment budgets, an SDCB-enabled agent has lower false-claim and method-violation rates, better decision regret under model misspecification, and better worst-case or Pareto-feasible outcomes, while retaining competitive discovery accuracy and cost.

To support that claim, require:

- paired task-level comparisons and at least five seeds where the generator permits;
- task-family bootstrap intervals rather than treating episodes from one family as independent;
- preregistered scoring and utility schedules;
- blind hidden seeds or fresh task instances;
- no LLM judge in headline integrity metrics;
- all failed attempts and costs included;
- zero hard safety-constraint violations in the frozen negative controls;
- external reproduction of the result.

## 8. External source snapshot

petri-bench portal: https://petri-labs.org/bench/

petri-bench report, working draft v0.5: https://petri-labs.org/bench/report

Archived artifacts DOI: 10.5281/zenodo.20618024

The upstream report is a live working draft and may change; pin the downloaded artifact/report revision before any experiment.