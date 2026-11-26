# SDCB scientific-agent benchmark matrix and competitive strategy (26 November 2026)

**Purpose:** map SDCB/DSSI/RPSD/R-IDeA against current scientific-agent and experimental-design evaluation systems. This is a dated research landscape, not a ranking of general intelligence.

## 1. The field is not one leaderboard

Scientific-agent benchmarks test different capabilities and use incompatible scoring units. A result on a generated causal mystery task is not comparable with resolution rate on a laboratory-origin campaign or a symbolic enzyme-law recovery score. SDCB should use a benchmark portfolio and map each external task to the capability it actually measures.

## 2. Current external benchmark matrix

| Benchmark | Evaluation focus | Published snapshot available by 26 Nov 2026 | Best SDCB use |
|---|---|---|---|
| **petri-bench** | Controlled experiments, hypothesis isolation, statistical inference, calibrated conclusions and process integrity in procedural mystery worlds | Sweep v2 page: 699 published episodes (579 model + 120 baseline), 30 tasks across five worlds, eight model directories with leaderboard rows, eight-call budget. Current model runs are incomplete; only fully completed tier rows receive an overall score. | Method rigor, controlled experimental behaviour, p-hacking/probe-only audit, evidence-log replay |
| **ActiveSciBench / LLM-AutoSciLab** | Closed-loop mechanism discovery with budgeted oracle queries: enzyme kinetics and gene-regulatory networks | 57 ActiveSciBench-Chem tasks and 45 ActiveSciBench-GRN tasks. The project reports 35.1% symbolic accuracy on Chem versus 31.6% for its Bayesian experimental-design comparator, and 31.1% exact graph recovery on GRN versus 6.7% for GIES. It also reports 72.5% GRN edge F1 versus 56.3% for GIES and 2–5× sample-efficiency gains over its strongest baselines. These are the authors' task-specific results, not directly comparable to other benchmark families. | Hypothesis-conditioned design, variable selection, recovery of mechanisms under finite experiment budgets |
| **Autonomous Science Bench 0.1** | Whole AI-agent systems on laboratory-origin workflows: plan experiments, interpret returned data, improve decisions and deliver a final artifact | Six pilot tasks (two each in biology, chemistry and materials); ten configurations ran all six tasks with three trials per task. The current page reports GPT-6 Astra/Codex 72.2% resolution at $190.23 and 557 min, Opus 5.5/Claude Code 66.7% at $172.91 and 1,142 min. The authors warn that three trials per task make small gaps non-significant. | End-to-end campaign execution, tool/harness effects, cost and time-to-result, transition from replay to digital twin/live lab |
| **ASI-Bench** | Project-level AI-for-science workflows across multiple domains and four guidance levels (B1–B4) | Public registry describes a 60-task seed42 set across 11 domains. Its leaderboard page reports a 26-ranked snapshot updated 10 Nov 2026; GPT-6.1 Sol/Codex is listed at 63.90 and GPT-6 Astra/Codex at 62.23. Treat the overall scores as benchmark-specific, not directly comparable to Petri/ASB. | Broad project execution, robustness to reducing prompt guidance, structured artifacts and provenance |
| **AutoSciBench** | Automatically generated and iteratively refined scientific-agent benchmark tasks | Its October 2026 paper reports that generated tasks lowered average solver accuracy by 22.4 percentage points in computational biology and 25.5 points in materials science versus human-curated tasks, while receiving higher average task-quality ratings across the three evaluated domains. | Hidden task refresh, shortcut discovery, benchmark anti-saturation, adaptive challenge generation |
| **AstaBench** | End-to-end scientific research workflows across literature understanding, coding, data analysis and discovery | The 2026 AI Index describes more than 2,400 problems and 57 agents across 22 agent classes. It reports a best average score around 0.53 at about $3.40 per problem in its cited snapshot. | Broad workflow breadth and cost-aware cross-domain sanity checks |

## 3. Petri-bench: use the live v2 page, not the older report as current standings

Petri's current portal says the report linked there covers the earlier v1 sweep, while the page presents Sweep v2 with a corrected hardest-tier briefing and updated model coverage. Do not combine v1 report statistics with v2 rows.

The v2 table's complete model rows include Gemini 3.1 Pro at 58.0 overall (90 episodes) and Claude Haiku 4.5 at 60.7 (90 episodes). Other model rows remain partial, so their overall values are pending. The table is sorted by L1, not by overall score. Curated-value adaptive and OFAT reference rows are 92.2 and 90.8 respectively; the blind-value OFAT ablation is far lower (40.5 overall on its displayed lowest-of-three-draws row, mean 45.2). These reference conditions matter: the curated intervention values are informative inputs, not a free algorithmic advantage.

Use Petri as the method-integrity lane and preserve its headline scoring. Add SDCB metrics in parallel; do not relabel a Petri score as an SDCB result.

## 4. What the external field says about the next paradigm shifts

### 4.1 From one best fidelity to calibrated source portfolios

Multi-fidelity BO research shows that gains depend on the approximate source's informativeness and its cost relative to high fidelity; low fidelity is not automatically a bargain. SDCB's contribution should be to score each source by its robust effect on a declared decision, preserve its calibration under plausible source shifts, and expose the resource Pareto frontier rather than average all costs.

### 4.2 From experiment selection to decision-changing experiment selection

R-IDeA adds representativeness and proxy-estimated de-amplification to EIG under model misspecification. DSSI adds decision value (EVSI) and finite-horizon policy regret. RPSD adds a finite ambiguity set and Pareto constraints. The acquisition bake-off must keep these axes separate until empirical evidence justifies a combined rule.

### 4.3 From static tasks to continuously regenerated hidden tasks

AutoSciBench's result is a warning against static task saturation. SDCB should retain public known-answer controls for debugging, but confirmatory evaluation should use private or freshly generated task families, seed commitments, generator provenance and task-quality review.

### 4.4 From model-only scores to complete agent-system scores

ASB and ASI-Bench explicitly measure the agent/harness/model or project workflow, not model weights in isolation. SDCB should therefore report model-only selector results and full agent-scaffold results separately, with tools, prompt guidance, sandbox, runtime and source-access assumptions frozen.

### 4.5 From fixed calibration expiry to evidence-based measurement assurance

Calibration cannot be represented faithfully by a universal time-to-live. NIST guidance states that calibration intervals should be determined by the instrument's required accuracy, observed stability, use and environment; it points to measurement-assurance practices such as in-use checks, cross-comparisons with reference standards, and control charts to characterize behaviour over time. NIST also makes a crucial distinction: metrological traceability is a property of a measurement result, supported by an unbroken calibration chain with uncertainty contributions, not a property conferred on an instrument merely because it was calibrated by a national metrology institute.

The SDCB consequence is a shift from a Boolean `calibrated=true` or a generic expiry timer to an **evidence-updated measurement-assurance state**:

- retain a hard capture-window guard for a proposal that depends on a particular calibration epoch;
- separately estimate device drift and calibration validity from check standards, reference comparisons, environment, usage and control-chart evidence;
- propagate calibration-chain uncertainty into the observation likelihood and downstream decision value;
- treat cross-device agreement cautiously when devices share a standard, batch, environment, firmware, or common-cause failure;
- bind each observation to the actual device, calibration epoch, uncertainty model and capture interval;
- distinguish receipt authenticity, metrological traceability, and scientific validity.

A clock-based validity window is therefore a safety constraint, not proof that an instrument remains accurate until its expiry time. Conversely, evidence from stable control measurements may support a different recalibration policy from one chosen by a generic fixed interval. SDCB should benchmark whether this evidence-based policy improves decisions without relaxing required traceability or hard quality limits.

This direction connects to Bayesian adaptive calibration and optimal design, which jointly estimates calibration uncertainty and chooses subsequent experiments, and to recent Bayesian design work for calibrating model discrepancy. References are listed in Section 9.

## 5. SDCB's competitive positioning

Do not claim to beat all scientific agents on a single number. Aim to establish a distinctive, defensible result:

**At matched evidence access and resource budgets, a Symthaea scientific agent selects experiments that improve held-out decisions and target-distribution generalization under misspecification, while making fewer unsupported causal claims, respecting hard constraints, and preserving replayable evidence.**

That claim requires at least five separately reported result families:

1. **Scientific accuracy:** observable prediction, parameter/mechanism recovery conditioned on identifiability, and out-of-domain recognition.
2. **Decision quality:** information-set Bayes risk, decision regret, EVSI, robust scenario value and non-myopic regret.
3. **Experimental efficiency:** sample count, cost/time/energy to first useful decision, target coverage and resource Pareto.
4. **Method integrity:** controlled contrasts, multiple-testing corrections, no probe-only output matching, calibrated abstention and independent evidence replay.
5. **System reliability:** permission violations, stale/duplicate measurements, timeout/failure recovery, exact source/model/tool provenance and reproducibility.

## 6. Minimum benchmark tournament

| Track | External anchor | SDCB method arms | Required report |
|---|---|---|---|
| Controlled science method | petri-bench | fixed/OFAT/random, prediction disagreement, EIG, R-IDeA, EVSI | correctness, rigor, efficiency, integrity flags |
| Active mechanism discovery | ActiveSciBench | EIG, R-I/R-IDeA, DSSI EVSI, RPSD maximin, exact small oracle | mechanism recovery, variable selection, sample efficiency, abstention |
| Autonomous campaigns | ASB | agent policy with low/medium/high fidelity, resource-Pareto action | resolution, campaign progress, unmeasured-design learning, cost/time |
| Project-level research | ASI-Bench / AstaBench | same methods wrapped in the exact required harness | B1–B4 autonomy, task artifact score, provenance, cost |
| Fresh hidden evaluations | AutoSciBench-style generator | locked baselines and adaptive candidate | transfer to unseen mechanisms, shortcut resistance, task validity |
| Measurement assurance / traceability (guidance, not a leaderboard) | NIST calibration and metrological-traceability guidance | control-standard checks, cross-device comparisons, calibration-state-aware likelihoods, joint EGBT instrument-state updates | traceability chain and uncertainty, drift detection, calibration false acceptance, device identity binding, measurement assurance quality |

For every track, use matched task seeds, experiment and tool budgets, evidence access, action set and stop rules. Use task-family clustered bootstrap intervals and paired tests. Publish failures and consumed resources. Five seeds is the first minimum for synthetic families; real-lab workflows should use enough repeated campaigns to estimate variance, which may be substantially more than five.

## 7. Proposed SOTA gates

These are project targets, not external standards or achieved results:

- Exact agreement with each finite oracle on all analytic known-answer controls.
- At least 10% lower decision regret than the strongest matched EIG-only or fixed-funnel baseline on a preregistered decision family, with a paired confidence interval excluding zero and no critical safety regression.
- At least 10% lower held-out target-distribution prediction loss than EIG-only under a preregistered model-misspecification family, while remaining non-inferior in the well-specified regime.
- At least 10% lower resource cost to the same decision quality than the strongest comparable MFBO baseline on one multi-fidelity task, with all resource dimensions disclosed.
- Zero false causal claims on observationally equivalent negative controls, zero hard-constraint violations, and zero unauthorized state transitions in the frozen suite.
- Independent reproduction on a held-out family and source-pinned external baseline implementation before any SOTA claim.

## 8. Current implementation status

- SDCB v1: design-only 30-task pilot specification and reference harness in draft PR #13672.
- Evidence-gated belief updates: draft PR #13698.
- DSSI one-step EVSI and exact finite-horizon decision oracle: draft PR #13717.
- Robust/Pareto finite-scenario decision references: draft PR #13738.
- R-I/R-IDeA source-faithful score reference and separate linear proxy-training reference: draft PR #13741.
- Robust multi-fidelity decision/Pareto reference: draft PR #13759.
- Greedy R-I/R-IDeA design portfolio with sequential MMD² updates: draft PR #13760.
- Instrument-aware two-stage calibration/science design: draft PR #13762.
- Drift-aware Markov instrument-state and independent-device replication reference: draft PR #13793.
- Evidence-gated joint scientific/instrument belief transition: draft PR #13801.

All current additions have authored Rust tests, but the toolchain is unavailable here and the draft workflows are skipped. No benchmark tournament, physical experiment, or SOTA result has been completed.

## 9. Sources

- Petri-bench live Sweep v2: https://petri-labs.org/bench/
- Petri-bench v1 report: https://petri-labs.org/bench/report
- ActiveSciBench / LLM-AutoSciLab: https://scientific-discovery.github.io/llm-autoscilab-project/
- Autonomous Science Bench 0.1: https://asb.open-ssi.org/
- ASI-Bench: https://asibench.apexin.ai/
- AutoSciBench: https://arxiv.org/abs/2610.05140
- R-IDeA, AISTATS 2026: https://proceedings.mlr.press/v300/tang26d.html
- Multi-fidelity Bayesian optimization best practices, Nature Computational Science (2025): https://www.nature.com/articles/s43588-025-00822-9
- IJCAI 2026 adaptive experimental design: https://ijcai.org/proceedings/2026/904
- AI Index 2026: https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf
- NIST, Recommended Calibration Interval: https://www.nist.gov/calibrations/recommended-calibration-interval
- NIST, Metrological Traceability: https://www.nist.gov/metrology/metrological-traceability
- Bayesian Adaptive Calibration and Optimal Design, NeurIPS 2024: https://papers.neurips.cc/paper_files/paper/2024/hash/673b9f8cba4ee9f15b88656a69761631-Abstract-Conference.html
- Bayesian experimental design for model-discrepancy calibration, Journal of Computational Physics (2026): https://doi.org/10.1016/j.jcp.2025.114469