# Model-Adequacy-Aware Experiment Design v0.1

**Status:** research protocol proposal, not an implementation or performance claim. This document extends `EXPECTED_INFORMATION_GAIN_SELECTOR_V0_1.md`; it does not replace the variance or EIG baselines.

## Executive hypothesis

A scientific discovery system should optimize for **valid knowledge, not just separation among the hypotheses it already has**. Expected information gain (EIG) is a useful criterion when the hypothesis space, likelihood, and priors are adequate. Under misspecification, a design can efficiently eliminate alternatives and leave a confidently wrong singleton.

The proposed next step is to benchmark a model-adequacy-aware discovery loop that can (1) discriminate among candidate laws, (2) detect evidence that the entire candidate set is inadequate, (3) request or construct a discrepancy/model-expansion path, and (4) validate any eventual singleton on measurements selected independently of the final fitting decision.

This is not a claim of a new theoretically optimal acquisition function. It is a falsifiable engineering/research agenda informed by work on experiment design under model misspecification. Published methods such as EGIG, R-IDeA, and maximin robust Bayesian experimental design must be cited and implemented faithfully before being called baselines; the simple scheduled-audit policy in PR #8326 is not a reimplementation of any of them.

## Why the existing audit pilot is not the finish line

PR #8326 establishes a useful starting point: plain EIG and an audit-augmented policy, a controlled sinusoidal discrepancy, shared seeds, and separate reporting for empty candidate sets, singleton outcomes, and unresolved sets. Four limitations need to be made explicit in the next phase:

1. **An empty candidate set is a symptom, not a diagnosis.** It may indicate model misspecification, but can also be caused by noisy observations, a bad assumed noise scale, an overconfident likelihood, an invalid numerical prediction, or a candidate-generation/implementation defect. Report which diagnostic triggered rejection and whether the rejection rule was calibrated on clean worlds.
2. **A singleton is only as meaningful as the prior and candidate representation.** If a law appears as many syntactic expressions, a uniform prior over expressions gives that family more probability mass. Algebraic deduplication alone cannot establish general semantic equivalence.
3. **Periodic uniform audits do not automatically guarantee representative coverage.** They can fail to sample high-risk regions, cost too much, or still be evaluated under selection-biased residuals. Compare them to targeted adequacy probes and keep audit-selection probabilities explicit.
4. **Fit and adequacy have different targets.** Better fit on queried points does not establish structural recovery, calibrated predictions, or safe generalization to a deployment/query distribution.

## Proposed state representation

Keep two logically distinct uncertainties visible:

- **Within-space uncertainty:** which candidate structural family/law/parameterization best explains the observations?
- **Adequacy uncertainty:** is the modeled candidate space sufficient for the observation process over the declared operating domain?

For research experiments, define an augmented model state (Z) whose members include both ordinary candidate laws and explicit alternatives from a predeclared discrepancy family. Examples of discrepancy alternatives include additive smooth residuals, a missing interaction term, a regime/change-point model, heteroscedastic observation noise, and a low-complexity shifted response. The hidden data-generating truth and its discrepancy amplitude must never be passed to the selector.

An expanded-state design criterion can ask how much a measurement tells us about (Z), rather than only the identity of a candidate inside the original set:

```text
A_aug(x) = I(Z; Y | x, observed_history)
```

This is a *research arm*, not a universal solution. It remains vulnerable when the discrepancy family is itself wrong, when priors are poorly calibrated, or when likelihood assumptions fail. Compare it against a model-criticism/expansion policy and robust-design baselines rather than assuming that including a discrepancy term solves misspecification.

### Prior and representation controls

- Give explicit prior mass to structural families first, then distribute within-family mass across parameterizations or expressions. Preserve a legacy uniform-over-candidates arm for ablation.
- Do not let duplicated syntactic candidates silently change total family prior mass. Include a candidate-duplication metamorphic test: adding a duplicate representation should not materially change a family-level selection decision when the intended prior is family-balanced.
- Record the equivalence policy. Exact syntax normalization and known algebraic rewrites are safe, bounded conveniences; finite probe agreement is not a proof of global equivalence.
- Report both family recovery and exact-expression recovery. They answer different questions.
- If posterior family weights cannot be justified, describe them as assumed weights and run prior-sensitivity stress tests.

## Loop design: discriminate, criticize, expand, validate

1. **Discriminate.** Use the current selector (variance, EIG, or cost-aware EIG) for the within-space question.
2. **Criticize.** Track prequential predictive log score, standardized residuals, predictive interval coverage, and residual structure. The simulator may use hidden truth for evaluator-only scoring, but the selection policy may not.
3. **Expand or abstain.** On a calibrated adequacy alarm, do not silently pick the least-wrong survivor. Preserve the evidence and either (a) generate a declared new candidate family, (b) fit a bounded discrepancy model, (c) request another measurement aimed at distinguishing candidate explanations from discrepancy alternatives, or (d) return `InadequateModelSpace` / `Unresolved`.
4. **Validate.** A singleton remains provisional until it survives a predeclared validation budget on audit measurements not reused to tune that candidate. Validation ends in one of: supported recovery, contradicted candidate, detected model-space failure, or unresolved. Do not convert missing budget into a positive validation result.

### Preventing selection-biased adequacy checks

Adaptive experiment selection changes the distribution of observed inputs. Residuals on those selected inputs are not automatically representative of the intended operating domain. Keep at least one of these controls in every confirmatory design:

- a held-out audit stream drawn from the target query distribution;
- randomized audits with recorded, nonzero selection propensities and an explicitly justified correction; or
- a deployment-distribution simulator where coverage and error are evaluated on fresh, independent inputs.

Use the first option when feasible. Do not claim calibrated predictive coverage under arbitrary covariate shift from ordinary residual plots alone. If the target distribution is unknown, report performance by query region and make the limitation explicit.

## Acquisition strategies to compare

All selectors receive identical histories, admissible experiments, candidate-pool versions, cost metadata, and observation budgets. The evaluator-only oracle receives hidden truth and is never visible to a selector.

| ID | Policy | Purpose |
| --- | --- | --- |
| B0 | Seeded uniform admissible experiment | Exploration baseline |
| B1 | Raw prediction variance | Existing disagreement baseline |
| B2 | Plain EIG | Information-theoretic in-space baseline |
| B3 | EIG per cost | Cost-aware in-space baseline |
| B4 | EIG plus scheduled uniform audits | Existing PR #8326 policy |
| R1 | Adequacy-triggered audit / model-expansion policy | Test whether explicit criticism and recovery help |
| R2 | EIG over candidate + predeclared discrepancy states | Test whether model-space uncertainty changes useful choices |
| R3 | Published robust acquisition method, only after faithful implementation | Literature baseline; report method/version and assumptions |
| O0 | Evaluator-only oracle | Upper bound/diagnostic only, never an eligible competitor |

The first implementation should not blend every criterion into one opaque weighted score. Prefer a **constrained/Pareto comparison**: first discard unsafe or inadmissible experiments; then compare information, expected adequacy-test power, target-distribution representativeness, and cost as separate dimensions. If a scalar acquisition is later tested, predeclare normalization and weights on training worlds only, freeze them, and report the underlying component scores.

## Benchmark matrix

Use controlled generator families so that the system can be scored against a known truth without leaking it into policy inputs. Separate pilot debugging from confirmatory evaluation.

### Worlds and perturbations

- At least 60 world configurations spanning at least six held-out structural families or transformation mechanisms; hold out complete generator families/mechanisms from tuning when practical.
- At least 20 paired seeds per world and policy for the confirmatory synthetic stage. Treat seeds as nested within worlds, not independent world-level evidence.
- Observation noise: low, moderate, and signal-dominated; Gaussian baseline plus heavy-tailed and heteroscedastic conditions.
- Misspecification magnitude: `0, 0.03, 0.06, 0.12, 0.24` in truth-scaled units, with a frozen definition of scale and no truth leakage to selectors.
- Distribution shifts: target-query covariate shift, unqueried-domain extrapolation, changed parameter regimes, and a change point.
- Candidate-space failures: omitted interaction, omitted law family, near-equivalent candidates, duplicate expressions, underdetermined laws, irrelevant candidates, and no candidate within tolerance.
- Measurement/process faults: unequal or misspecified costs, invalid predictions, non-finite values, correlated measurements, and an observation instrument with biased noise.
- Geometry/decision edge cases: balanced hypothesis partitions versus rare outliers, boundary-only separation, undefined regions, and high-information experiments that lie outside the target operating distribution.

Predeclare which dimensions are screening-only and which are confirmatory. Do not multiply a small number of seeds across many nearly identical configurations and call that broad generalization.

### Outcome taxonomy

Every run must end in exactly one terminal category and include its evidence trace:

1. **Supported recovery:** the family and law satisfy the benchmark's predeclared structural and held-out predictive criteria.
2. **False singleton:** one candidate is accepted although the hidden world is outside that candidate's allowed equivalence/tolerance class.
3. **Detected model-space failure:** a calibrated test triggers and the policy explicitly reports inadequacy or requests/attempts expansion.
4. **Unresolved:** multiple plausible candidates remain after budget exhaustion.
5. **Invalid/failed run:** an implementation, numerical, cost, or protocol error prevents a scientifically interpretable result.

An empty candidate set is not by itself category 3. It is category 3 only if a valid, calibrated adequacy rule supports that interpretation; otherwise classify it as invalid/failed or unresolved according to the predeclared protocol.

## Metrics and advancement targets

Never collapse these into a single unqualified headline score.

| Metric | Proposed target (not result) |
| --- | --- |
| False-singleton rate on held-out misspecified worlds | At least 20% relative reduction versus plain EIG, with a paired, world-clustered confidence interval supporting a reduction |
| Adequacy detection | At least 80% detection sensitivity at no more than 5% false-alarm rate on clean worlds, after calibration and before confirmatory evaluation |
| Clean-world structural recovery | No more than 5 percentage-point loss versus plain EIG at equal observation budget |
| Held-out predictive coverage / error | Report by regime and input region; no claim of calibrated coverage unless the evaluation design supports it |
| Cost to supported recovery | Median and 90th percentile; include audits, expansion steps, and failed measurements in actual cost |
| Scientific validity | Independent validity rubric and invalid-result rate; no improvement claim from fit alone |
| False-alarm / abstention burden | Report how often the policy abstains, expands, or declares inadequacy on clean worlds |
| Transfer under shift | Held-out error and recovery over fresh target-distribution inputs |
| Runtime and overhead | Selection latency, score-evaluation count, and total wall time, separately from experiment cost |
| Reproducibility | Frozen manifest, seed schedule, toolchain, source commit, data/generator digests, and machine-readable per-run receipts |

The two primary confirmatory contrasts should be frozen before the run: (a) false-singleton rate of R1/R2 versus B2 on misspecified worlds; and (b) supported-recovery rate of R1/R2 versus B2 on clean worlds. Use a paired hierarchical bootstrap that resamples worlds first and seeds within sampled worlds. Report absolute risk differences as well as relative reduction. For multiple primary policy comparisons, predeclare a multiplicity correction or narrow the primary hypothesis to one candidate policy.

The thresholds above are advancement goals, not universal constants. Calibrate the adequacy alarm and freeze its thresholds using training/validation worlds; use held-out worlds only for confirmation. A policy must not win solely by converting false singletons into unlimited abstentions: pair false-singleton reduction with detection sensitivity, clean-world recovery, and cost-to-supported-recovery constraints.

## Required adversarial and metamorphic tests

- **Candidate duplication:** replicate one expression under a new ID. Under family-level priors, verify that the total family mass/selection ranking does not change beyond numerical tolerance.
- **Permutation invariance:** reorder candidate arrays and verify scores/decisions are invariant except explicitly documented deterministic tie-breaking.
- **No-oracle leakage:** perturb the hidden truth while keeping all model-visible state identical; the selector's decision for that step must remain identical.
- **Empty-set semantics:** inject contradictions, invalid numeric outputs, and extreme but valid residuals separately; verify they produce distinct typed outcomes.
- **Audit independence:** ensure validation observations are not reused to optimize the candidate they validate; verify audit RNG and selection propensities are captured.
- **Target-shift split:** show that an acquisition optimized on training-query inputs is not evaluated solely on those same inputs.
- **Noise miscalibration:** overstate and understate the assumed `sigma`; chart false singleton, adequacy detection, and recovery versus noise misspecification.
- **Cost scaling:** multiply every cost by the same positive constant; cost-normalized ranking should remain stable. Change one candidate's cost and verify that only documented cost-dependent decisions change.
- **Budget monotonicity diagnostics:** under a fixed world and seed schedule, report rather than assume monotonic outcomes as budget increases; an adaptive decision rule can change paths and invalidate naive monotonicity assumptions.
- **Numerical convergence:** retain the current 256-vs-2,048 quadrature comparison, and supplement it with analytical/special-case tests and independent calculation where feasible. Agreement with a refined quadrature is not a rigorous bound on quadrature error.

## Comparison to current research and industry

Relevant external references differ in objective and evaluation protocol:

- Callahan & Catanach, AISTATS 2026, *On the Misinformation in a Statistical Experiment*: analyzes how misspecified inference can make classical information-maximizing designs amplify bias; develops criteria with inferential-error/model-error concerns. This is the motivation for evaluating adequacy-aware alternatives, not evidence that our simple audit policy is equivalent.
- Tang, Sloman & Kaski, AISTATS 2026, *Representative, Informative, and De-Amplifying*: proposes R-IDeA to account for representativeness, informativeness, and error amplification under misspecification. Treat it as a distinct method that needs a faithful reproduction/adapter before using it as a benchmark arm.
- Abdulsamad et al., 2026, *Maximin Robust Bayesian Experimental Design*: formulates a max-min robust-design problem with information-theoretic constraints and a Sibson-mutual-information objective; its estimator and assumptions need separate study before an implementation claim.
- NIST's Sequential Bayesian Experiment Design / `optbayesexpt` is a practical reference for adaptive laboratory parameter estimation and cost/resource-aware sequential measurement. It is adjacent engineering practice, not a law-discovery SOTA score.
- NewtonBench reports 324 interactive scientific-law tasks across 12 physics domains and explicitly varies law/system complexity; it is relevant to generalizable interactive discovery.
- SciLaws-Bench reports separate numerical fit, scientific validity, and structure-recovery metrics. Its task data retains per-task upstream license requirements.

**External-score discipline:** do not copy a live leaderboard value into a benchmark receipt without a dated snapshot, source URL, exact table/split, task version, evaluator, and digest. The official SciLaws-Bench README and dynamic project page can show different leaderboard snapshots as results change. Pin a source commit or archived table and record the access date. Do not compare scores from different subsets, tool access, budgets, judges, or task licenses as if they were one common competition. The public external suites are the transfer stage, not a substitute for controlled internal misspecification tests.

References:
- https://proceedings.mlr.press/v300/callahan26a.html
- https://proceedings.mlr.press/v300/tang26d.html
- https://arxiv.org/abs/2603.14094
- https://www.nist.gov/programs-projects/sequential-bayesian-experiment-design
- https://github.com/HKUST-KnowComp/NewtonBench
- https://github.com/yiyihum/SciLaws-Bench

## Implementation sequence

**P0 — Protocol and outcome integrity (this document).** Freeze the taxonomy, primary contrasts, leakage controls, and manifest fields. Do not change current EIG behavior.

**P1 — Simulator instrumentation.** Make each policy emit a typed terminal outcome, model-visible inputs, selected query, cost, audit flag, and reason for termination. Include the hidden truth only in evaluator-side records.

**P2 — Adequacy-triggered baseline.** Add prequential residual diagnostics and a validation budget using an independent/randomized audit stream. Calibrate on clean training worlds; do not let a diagnostic's threshold be tuned against the confirmatory worlds.

**P3 — Discrepancy-state arm.** Add a small, frozen set of candidate discrepancy generators and evaluate augmented-state EIG against plain EIG and the P2 policy. Make the set/priors explicit and include an omitted-discrepancy stress condition.

**P4 — Confirmatory benchmark.** Run the frozen synthetic matrix with exact-head receipts and world-clustered intervals. Keep failures, invalid runs, and incomplete jobs visible; never treat queued work as evidence.

**P5 — External adapter.** After passing synthetic gates, select permitted and version-pinned tasks from NewtonBench and/or SciLaws-Bench. Respect task-level licenses and preserve original evaluation settings. Report external results separately.

## Minimum run receipt

Each run should record:

- protocol version and benchmark-manifest SHA-256;
- source commit and dirty-tree state;
- toolchain/build profile and simulator version;
- world-generator family/version and opaque world ID;
- policy version, prior mode, discrepancy family, and all policy-visible settings;
- seed IDs and RNG algorithm;
- noise/discrepancy regime (evaluator side); do not expose hidden values to the policy;
- ordered experiment requests, model-visible predictions, measured outcomes, cost, and audit/validation split;
- terminal category, structural recovery, validation score, adequacy decision, and failure reason;
- bootstrap seed, aggregation code/version, and raw per-run artifact digest.

A leaderboard number without these fields is a result that cannot yet be independently audited.

## Claim boundary

Until P4 completes, describe MAED as a proposed benchmark and research protocol. Do not claim that it outperforms EIG, R-IDeA, maximin robust design, industrial scientific ML, or human researchers. The immediate win is operational: Symthaea should be able to tell the difference between **“I found a survivor,” “I validated a law,” and “my model space failed.”**
