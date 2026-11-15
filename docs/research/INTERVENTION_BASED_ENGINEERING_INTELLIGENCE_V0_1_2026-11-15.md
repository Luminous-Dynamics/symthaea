# EGDDM D4/D5 — intervention-based change impact and causal engineering intelligence v0.1

**Status:** research protocol and issue specification. No intervention corpus, model result, causal estimate or benchmark score has been produced.

## 1. Separate impact prediction from causal attribution

Two questions are frequently collapsed:

- **Change-impact prediction:** which components, tests, contracts, obligations or deployments are likely to be affected by a proposed change?
- **Causal attribution:** did the proposed change actually cause a specific observed outcome, relative to what would have happened without it?

A dependency graph can be useful for the first question. It is not sufficient evidence for the second. Co-change, common ownership, call edges, temporal precedence and shared configuration are candidate predictors or confounders, not causal proof.

D4 should measure affected-scope prediction. D5 should measure intervention-tested causal claims. Publish separate scorecards.

## 2. Intervention record

Every causal claim should be represented by a preregistered intervention record:

- exact repository commit/tree and relevant blob identities;
- configuration-world digest and execution image;
- treatment unit and intervention patch/diff digest;
- comparison/control condition and why it is comparable;
- baseline observation window and frozen baseline metrics;
- predicted affected components and predicted effect intervals;
- primary/secondary outcomes and measurement method;
- confounders, exclusion rules and stopping conditions;
- sandbox/runner identity and isolation evidence;
- post-intervention observations and their independent receipts;
- estimated effect, uncertainty interval, and residual limitations.

The record is a hypothesis and experiment plan until independent observations arrive. A predicted outcome must never be written as an observed fact.

## 3. Experimental design

Prefer randomized assignment of eligible changes or components to treatment and control where operationally safe. For non-randomized engineering changes, use preregistered matched controls, interrupted time-series or difference-in-differences only when their assumptions are documented and tested.

For a simple matched control, estimate:

\[
\widehat{\tau}=
(Y_{T,after}-Y_{T,before})-
(Y_{C,after}-Y_{C,before}).
\]

This difference-in-differences estimate is not automatically causal. Parallel-trend assumptions, concurrent changes, version drift, spillover between components, selection effects and measurement changes must be tested or disclosed.

Use at least four control types:
1. **Positive control:** known intervention that should change a specified outcome.
2. **Negative control:** intervention or outcome that should not be causally affected.
3. **Sham control:** similar procedural activity without the causal treatment.
4. **Confounder control:** known co-change or environment shift that should not be attributed to the candidate intervention.

All repository-controlled execution must occur in an independently administered disposable sandbox. The EGDDM parser/validator remains read-only and must not execute repository code.

## 4. Benchmark tasks

Construct task families from:
- synthetic controlled dependency/contract mutations;
- feature-flag and configuration changes;
- API/schema version changes with multiple consumers;
- build/test toolchain upgrades;
- rollback and remediation episodes;
- real historical changes with a known intervention and independently observed downstream result.

Hold out repository families and forward time windows. The candidate sees only the pre-intervention snapshot, admissible history and preregistered task context. Outcomes, treatment labels and gold affected regions remain hidden until grading.

## 5. Baselines

Compare:
- direct dependency traversal;
- lexical and symbol-level change localization;
- historical co-change prediction;
- temporal precedence only;
- static graph plus configuration-aware traversal;
- graph plus change history without intervention data;
- causal model without negative controls;
- full EGDDM with preregistered intervention records;
- expert-human predictions.

Use identical source snapshots, context budgets, configuration worlds and task cutoffs. Report D4 impact prediction separately from D5 causal attribution.

## 6. Metrics

D4:
- affected-region precision/recall/F1 at file, symbol, test and obligation granularity;
- critical-miss rate;
- ranking quality and fixed-budget coverage;
- change-impact calibration and abstention quality.

D5:
- intervention-effect estimation error;
- causal edge precision/recall on adjudicated intervention graphs;
- prediction interval coverage and width;
- calibration of effect magnitude and direction;
- false causal attribution under negative controls;
- causal abstention precision;
- treatment/control validity and residual confounder disclosure.

D7 should later test whether recommendations based on D4/D5 predictions improve real project outcomes.

## 7. Prospective targets

These are research goals, not achieved measurements:
- D4: ≥95% recall and ≥80% precision on held-out affected obligations, with zero missed critical controls in the planted suite.
- D5: beat dependency-only and co-change baselines on held-out intervention prediction, with calibrated uncertainty intervals.
- D5: zero false causal attributions on the explicit negative-control suite.
- D5: ≥90% coverage of preregistered prediction intervals on a sufficiently powered held-out set.
- Abstention: correctly leave causal status unknown when intervention evidence is absent or confounded.

Publish confidence intervals and exact denominators. A zero-error result on a small synthetic set is not a zero-risk guarantee.

## 8. Research rule

The system should be rewarded for discovering that the available evidence cannot distinguish two competing explanations. It should not be rewarded for producing a complete-looking causal graph. Unsupported causal claims are worse than an explicit unknown.
