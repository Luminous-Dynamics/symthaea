# EVAL-JUDGE-METROLOGY-001 — Evaluate the evaluator

**Status:** protocol and artifact-format proposal; example fixtures are non-scoring. No real judge, Symthaea model, or industry baseline has been measured here.

**Tracking:** [Issue #9022](https://github.com/Luminous-Dynamics/symthaea/issues/9022). This work composes with the evidence behavior contracts in [PR #9005](https://github.com/Luminous-Dynamics/symthaea/pull/9005) without treating their automated adjudicator as ground truth.

## 1. Paradigm shift: a judge is a measurement instrument

Agreement and repeatability are not the same as validity. A deterministic judge that gives the wrong answer every time is repeatable but useless. A judge may have high average accuracy while exhibiting severe position bias, prompt fragility, or larger failures on adversarial items.

Recent work makes these failure modes concrete:

- **AgentJudgeBench (EMNLP 2026)** examines judge reliability on tool-calling workflows. In its study, alignment declined with difficulty; on hard instances without ground-truth exposure, judges clustered at 77–82% alignment, and rubric mitigations did not generalize uniformly across generator–judge pairs: https://aclanthology.org/2026.emnlp-main.659/
- **Diagnosing the Reliability of LLM-as-a-Judge via IRT (ICML 2026)** distinguishes intrinsic consistency under prompt variations from alignment with human judgments and uses an item-response model to make reliability interpretable: https://proceedings.mlr.press/v306/choi26n.html
- **Who can we trust? LLM-as-a-jury (ICML 2026)** models judge-specific reliability instead of assuming all judges are equally accurate: https://proceedings.mlr.press/v306/qian26i.html
- **All Verdicts are Not Equal (October 2026 preprint)** evaluates accuracy jointly with repeatability and order invariance, finding that deterministic verdicts can still be wrong: https://arxiv.org/abs/2610.12083
- **NIST AI 800-3 (February 2026)** distinguishes fixed-benchmark performance from generalized accuracy and recommends explicit statistical assumptions and variance decomposition: https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

These reported numbers are tied to the papers' own datasets and methods. They are not measurements of Symthaea or claims that the same rates hold for our benchmark.

## 2. Frozen audit protocol

Before trials, commit the exact identities and digests for judge provider/model/revision, API version, prompt, rubric, decoding/settings, reference-label source and protocol, perturbation policy, scoring rules, case set, and replicate schedule. The full evaluator plan contains reference labels and must remain evaluator-only; never expose gold labels or the reference adjudication evidence to the judge before all verdicts are sealed.

Every planned trial is a predeclared item × perturbation × replicate cell. At minimum, include:

- **Baseline:** canonical prompt and canonical candidate/evidence presentation.
- **Prompt variant:** a semantically equivalent prompt/rubric rendering, declared before scoring.
- **Order swap:** reverse A/B presentation order while preserving semantic content.
- **Repeat:** repeat the same configuration to detect stochastic or infrastructure instability.

These perturbations are distinct. Do not conflate presentation-order flips with prompt-variant flips, and do not assume repeated variants of the same item are independent statistical items.

## 3. Trial accounting and statuses

Each planned trial receives exactly one terminal record. Statuses are completed, abstained, error, missing, and invalid_receipt. Only completed binary verdicts enter conditional accuracy. All assigned trials remain in the coverage denominator; missing, error and invalid records must not disappear. An abstention is neither a pass nor a fail, but abstention rate and quality must be reported separately.

The Rust scorecard implements baseline Brier score (lower is better) and fixed 10-bin expected calibration error, with null metrics when no confidence-labelled baseline trials are available. The binning rule is fixed in this protocol version and is not selected after observing results.

The machine-readable contract includes:

- **schema/judge-reliability-plan.schema.json** — evaluator-only plan and frozen perturbation grid.
- **schema/judge-reliability-report.schema.json** — terminal trial receipts.
- **fixtures/judge-reliability-plan.example.json** — four synthetic item profiles across difficulty bands and three invariance interventions.
- **fixtures/judge-reliability-report.example.json** — deterministic fixture outputs, including a consistently wrong judge, a position-biased case, and a prompt-fragile case.

The example files contain zero digest placeholders and a NOT_IMPLEMENTED_YET marker; they are schema/ontology fixtures only and must fail qualified-run admission.

## 4. Report these metrics separately

1. **Verdict accuracy** on completed binary verdicts versus the frozen reference label; include precision, recall, specificity, balanced accuracy and per-label counts when estimable.
2. **Coverage:** completed binary verdicts divided by all planned trials. Report abstain, error, missing and invalid-receipt rates separately; also show conditional accuracy so coverage is not hidden.
3. **Prompt flip rate:** mismatches between baseline and prompt-variant verdicts among completed comparable pairs; report planned comparisons, completed comparisons and flips.
4. **Position flip rate:** mismatches between baseline and reversed-order verdicts, separately from prompt variation.
5. **Repeat disagreement rate:** mismatches between baseline and same-configuration repeats. Repeatability is not validity.
6. **Trusted-verdict rate:** proportion of reference-evaluable planned items whose baseline verdict is correct and whose every predeclared invariance trial completes with the same correct verdict. Missing or abstaining perturbations do not count as trusted. A deterministic-but-wrong judge therefore scores zero even if it repeats the same verdict perfectly.
7. **Stratified reliability** by contract family, difficulty, perturbation kind and reference source; include uncertainty intervals and label provenance.
8. **Calibration:** confidence means the judge's estimated probability that its verdict is correct. The Rust scorecard reports Brier score and expected calibration error over fixed 10 equal-width confidence bins (0.0–0.1 through 0.9–1.0), using completed baseline verdicts with evaluator-reference labels and recorded confidence. It reports the confidence denominator explicitly; self-reported confidence is not calibrated by definition.
9. **Human alignment:** use blinded reviewers, freeze the audit sample, record disagreements and adjudication provenance, and publish judge-to-human agreement.

Every rate must include its numerator and denominator. Undefined estimates remain null, not zero. Do not collapse critical dimensions into one ranking. A composite can only be secondary, preweighted, and accompanied by per-family and per-perturbation results.

## 5. Initial acceptance targets

- Reject 100% of deterministic malformed, duplicate, missing, incompatible and post-hoc mutation controls in the committed synthetic test suite.
- Score 100% correctly the enumerated synthetic controls: stable-correct, consistently-wrong-but-stable, position-biased, prompt-fragile, unstable-repeat, abstaining, and missing/error cases.
- The consistently-wrong-but-stable judge must not be classified as trustworthy.
- Keep all planned denominators and report every missing or invalid trial.
- Before any real evaluator or Symthaea superiority claim, complete a blind reference-label audit and matched-budget held-out comparison.

These are qualification targets for enumerated fixtures—not achieved metrics and not targets for open-world accuracy.

## 6. Analysis model and external comparison

Start with exact finite-sample descriptive metrics on a frozen set. If the intended claim generalizes beyond that set, predeclare the target population and use an appropriate hierarchical/GLMM or item-response model only with adequate sampling, diagnostics, and stated assumptions. Estimate item difficulty and judge-specific effects where supported; do not infer broad capability from correlated perturbation copies of a handful of cases.

Compare judges under the same items, source snapshots, rubric semantics, output schema, evidence exposure and budget. Publish per-judge accuracy, balanced accuracy, prompt flips, position flips, repeat disagreement, trusted-verdict rate, calibration and call cost. Compare against blinded human/adjudicated references and deterministic programmatic oracles where available. Neither a panel vote nor an LLM judge alone constitutes gold truth.

Keep benchmark accuracy on the fixed item set separate from generalized accuracy, as emphasized by NIST AI 800-3. Report source version, model version, prompt/rubric digests, scoring-code revision and exact assigned denominators for every comparison.

## 7. Claim boundary and next implementation step

This document and fixture set define a preregistered reliability audit contract. The next step is a Rust validator/CLI that verifies exact plan/report commitments, one record per planned trial, perturbation-grid completeness, judge/revision binding, and deterministic scorecards; hostile fixtures must prove tampering and missing records fail closed. An independent blind human audit is required before judge labels can be promoted into the evidence-behavior benchmark.

A valid audit receipt means the accounting and declared protocol verify. It does not prove judge identity, execution, evaluator independence, semantic truth, safety or generalization.
