# PITB Causal Intervention Protocol (2026-10-24)

Status: research protocol proposal; not yet implemented or empirically validated.

## Purpose

PITB should measure whether a proof insight transfers for the right mathematical reason, not merely whether a prover retrieves a related proof. This protocol extends the frozen family split and leakage-candidate discovery work with a preregistered, evidence-bearing intervention suite.

Compilation alone does not establish that a benchmark statement is meaningful, non-vacuous, or the intended proposition.

## 1. Model relationships explicitly

Represent the benchmark as a graph with distinct node types: family, source artifact, formal statement, proof insight, and evidence record. Give every artifact an immutable path and content digest; give every statement its source provenance and exact proposition digest.

Use typed, evidence-bearing edges rather than a generic “related” label. Suggested types:

- same_artifact
- exact_statement_duplicate
- alpha_equivalent_statement
- imports_or_uses
- shared_lemma_dependency
- derived_from
- same_source_problem
- strategy_dependency
- conceptual_similarity
- unknown

Each edge records direction where meaningful, evidence references/digests, review status, reviewer, and rationale. A lexical candidate is not a dependency verdict. The policy layer—not the candidate generator—maps reviewed relation types to quarantine consequences.

Exact duplicates and confirmed source/dependency links are hard-quarantine candidates. Conceptual similarity remains an analysis signal. Unresolved cross-split links are explicitly reported and handled under a preregistered conservative policy.

## 2. Preserve the frozen split

1. Pin the family catalog, split assignment, and content digests.
2. Discover candidates against that pinned catalog.
3. Adjudicate candidates with source evidence, hiding test outcomes from reviewers where feasible.
4. Build connected components only from explicitly designated hard and unresolved-hard relationships.
5. Publish every component, split membership, evidence, and disposition.
6. Never silently move test items to improve sample size. If quarantine leaves too few independent clusters, report the confirmatory study as underpowered and create a new prospective benchmark version.

Reconcile the full catalog across eligible, quarantined, excluded, and unresolved/unknown dispositions. State whether categories are mutually exclusive. Never omit unknowns from denominators.

## 3. Counterfactual intervention suite

Every intervention must have a transformation receipt, original/result statement digests, an expected semantic effect, and an independent check. Script success alone is not proof that the intended transformation occurred.

| Intervention | Construction | Expected behavior | Failure signal |
|---|---|---|---|
| Necessary-premise deletion | Remove a premise required by the strategy, while keeping the task well-typed and discriminative | Reject the original strategy, abstain, or replan | False applicability |
| Irrelevant renaming | Alpha-rename bound variables and consistently rename irrelevant identifiers | Preserve strategy selection | Syntax dependence |
| Irrelevant-context deletion | Remove declarations independently shown to be unused | Avoid unnecessary abstention or loss of success | Context fragility |
| Incompatible decoy | Add a high-similarity example with materially different conditions | Select a valid insight or abstain | Similarity-driven false acceptance |
| Hypothesis strengthening | Add a premise that should not invalidate a valid strategy | Preserve applicability where justified | Excessive conservatism |
| Equivalent reformulation | Use independently checked equivalent statements | Transfer across representation changes | Formulation dependence |
| Dependency substitution | Replace a supporting lemma with a proved equivalent interface | Preserve interface-level reasoning | Hidden artifact dependence |
| Adversarial near miss | Change one material condition while keeping wording close | Detect the applicability boundary | Boundary blindness |

Invalid or ambiguous transformations are excluded from the primary score and counted separately. Do not assume a transformation is valid merely because it compiles.

## 4. Make insights executable contracts

An insight record should specify:
- strategy identifier/version and source provenance;
- typed preconditions over hypotheses, goal shape, structures, and available lemmas;
- intended intermediate obligations and expected proof-search actions;
- known counterconditions and abstention/replanning triggers;
- evidence supporting the applicability boundary;
- a calibrated applicability prediction, tuned on validation data only.

The evaluator distinguishes applicable, not_applicable, and unknown. Unknown is not silently counted as a correct rejection. Strategy selection and proof execution are separate outcomes.

## 5. Score separate dimensions

### Transfer efficacy
- Kernel-checked exact-target proof success.
- Incremental gain over the strongest validation-selected baseline under matched budgets.
- Family/component-clustered confidence intervals.
- Success per unit of wall-clock time and compute/token/tool budget where reliably measured.

### Selectivity and causal sensitivity
- False applicability rate on reviewed inapplicable variants.
- Missed applicability rate on reviewed applicable variants.
- Strategy-choice change under necessary-premise deletion.
- Invariance under irrelevant renaming and context deletion.
- Calibration (Brier score/reliability curves) and separate abstention coverage.
- Risk/coverage curves: error among answered cases versus coverage.

### Integrity and operations
- Candidate precision by adjudicated relation type.
- Reviewer agreement and adjudication coverage.
- Source provenance and dependency-evidence coverage.
- Quarantined, unresolved, and eligible family/component counts.
- Transformation validity and receipt completeness.
- Independently reviewed statement-defect rate.
- Prospective-task performance after model/checkpoint freeze.

Never collapse whole-proof pass rate, semantic transfer selectivity, formalization fidelity, real-project success, and cost into one headline score.

## 6. Required baselines and ablations

Use identical frozen tasks, proof environment, tools, timeouts, retry rules, and compute budgets.

Required baselines:
1. No-retrieval proof search.
2. Raw proof-fragment retrieval.
3. BM25 lexical retrieval.
4. Typed-AST/theorem-signature retrieval.
5. Import/dependency-graph retrieval.
6. Premise retrieval.
7. Strongest available learned retrieval baseline.
8. Applicability contracts without HDC.
9. HDC-enhanced retrieval/contracts.
10. Oracle relationship/contract labels as a diagnostic upper bound only.

Ablate retrieval representation, relationship graph, contract checker, counterfactual training, and abstention policy independently. HDC earns a place only if it adds reproducible value over strong non-HDC baselines at matched cost without degrading selectivity or integrity.

## 7. Statistical design and gates

Retain the research target of at least +10 percentage points in semantically qualified exact-target proof success. It is a target, not an assumed result. A confirmatory claim requires the family/component-clustered 95% confidence interval for treatment minus control to have a lower bound above zero and to meet a preregistered minimum-effect criterion.

At least 20 independent eligible test-family clusters is an operational floor, not a power guarantee. Before unblinding, simulate power using cluster sizes, baseline success, intracluster correlation, exclusions, and target effect. If power is inadequate, report an exploratory pilot and gather prospective families; do not weaken the gate after seeing results.

Preregister primary outcomes, baseline selection, exclusions, missing-data policy, multiplicity correction, stopping rule, and allowed tuning access. Tune on train/validation only. Seal test labels and receipts until the analysis plan is frozen.

## 8. Threat model and negative controls

Explicitly test exact/near duplicates, common upstream sources, shared lemma ancestry, equivalent propositions with different syntax, source-level dependence invisible to titles, vacuity and statement defects, model/prompt contamination, invalid transformations, permissive evaluator shortcuts, missing/stale receipts, and post-hoc baseline selection.

Negative controls should include high-similarity incompatible pairs, unrelated pairs sharing generic terms, a broken dependency edge, missing receipt, altered digest, invalid transformed statement, and a test-only hard link injected to ensure the audit detects cross-split dependencies. Each control must fail closed for the intended reason.

## 9. Reproducibility receipt

Every run should record repository commit and catalog digest; split and relationship-inventory digests; model/checkpoint, prompt, and retrieval-index digests; Lean/toolchain and dependency-lock identifiers; task/transformation IDs; budget, timeout, retries, and environment; per-task result, failure category, proof-check status, resource usage; evaluator version; and receipt schema version.

A receipt establishes what was run and recorded, not that the proposition is meaningful. Statement audits remain independent.

## 10. Execution sequence

**A — Integrity:** freeze catalog; validate inventory; adjudicate the existing 85 candidates plus new candidates with source provenance; publish component and uncertainty counts.

**B — Pilot:** build a small, manually reviewed intervention set across unrelated families; independently validate transformations; exercise negative controls and deterministic receipts.

**C — Baselines:** run retrieval baselines under matched budgets on train/validation; select the strongest baseline before test unblinding.

**D — Confirmatory evaluation:** proceed only after the integrity audit and cluster/power gate pass; publish denominators, exclusions, intervals, costs, and failures.

**E — Prospective track:** add real-project obligations after model freeze and retain immutable snapshots for reproducibility.

## Claim discipline

This document is a proposed protocol, not evidence of implemented tooling, validated transformations, improved proof success, or state-of-the-art performance. Report static inspection, schema parsing, compilation, tests, independent verification, and empirical results as separate statuses.
