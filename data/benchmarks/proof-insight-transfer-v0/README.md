# Proof Insight Transfer Benchmark v0

This directory defines a proposed benchmark format. **It currently contains no dataset and no benchmark results.**

## Files

- `insight-record.schema.json` — JSON Schema Draft 2020-12 for one provenance-linked insight record, including semantic-validation evidence and a constructive-solving track for unknown witnesses, proof obligations, and answer equivalence (RPE or formal/expert checking).
- `proof-progress-trace.schema.json` — strict trace contract for content-addressed proof-state snapshots, per-instance obligation lineage, resource counters, and terminal checker/semantic-gate receipts. Its synthetic fixture is validated against relational invariants in CI; splitting a goal is not counted as a verified closure, and this format is not a trace collector or a measured result.
- `counterfactual-precondition-suite.schema.json` plus `fixtures/counterfactual-precondition-suite.synthetic.json` — paired base/counterfactual targets with typed mutation class, source-artifact digest, separate base/variant artifact and statement digests, dual-review adjudications and outcome-row IDs. CI validates seven synthetic contrasts and rejects malformed/leakage/adjudication mutations. The fixture is contract-only, not mathematics or measured model behavior.
- `counterfactual-sensitivity-report.schema.json` — strict analysis-report contract for paired counterfactual metrics, component-cluster bootstrap intervals, mutation-class slices, and an always-unassessed confirmatory decision.
- `crates/bridges/symthaea-proof-insights/src/bin/analyze_counterfactual_sensitivity.rs` — deterministic Rust analyzer that joins suite contrast IDs to outcome JSONL, checks source artifact/insight digests and distinct base/variant target-artifact and statement digests, checks matched model/config/prompt/budget and per-reviewer evidence, and audits split/component leakage across the whole suite. It reports paired adherence, false acceptance, semantics-preserving stability, Brier score, and component-clustered 95% intervals; it never declares a confirmatory pass.
- `fixtures/counterfactual-outcomes.synthetic.jsonl` — 14 synthetic outcome rows for the seven-pair contract fixture. These rows exercise joins and report generation only and are not scored theorem-prover behavior.
- `transfer-selectivity-outcome.schema.json` — JSON Schema Draft 2020-12 for one source-insight/target-obligation compatibility outcome. It separates expert adjudication from system prediction, requires provenance and run receipts, records abstention and precondition checks, and supports false-accept/false-reject/recall/selective-risk analysis. It validates structure, not mathematical truth or family-disjointness.
- `crates/bridges/symthaea-proof-insights/src/bin/analyze_transfer_selectivity.rs` — deterministic Rust JSONL analyzer for the transfer-selectivity track. Before scoring, it independently checks the complete content-addressed source/target/adjudication/prediction/run envelope, required nullable probability fields, digest syntax, safe paths, declared budget caps, one requested split, unique pairs, dual-review consistency, and family-connected component IDs. It reports class-conditioned accept/reject/abstain rates, component-clustered bootstrap intervals, probability coverage, Brier score, and ten-bin reliability data. It never declares the overall confirmatory gate passed because the finite mutation-suite receipt is external to this binary.
- `transfer-selectivity-analysis-report.schema.json` — Draft 2020-12 schema for the Rust aggregate report, including raw denominators, component-clustered intervals, calibration coverage and reliability, and fail-closed confirmatory decision fields. The CI contract rejects reports that assert an overall pass without an independently evaluated mutation-suite receipt.
- `temporal-replay-manifest.schema.json` — JSON Schema Draft 2020-12 for the evaluator-side chronological repository replay manifest; it is a format/guardrail, not a generated Dalek dataset or a measured result.
- `temporal-timeline-audit.schema.json` — strict schema for the Rust-produced historical timeline audit, including README-vs-CSV count discrepancies, per-row source-file availability, adjacent completion-commit ancestry as explicit Git DAG evidence, and an explicit rejected/redacted status for unsafe machine-local paths. CSV source order is preserved; adjacent commits are not required to form a linear chain because parallel branches can be merged into the pinned head. Unsafe paths are not read or echoed in audit output.
- `temporal-dataset-join-audit.schema.json` — strict schema for both the unchanged path-only diagnostic and a second declaration-scoped theorem/proof transition join. The latter requires an exact declared-name match, proof text present within the named completion declaration but absent in every direct parent, an ancestral target-state commit, and the full typed statement prefix through the `:= by` boundary matching at both target time and completion. Multiple declarations sharing the requested short name produce `ambiguous_declaration_identity` plus their timeline positions; multiple proof-transition candidates have a separate ambiguity status. Ambiguous, missing, unsafe, or statement-drift cases remain explicit non-success states; the audit never opens or echoes machine-local paths.
- `splits.json` — frozen 372-family pre-screen assignment, pinned to OpenAI math commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`, with source fingerprints and a partition digest.
- `evaluation-protocol.md` — pre-registered baseline arms, primary endpoint, equal-budget rules, statistics, leakage controls, mutation gates, run-manifest contract, transfer-selectivity and counterfactual precondition-sensitivity tracks, proof-progress trace diagnostics, and the separate insight-component causal-ablation track.
- [`../../docs/research/proof-insight-transfer-sota-gap-analysis-2026-10-20.md`](../../../docs/research/proof-insight-transfer-sota-gap-analysis-2026-10-20.md) — research comparison with LeanSearch v2, T², TheoremBench, VeriSoftBench and First Proof, plus the 21 October AutoformBench-inspired infrastructure-construction and FormalRewardBench-inspired proof-critic challenge strata. These are distinct evaluation tracks, not comparable score claims.
- [`../../docs/research/proof-insight-transfer-temporal-challenge.md`](../../../docs/research/proof-insight-transfer-temporal-challenge.md) — separately reported chronological replay track for real repository verification, drawing on Dalek Lean Bench and VeriSoftBench; design only, no run or result claimed.
- `exposure-audit.md` — records a known exploratory inspection of test family `089`; that family remains in the immutable split manifest but must be excluded from confirmatory test denominators.
- `transfer-adjudications.md` — records two static cross-family negative controls and the typed compatibility/inequality-direction gate. They are curated annotations, not measured model failures.
- `candidates/156-borsuk-nine-finite-obstruction.json` — training-family strategy annotation plus a curator-rejected transfer to family 157, with machine-readable compatibility codes; not a scored record.
- `candidates/157-hadwiger-invariant-chain.json` — unreviewed strategy annotation for separating witness construction from an invariant-comparison bridge; not a scored record.
- `candidates/025-short-egyptian-greedy-residue.json` — unreviewed training-family strategy annotation for exact ceiling–residue decomposition, a strictly decreasing numerator, and a separate proof of denominator ordering; no transfer attempt or score is claimed.
- `candidates/026-eventual-subset-density-transfer.json` — unreviewed training-family strategy annotation for transporting positive linear counting bounds across eventual set inclusion by paying a finite-prefix constant, then deriving positive lower asymptotic density. It also records one static rejection against the constructive Egyptian-fraction existence target in train family 025; no generated transfer attempt or score is claimed. CI checks its pinned source digest, exact statement digest, train-only family assignment, schema, and directional-contract consistency.
- `temporal-replay-manifest.schema.json` and `temporal-target-bundle.schema.json` — evaluator-side historical visibility manifest plus the allowlisted solver-context bundle format. The Rust `--verify-temporal-manifest` command audits source history; `--materialize-temporal-target` emits approved visible artifacts; and `--verify-temporal-bundle` checks the bundle's payload digest, content digests, target statement binding, and structural timeline rules.
- [`../../docs/research/proof-insight-transfer-temporal-challenge.md`](../../../docs/research/proof-insight-transfer-temporal-challenge.md) — separately reported chronological repository-replay design and the manifest-validation command; no temporal task set has been generated or scored.
- [`../../docs/research/dalek-temporal-data-audit-2026-10-20.md`](../../../docs/research/dalek-temporal-data-audit-2026-10-20.md) — read-only audit of pinned Dalek timeline, dataset joins, historical-checkout behavior, and a concrete theorem-statement drift case; audit only, no model run.
- `pilot-report-2026-10-20.md` — pilot source audit, direct-transfer rejection, provenance notes, and next acceptance gates.
- `crates/bridges/symthaea-proof-insights/src/bin/validate_transfer_contract.rs` — fail-closed consistency checker for typed source/target obligations, inequality direction, domain/object-type compatibility, and target-side proof receipts; its rules do not establish mathematical truth.
- `crates/bridges/symthaea-proof-insights/src/main.rs` — `--verify-temporal-manifest` validates time-indexed artifact visibility, hidden-proof exclusion, pinned Git ancestry, full source-file digests, and exact source line-span digests. `--audit-temporal-timeline` and `--audit-temporal-dataset` reproduce the pinned Dalek source audits; the dataset audit now reports both path-only ambiguity and declaration-scoped proof-transition identity with target-time/completion statement checks. These remain textual metadata/provenance audits only; they do not claim Lean replay, semantic equivalence, or measured proof-insight transfer.
- Future `manifest.json` — indexed source artifacts and inclusion/exclusion/unknown accounting produced by the Rust indexer.
- Future `records.jsonl` — one schema-conformant JSON object per line, only after source evidence has been inspected and rights reviewed.
- The indexer emits its own `family_split_plan` and each family's `split`; compare them to `splits.json` with `--verify-splits` before creating any benchmark examples.

## Transfer-selectivity analysis

The analyzer consumes already-adjudicated outcomes; it does not adjudicate mathematical truth. Each input file must contain **only one selected split**. Example, after the isolated crate has been validated:

```bash
cargo run --locked --bin analyze_transfer_selectivity -- \
  data/benchmarks/proof-insight-transfer-v0/test-transfer-selectivity.jsonl \
  test \
  transfer-selectivity-report.json \
  10000 \
  20261021
```

Use `external_challenge` for a separately maintained external set. Do not mix train, validation, test, and external-challenge rows in one analysis input. The report hashes the exact input bytes, fixes the bootstrap seed and replicate count, resamples declared family-connected components, and emits 95% percentile intervals for compatible recall, incompatible false-accept rate, compatible false-reject rate, class-specific abstention rates, binary coverage, and selective risk.

`prediction.compatibility_probability` is distinct from action confidence: it means the estimated probability that the proposed transfer is compatible, regardless of whether the action is accept, reject, or abstain. The analyzer reports its scored coverage, Brier score, and ten equal-width reliability bins without imputing missing values. Probability metrics are descriptive until the rubric, calibration coverage, sample size, and independent cluster-level uncertainty are reviewed; missing calibration values do not count as a model failure or a success.

The analyzer deliberately emits `overall_confirmatory_decision = not_evaluated_...`: pair metrics alone cannot satisfy the preregistered finite mutation-suite gate or the primary exact-target proof-utility gate. The pair-track thresholds are diagnostics, not an overall SOTA or mathematical-correctness verdict.


## Counterfactual sensitivity analysis

The Rust `analyze_counterfactual_sensitivity` binary joins the suite's stable base/variant outcome IDs to the transfer-selectivity JSONL rows. For example, after both files have been checked against their JSON Schemas:

```bash
cargo run --locked --bin analyze_counterfactual_sensitivity -- \
  counterfactual-suite.json counterfactual-outcomes.jsonl test \
  counterfactual-sensitivity-report.json 10000 20261022
```

It checks source and target provenance, exact base/counterfactual statement digests, component IDs, labels/statuses, independent reviewer identities/evidence/reason codes, the budget caps and outcome-row uniqueness. The report separates compatible-base recall, counterfactual compatibility recall, sensitivity and false acceptance on incompatible variants, paired contract adherence, decision stability on semantics-preserving pairs, probability Brier score and the mean absolute probability shift for no-op variants. It reports per-mutation-kind sensitivity / false acceptance and component-bootstrap percentile intervals for the aggregate metrics.

The report calls rates `ratio` and Brier / probability-delta metrics `mean`; it includes hashes of exact suite and outcome bytes. It always emits an unassessed confirmatory decision. This is intentional: a well-formed synthetic join does not establish expert-adjudicated compatibility, checker-backed transfer utility, or a SOTA result. CI uses the seven-pair, fourteen-row synthetic fixture to exercise the end-to-end path and mutates missing-row, provenance, reviewer-evidence and over-budget cases. The primary exact-target proof-utility endpoint remains unchanged.

## Required discipline

1. Pin the upstream repository commit and each source file digest. A branch name like `main` is not a reproducible source identity.
2. Distinguish headline theorem, formalized supporting result, informal manuscript claim, and reasoning summary.
3. Cite a specific source path and locator for each strategy claim. Keep excerpts short; prefer locators and authored paraphrases.
4. A strategy record must state preconditions and at least one failure boundary or explicitly state that the boundary is unknown.
5. Keep candidate transfer separate from verified transfer. A transfer is verified only when the exact target statement is checked by the declared proof checker under a recorded axiom/dependency policy.
6. Use the indexer's fixed family split; never reshuffle after inspecting difficulty or results. Keep companion results or alternate proofs in their original family. Audit cross-family dependencies and pretraining exposure separately; a family split is necessary but not sufficient.
7. Add semantic statement checks: confirm the formal statement faithfully encodes the intended claim, then use dependent-successor tests where meaningful downstream declarations exist. Include mutation controls so tests are shown to detect deliberately incorrect target variants.
8. Separate theorem proving from constructive problem-solving: on the latter track, the answer/witness must be discovered and all associated conclusions discharged. Track typechecked witness, proved obligations, and answer equivalence separately; compiling a plausible candidate does not count as solved.
9. Keep failed, timed-out, out-of-scope, and not-attempted states distinct. No silent drops.
10. No HDC/embedding field is authoritative for truth. Vector indexes, if added, are derived cache data and must be ablatable.
11. The frozen split manifest was generated from the pinned catalog with an independent implementation of SHA-256 ranking. It has not yet been matched against a compiled Rust indexer output; run the crate and `--verify-splits` before relying on it as an engineering-verified artifact.
12. Follow `evaluation-protocol.md`: select settings and the strongest baseline among A–E and I on validation families only, freeze the primary contrast and compute budget before the test run, preserve all exclusions, and report paired family-level outcomes with uncertainty. Do not make SOTA claims from this pre-screen split alone.
13. Run the analyzer's `--mode power` using raw validation JSONL before held-out scoring. It recomputes baseline scores and eligible-family sizes, then simulates the primary gate over 0/+10/+15/+20pp effects and ICC 0/0.05/0.10/0.20. Require the registered +20pp/ICC 0.20 power and zero-effect false-claim gates; otherwise expand the external set or label the study a pilot.
14. After `--mode power` emits a passing report, hash the report and add its digest as `power_report_sha256` to the final frozen run config. Then pass that same file using `--power-report` to analyze mode; the analyzer independently reruns the validation-only scenarios before it will score test rows. See the [executable analysis contract](analysis-contract.md) for the exact two-stage commands and no-go behavior.

## No claims yet

Creating this schema or protocol does not establish that the records are complete, that the schema has been validated by an independent JSON Schema implementation, that proof insights transfer, or that Symthaea outperforms any baseline. Those require the stated validation gates and actual measured experiments; see also [the research design](../../../docs/research/proof-insight-transfer-2026.md).

15. After the protocol amendment, every future frozen run config must hash the current protocol bytes. The secondary transfer-selectivity outcome schema separates gold labels from model decisions; keep unresolved reviewer disagreements as `unknown`, report compatible-positive recall as well as incompatible false-accepts, and never interpret a structurally valid JSON row as a verified mathematical judgment.

16. For transfer selectivity, score only the preselected split using `analyze_transfer_selectivity`; report raw denominators, component counts, abstention, false acceptance, compatible recall, Brier score/reliability when available, and cluster-bootstrap intervals. The analyzer's overall decision is intentionally fail-closed pending an independent negative-control receipt and the primary proof-utility gate.
