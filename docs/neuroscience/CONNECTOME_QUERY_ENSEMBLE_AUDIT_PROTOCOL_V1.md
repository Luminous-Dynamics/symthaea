# Finite reconstruction query robustness audit V1

**Status:** first Rust evaluator, JSON Schema and synthetic fixture committed; no Rust compilation or biological result claimed.  
**Tracking:** [#9344 — Query robustness across plausible connectome reconstructions](https://github.com/Luminous-Dynamics/symthaea/issues/9344).  
**Tool:** `connectome-query-ensemble-audit`.

## Research question

Does a declared scientific-query decision remain stable when evaluated on multiple explicit, provenance-bound plausible connectome reconstructions?

A single maximum-score/MAP graph can conceal uncertainty in cell identity, neurite continuity, synapse attachment and correction choices. This evaluator runs the same query contract on every graph in a finite scenario set and reports the range and decision stability.

V1 is intentionally a **finite unweighted scenario evaluator**. It does not estimate a posterior, confidence interval, probability of biological truth, or probability that the scenario set contains the true reconstruction.

## Run

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-ensemble-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_QUERY_ENSEMBLE_AUDIT.example.json \
  --output query-ensemble-report.json

cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-query-ensemble-audit
```

Input schema: `CONNECTOME_QUERY_ENSEMBLE_AUDIT.schema.json`. The fixture is synthetic and deliberately contains one reconstruction-sensitive path query plus two decision-stable control queries.

## Input and provenance contract

The input binds:
- analysis ID and exact dataset revision commit SHA;
- scope declaration (`exhaustive_declared_scenario_space` or `sampled_sensitivity_set`);
- at least two uniquely identified reconstruction scenarios;
- canonical SHA-256 for each directed weighted graph;
- one or more lineage group IDs per scenario, to preserve shared evidence ancestry;
- stable query IDs, supported graph metrics and preregistered decision rules.

Graph digests use SHA-256 over a domain separator followed by sorted directed edges encoded as big-endian source/target u64 and the big-endian IEEE-754 weight bit pattern. Duplicate directed edges, non-positive/non-finite weights, malformed digests and graph-digest mismatches fail closed.

Scenario results are sorted by scenario ID. The canonical scenario-set digest is independent of input scenario ordering and binds scenario IDs, graph digests and sorted lineage group IDs.

Lineage group IDs are metadata about shared ancestry, not evidence that scenarios are independent. The evaluator makes no independence assumption and does not multiply scenario counts or feature confidences.

## Query language V1

Supported metrics:
- `edge_weight(source,target)`;
- weighted `in_degree(node)`;
- weighted `out_degree(node)`;
- directed `path_exists(source,target)`.

Supported decisions:
- `boolean_is` for path existence;
- `greater_than`, `greater_or_equal`, `less_than`, `less_or_equal`;
- `between_inclusive`.

A path query requires a boolean decision; numeric graph queries require a numeric decision. Query thresholds and interval endpoints must be finite.

## Result semantics

For each query, the report includes:
- minimum and maximum output across scenarios;
- range width;
- per-scenario graph digest, query value and decision result;
- counts meeting and not meeting the decision;
- one explicit decision status.

Statuses:
- `ROBUST_DECISION_MET`: every submitted scenario meets the declared decision;
- `ROBUST_DECISION_NOT_MET`: no submitted scenario meets the declared decision;
- `SENSITIVE_TO_RECONSTRUCTION`: at least one scenario meets and at least one does not.

The report-level status `EVALUATED_FINITE_SCENARIO_SET` means only that the submitted finite set passed validation and all queries were evaluated. Its probability claim is always `NONE_FINITE_SCENARIO_SET_ONLY`.

Scope matters:
- `exhaustive_declared_scenario_space` means exhaustive only relative to the explicitly declared finite scenario space; it does not prove that the space contains every biologically plausible reconstruction.
- `sampled_sensitivity_set` is even narrower: agreement is evidence only about the submitted sensitivity scenarios and cannot establish robustness over all plausible reconstructions.

No result is called `NOT_IDENTIFIABLE` merely because a finite scenario set contains disagreement. That label requires a separately specified identification analysis. Likewise, a robust result over an incomplete scenario set must not be promoted to global scientific certainty.

## Checked-in synthetic fixture expectations

The example has three scenarios sharing a source-volume lineage, with two segmentation/adjudication branches. It produces:
- `critical-path-1-to-4`: `SENSITIVE_TO_RECONSTRUCTION` (path exists in two scenarios and not in one);
- `spurious-edge-below-threshold`: `ROBUST_DECISION_MET` (all three scenarios meet the weight threshold);
- `node-3-outgoing-weight`: `ROBUST_DECISION_MET` (all three scenarios meet the weighted out-degree threshold).

These are exact synthetic fixture expectations, not biological measurements.

## Required next extensions

1. Add a typed correction/action scenario generator that preserves edit preconditions, conflicts and source revisions; do not independently toggle dependent edits.
2. Add query families for synapse counts, compartment-conditioned connectivity, motif counts and morphology summaries, each with a frozen semantic contract.
3. Define a scenario-construction audit: provenance ancestry, completeness of enumerated alternatives, duplicate/equivalent scenarios and excluded hypotheses.
4. Add an independently authored finite-space oracle and exhaustive enumeration tests for scenarios generated from small edit spaces.
5. Add partial-identification intervals and robust decision semantics under explicit bounded assumptions.
6. Only add weighted/probabilistic ensembles after a separate calibration and dependence contract is validated. If weights are merely user priorities or stress-test frequencies, label them as analysis mass, not probability.
7. Bind scenario set, query, dependencies, uncertainty semantics and source currentness into #9255 certificates, composing with #9210/#9239/#9203/#2618 and metric uncertainty provenance #7412.
8. Evaluate prospective scientific-query stability on independently adjudicated biological reconstructions, grouped by source volume/neuron/root/operation.

## Promotion metrics

- Zero missed/duplicate graph edges and digest mismatches in malformed-input controls.
- Exact agreement with an independent oracle on every exhaustively enumerable synthetic scenario/query pair.
- Zero false `ROBUST_DECISION_MET` claims on exhaustive synthetic counterexamples where one submitted scenario violates the decision.
- Zero unreported `SENSITIVE_TO_RECONSTRUCTION` outcomes.
- Deterministic scenario-set digest under input ordering changes.
- Scenario-construction coverage and dependency assumptions reported independently from query evaluation.
- No posterior/coverage/probability claims until weighted ensemble calibration and dependence assumptions pass separate tests.

## Non-claims

V1 does not prove biological correctness, complete scenario coverage, calibration, posterior probability, or query truth. A graph digest proves identity of the serialized graph under this canonicalization, not that the graph is the correct biological reconstruction. The tool does not authorize edits to source annotations or production connectome data.
