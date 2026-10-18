# Offline decision-quality evaluator v1

**Status:** draft evaluator implementation and authored unit tests. It has not been compiled or executed in this environment.

## Goal

The evaluator gives the decision-quality corpus an executable scoring path without embedding benchmark scoring into the policy. It compares a **frozen policy-output file** against oracle-only expectations after inference. It scores finite fixture contract correctness; it is not a recipe planner, a VOI optimizer, or a proof of food safety or household benefit.

## Run locally

From the repository root, after the Rust toolchain is available:

```sh
cargo run -p symthaea-culinary --bin culinary_decision_quality_eval -- \
  crates/domains/symthaea-culinary/fixtures/culinary-decision-quality-v1/corpus.json \
  /path/to/frozen-policy-outputs.json \
  /path/to/evaluation-report.json
```

Omit the third path to print the report to stdout. Exit code 0 means every corpus case had exactly one output, the explicit expected fields matched, and no implemented semantic guards fired. Exit code 1 means the scored report did not pass. Exit code 2 means usage, input, version, or parse validation failed.

## Policy output contract

The candidate policy runs separately and receives **only** the `policy_input` object from each corpus case. Never give it `oracle_only_expected`, the complete case object, evaluator output, or scorer source during a held-out run. Freeze its output artifact before the oracle is joined.

The output file has this shape:

```json
{
  "corpus_id": "culinary-decision-quality-voi-v1",
  "corpus_version": "1.0.0",
  "outputs": [
    {
      "case_id": "DQ-001-stale-price-can-flip-choice",
      "disposition": "request_measurement",
      "selected_candidate_ids": [],
      "measurement_id": "remeasure_gamma_item_prices",
      "flags": ["stale_evidence"],
      "source_refs_used": [],
      "consumer_output_fields": [],
      "evidence_claims": [],
      "permutation_outputs": []
    }
  ]
}
```

Each output needs exactly one `case_id` and a disposition from `recommend`, `preserve_choice_set`, `request_measurement`, `human_review`, or `no_feasible_candidate`. Provide the explicit result fields a case expects, including empty arrays and explicit `null` values where applicable. Output sets are compared independently of order; the permutation track instead compares the two complete `decision` objects inside `permutation_outputs`.

The scorer rejects version mismatch, missing outputs, duplicate case IDs, and unexpected case IDs. It reports per-case disposition agreement, all expected fields it could score, and independently detected guard violations for selected unresolved gates, recommendations depending on missing/stale evidence, selected affordability-limit violations, use of known-at-ineligible source references, privacy-projection excess, relabeling simulated evidence as observed, and continued selection of a recalled lot.

## What it does not prove

- It does not implement the candidate policy and does not calculate expected VOI.
- It does not authenticate source refs, confirm source truth, perform nutrient unit conversion, or validate a full JSON Schema.
- It does not treat the corpus's descriptive `must_not` strings as an arbitrary executable rule language. Only the documented explicit expected fields and implemented semantic guards are scored.
- It does not establish probability calibration, clinical quality, allergen/food safety compliance, source coverage, or real-world household improvement.
- The evaluator is not a security sandbox. Process separation, sealed holdout storage, access controls, and a trusted evaluation harness are needed to prevent oracle leakage.

The first useful qualification test is exact agreement on this finite synthetic corpus. Before tuning on any held-out set, record the corpus SHA-256, freeze the policy version/configuration and output artifact, and publish the complete report—including failures. Any change to expected outputs requires a new corpus version.
