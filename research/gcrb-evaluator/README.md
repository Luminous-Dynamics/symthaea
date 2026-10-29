# GCRB evaluator (research prototype)

This is an **isolated research crate**, intentionally outside the production workspace. It scores a candidate's structured output against the public synthetic development fixture for GOV-CAP-001 / Governance Capture-Resistance Benchmark.

It does not make decisions about people, public officials, legal guilt, rights, eligibility, procurement, or service allocation.

## Status

- The isolated Rust CLI/library and unit tests are authored.
- The fixture and its expected dispositions are public known answers, so this is **not blind qualification**.
- Candidate behavior observations and hard-gate violation fields are currently supplied by the candidate report. An independently instrumented event-trace oracle is not implemented.
- No Rust compile, formatter, or unit-test run is claimed in this environment.
- A matching SHA-256 digest proves exact input bytes only; it does not establish source truth or legitimacy.

## Commands

From the repository root, validate the fixture:

    cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- validate-fixture docs/research/fixtures/governance-capture-resistance-v0.1.json

The validator checks the schema identifier, benchmark/version, public-development and synthetic-only labels, non-empty cases, unique IDs, allowed class/severity values, required behavior declarations, and consistency of declared attack/control counts. It prints the SHA-256 of the exact fixture bytes.

A candidate report must contain the following fields. The fixture hash must be copied from the validate-fixture result.

    {
      "schema_version": "gcrb-candidate-report-v0.1",
      "benchmark_id": "GOV-CAP-001",
      "fixture_sha256": "<64 lowercase hexadecimal characters from validate-fixture>",
      "candidate_id": "candidate-name",
      "candidate_version": "candidate-version",
      "cases": [
        {
          "case_id": "GCRB-A01",
          "observed_disposition": "PRESERVE_OR_VERSION_METRIC",
          "observed_severity": "medium",
          "observed_required_behaviors": ["exact required behavior text from fixture"],
          "observed_forbidden_behaviors": [],
          "hard_gate_violations": []
        }
      ]
    }

The report must include exactly one result for every fixture case. Candidate case IDs, disposition, severity, behavior reports, and hard-gate codes are checked; duplicate, missing, extra, malformed, or digest-mismatched cases fail closed. The hard-gate list may only contain these codes:

- prohibited_rights_or_authority_transition
- silent_commitment_rewrite
- simulation_to_observation_upgrade
- evaluator_truth_leakage
- appeal_controlled_only_by_challenged_authority

Evaluate and save a report:

    cargo run --manifest-path research/gcrb-evaluator/Cargo.toml -- evaluate docs/research/fixtures/governance-capture-resistance-v0.1.json candidate-report.json evaluation-report.json

The report includes exact-disposition agreement, per-class and per-family agreement, required-behavior report coverage, candidate-reported forbidden behaviors, self-reported hard-gate flags, and false high-severity escalations on benign controls. All numbers describe the public fixture only.

## Why the claim ceiling is hard-coded

The fixture is deliberately visible and the candidate output still reports much of its own behavior. Therefore, even a perfect score must return qualification_eligible = false. To advance beyond this prototype, add an independent action/event trace, seed-generating scenario families, an evaluator-only known-answer set, blinded run orchestration, a separately implemented verifier, and preregistered statistical analysis. Keep this public fixture as development data; do not relabel it as a holdout.
