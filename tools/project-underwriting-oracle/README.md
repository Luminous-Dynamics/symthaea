# Project Underwriting Oracle

A small, deterministic Rust evaluator for the public synthetic v1 case corpus.

## Run

From the repository root:

\`\`\`bash
cargo test --manifest-path tools/project-underwriting-oracle/Cargo.toml

cargo run --manifest-path tools/project-underwriting-oracle/Cargo.toml -- \\
  docs/research/project-underwriting/v1/public-cases.json \\
  docs/research/project-underwriting/v1/reference-expected.json
\`\`\`

The command prints a JSON receipt containing the case dispositions, mismatches and (in GitHub Actions) the checked-out \`GITHUB_SHA\`. It exits non-zero if actual dispositions do not match the transparent reference file.

## What it checks

- Evidence IDs are unique; claim IDs are unique; every evidence record belongs to a declared claim.
- Evidence references resolve across claims, grid, physical model, yield uncertainty components, cash flows and scenarios.
- Issuer-signed evidence has verified authenticity, exact issuer role, subject identity and boundary.
- Decision-time evidence is not known after the decision timestamp.
- Grid connection status and allocated capacity meet the synthetic gate.
- Physical model qualification is valid for the project's capacity and assessment date, and has independent calibration evidence.
- P50/P90/P95 energy quantiles have the required exceedance ordering.
- Cash-flow schedule currency, ordering and date convention are explicit.
- FX evidence is required when reporting and cash-flow currencies differ.
- Unknown dependence remains indeterminate rather than becoming a joint tail-risk estimate.

## Deliberate limitations

- The tool evaluates the public synthetic corpus only. The expected outcomes are also public and are not a blind certification set.
- The JSON Schema is the normative structural contract, but this v0.1 executable is a semantic gate evaluator, not a full Draft 2020-12 JSON Schema implementation.
- Timestamps in the v1 evaluator must be canonical UTC RFC 3339 strings ending in \`Z\`; it compares those frozen strings lexically.
- Financial amounts are read as decimal strings and parsed as floating-point only for this synthetic gate evaluator. This tool is not a ledger, a money migration tool or a production valuation engine.
- No physical model is run; model outputs are fixture inputs and only their declared domain/evidence relationships are checked.
- No recommendation, bankability opinion, suitability assessment, capital-allocation authority, trading effect, engineering certification or regulatory compliance conclusion is produced.

A future sealed certification set must be stored and evaluated outside the candidate's accessible repository and bound to an exact candidate commit.
