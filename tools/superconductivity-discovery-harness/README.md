# Superconductivity discovery registry harness

This is a standalone Rust tool for validating the versioned superconductivity discovery registry.

## Run

From the repository root:

```bash
cargo test --manifest-path tools/superconductivity-discovery-harness/Cargo.toml
cargo run --manifest-path tools/superconductivity-discovery-harness/Cargo.toml -- \
  --schema data/schemas/superconductivity_discovery_case_v0_1.schema.json \
  --instance data/research/sc_superconductor_discovery_cases_v0_1.json \
  --out /tmp/sc-discovery-registry-receipt.json
```

The audit:
 
1. Parses and meta-validates the declared JSON Schema Draft 2020-12 contract.
2. Refuses `$ref`, `$dynamicRef` and `$recursiveRef` so the schema cannot cause external or indirect schema resolution.
3. Validates the registry instance against the committed schema.
4. Enforces unique candidate/source identifiers, requires a DOI or arXiv identifier for every source, and verifies closed source references.
5. Keeps measured transitions, ML predictions, and first-principles calculated Tc values in three separate channels: `tc_observations`, `tc_prediction`, and `tc_calculations`. A material may legitimately have both an experimental Tc observation and a calculated Tc estimate, but the two are never merged.
6. Requires actinide-bearing candidates to carry an explicit computational-only or exclusion gate.
7. Emits deterministic SHA-256 digests and record counts in a JSON receipt.

The `jsonschema` dependency is pinned and built with default features disabled. The validator does not retrieve HTTP or filesystem references. Cargo may still need to download dependencies when the package is first built; validation itself is offline and reads only the explicit input files.

## Claim ceiling

A passing registry audit verifies data shape and selected evidence semantics. It does not verify source truth, reproduce the source calculations, establish phase stability, or qualify a material experimentally or for engineering use.

## Ranking metrics runner

The standalone harness also provides a deterministic top-K ranking calculator.

```bash
cargo test --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_discovery_metrics
cargo run --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_discovery_metrics -- \
  --predictions path/to/predictions.json \
  --budgets 10,25,50,100 \
  --out /tmp/sc-discovery-metrics.json
```

Prediction input is a JSON array of unique records with candidate_id, score (finite number), and relevant (boolean). Scores are ranking values only; the runner does not interpret them as calibrated probabilities.

The report contains precision@K, recall@K, random-prevalence expectation, enrichment over random, average precision and input SHA-256. Equal-score rows are ordered by candidate ID for deterministic replay. Invalid/duplicate budgets and duplicate candidates fail closed. When no positive labels exist, recall and enrichment are null rather than fabricated.

The workflow runs this binary against a clearly synthetic six-row plumbing fixture; this only validates deterministic metric execution and does not represent material evidence, model quality or scientific discovery. Confidence intervals, calibration, OOD selective risk, cost-normalized comparisons and prospective validation remain separate benchmark work and must not be inferred from these ranking metrics.

The GitHub Actions workflow creates an isolated Cargo.lock for each exact run, uses --locked for the subsequent build/test commands, and uploads the generated lockfile, toolchain versions, source digests and receipts as one run artifact. This allows the resolved transitive dependency graph to be reviewed and committed after the first successful exact-head run rather than hand-authoring a lockfile. The lockfile is generated from the pinned Rust 1.96.0 toolchain and exact direct dependency versions.
