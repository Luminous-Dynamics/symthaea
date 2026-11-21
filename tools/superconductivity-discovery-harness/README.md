# Superconductivity discovery registry harness

This is a standalone Rust tool for validating the versioned superconductivity discovery registry.

## Run

From the repository root:

```bash
cargo generate-lockfile --manifest-path tools/superconductivity-discovery-harness/Cargo.toml
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml
cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml -- \
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

## Condition-bound observation event contract

The registry identifies source records and candidate materials; the observation-event schema preserves each reported measurement/calculation/prediction as its own condition-bound event.

Schema: `data/schemas/superconductivity_observation_event_v0_1.schema.json`.

Synthetic smoke fixture (never a real material claim): `tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json`.

Run:

```bash
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_observation_event_audit
cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_observation_event_audit -- \
  --schema data/schemas/superconductivity_observation_event_v0_1.schema.json \
  --fixture tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json
```

The auditor refuses orphan source IDs, duplicate observation IDs, non-detection encoded as zero, unknown pressure coerced to ambient, ambient pressure without explicit zero, applied pressure without a positive value, and calculated/predicted values represented as experimentally detected. Negative controls exercise each refusal.

This is an observation-level data contract. It does not verify the underlying source claim or establish scientific truth; production imports still require a source-pinned, rights-reviewed record and appropriate human adjudication.

## Process history and metastable states

Observation events may carry an ordered `process_history` separate from the conditions under which the property was measured. Each step records operation, intended target, execution state, actual values and—when sensor-verified—source-linked readback.

For a pressure-quenched sample this allows a record to state:
- treatment pressure and quench temperature during the process;
- decompression to ambient pressure;
- the final measurement pressure and temperature;
- the measured Tc criterion and sample identity.

The process history never overwrites or infers the measurement pressure. Planned/commanded steps cannot contain actual values; sensor-verified steps require a resolvable source and locator. Step IDs and sequence numbers must be unique/monotonic. Synthetic negative controls exercise those rules.

## Synthesis protocol and execution contract

Schema: `data/schemas/superconductivity_synthesis_protocol_v0_1.schema.json`.

The schema separates:
- the target material and precursor lots;
- planned ordered operations and setpoints;
- execution runs with commanded, unverified, sensor-verified, failed or aborted states;
- actual sensor/controller readback and source references;
- realized phases, phase fractions, characterization-event references and failures.

The Rust executable validates one synthetic protocol fixture:

```bash
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_synthesis_protocol_audit
cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_synthesis_protocol_audit -- \
  --schema data/schemas/superconductivity_synthesis_protocol_v0_1.schema.json \
  --fixture tools/superconductivity-discovery-harness/fixtures/synthesis_protocol_smoke_v0_1.json \
  --observation-schema data/schemas/superconductivity_observation_event_v0_1.schema.json \
  --observations tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json
```

It refuses commanded setpoints represented as actual execution, unresolved sensor evidence, failed runs without failure reasons, execution steps absent from the plan, outcomes without a valid run, duplicate/non-monotonic planned steps and reported phase fractions above 100%.

The synthetic fixture is an abstract pressure-hold/decompression example; it is not an experimental recipe, pressure-vessel instruction or materials result. A validated protocol record is not safety authorization.

The synthesis auditor also validates the outcome-to-characterization edge. Each outcome event ID must resolve to the observation-event fixture, use an observed assertion, and match the same material formula and sample ID. This catches detached Tc records, prediction-as-characterization, specimen mismatches and wrong-material joins.

## Pairing and phase-coherence evidence scope

Observation assertions may now use distinct property codes for pairing energy, pair-transfer effective charge, superfluid stiffness/density, phase-correlation length/time, Josephson coupling, vortex density and kinetic inductance.

Each assertion can declare a measurement scope: `local_probe`, `mesoscopic`, `bulk`, `device`, `theoretical_model` or `unknown`.

The auditor refuses to promote a local-probe pairing signal to bulk Tc, or to use a spectroscopic gap as a Tc criterion. Observed pairing energy is represented in meV and must identify an experimental scope. The smoke fixture has separate synthetic local pairing and bulk stiffness events. These are schema tests, not observations from the cited paper.

The Niu et al. Science 2026 Bi2212 shot-noise study reports STM measurements at cryogenic base temperatures of 4.2 K, 2.3 K and 0.3 K, with a heterogeneous pairing-onset energy scale up to ~70 meV. That energy is not a Tc. The experiment did not directly measure pair transfer at 300 K or demonstrate bulk room-temperature superconductivity. Preprint: arXiv:2409.15928 (first submitted 24 September 2024); Science DOI: 10.1126/science.adt3444.


## Disorder screening and phase-coherence benchmark references

New source records distinguish:
- metallic outer planes screening dopant disorder while remaining metallic and weakly proximitized;
- single-plane Bi-2201 as a dimensionality-reduction control;
- checkerboard spatial-order threshold as one family-specific phase-coherence mechanism;
- field-free stiffness-tensor and resonator kinetic-inductance metrology.

A local gap or ARPES spectral signature does not establish a new Tc. The surface-doped F0223 state is not assigned a Tc in this registry. For stiffness extraction from resonator kinetic inductance, retain geometry, current-distribution and electrodynamic assumptions; do not convert with a universal geometry-free constant.


## Same-specimen pairing / phase-coherence comparison

Schema: \`data/schemas/superconductivity_pair_coherence_comparison_v0_1.schema.json\`.

This comparison contract links one local pairing observation and one bulk/mesoscopic coherence observation while enforcing exact material, material-state, sample and batch identity. It also checks fixed pressure semantics/value, field, doping and strain conditions. It refuses:
- cross-sample, cross-batch, cross-material or cross-state joins;
- predicted/calculated values used as observed pairing or coherence evidence;
- a local pairing signal relabelled as bulk Tc;
- non-detected or unresolved specimen records;
- pressure/doping/strain mismatch;
- conversion of pairing energy in meV to a temperature.

Run the synthetic-only audit:

\`\`\`bash
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_pair_coherence_audit
cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_pair_coherence_audit -- \\
  --comparison-schema data/schemas/superconductivity_pair_coherence_comparison_v0_1.schema.json \\
  --comparison tools/superconductivity-discovery-harness/fixtures/pair_coherence_comparison_smoke_v0_1.json \\
  --observation-schema data/schemas/superconductivity_observation_event_v0_1.schema.json \\
  --observations tools/superconductivity-discovery-harness/fixtures/observation_events_smoke_v0_1.json
\`\`\`

The fixture uses synthetic local pairing-energy and bulk stiffness observations on one synthetic specimen. It does not represent the Science paper or any physical measurement, and it does not infer a pairing temperature or a new Tc.


## Current seed-registry inventory

The current v0.1 registry contains 23 source records, 13 candidate/material references and 18 benchmark catalogs. This is a source inventory, not 23 independent experiments or 13 validated new superconductors. The new metallic-screening, single-plane, spatial-coherence and stiffness-metrology sources are separately represented, and the source-level measurement/claim boundaries are preserved.

Measurement scope is now required on every observation assertion. An observed Tc must be explicitly mesoscopic, bulk or device scope; local_probe and unknown fail closed. Calculated and predicted records retain their own scope (for example theoretical_model). This prevents unscoped measurements from being silently treated as bulk evidence.

## External literature-library import identity

Observation events may carry a source.external_catalog object when extraction_method is database_import. It preserves library_id, external_record_id, API version, dataset version, snapshot ID, retrieval time, ETag, X-Data-Version and parent paper/material/sample/state/result IDs.

The Rust observation auditor now requires those fields on database imports, checks X-Data-Version against the declared dataset version, and verifies imported parent material/sample/state/result IDs against the event. A synthetic SCLib-shaped example and negative controls are included. No real external dataset has been downloaded or mirrored.


## Budgeted discovery selection auditor

The budgeted selection audit extends deterministic ranking metrics with family/source-disjoint holdouts, explicit target-track semantics, OOD abstention, calibration and fixed-cost policy comparisons.

Schema: `data/schemas/sc_budgeted_selection_v0_1.schema.json`.

Synthetic fixture: `tools/superconductivity-discovery-harness/fixtures/budgeted_selection_smoke_v0_1.json`.

Run:

```bash
cargo test --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_budgeted_selection_audit
cargo run --locked --manifest-path tools/superconductivity-discovery-harness/Cargo.toml --bin sc_budgeted_selection_audit -- \
  --schema data/schemas/sc_budgeted_selection_v0_1.schema.json \
  --fixture tools/superconductivity-discovery-harness/fixtures/budgeted_selection_smoke_v0_1.json \
  --out /tmp/sc-budgeted-selection-smoke.json
```

The auditor compares model probability-per-cost, a chemistry heuristic, deterministic random selection and an exact synthetic-truth oracle under the same total follow-up budget. It reports selected count, true positives, precision, recall, positives per cost unit, enrichment versus random and regret against the oracle. Brier score, five-bin expected calibration error, selective classification error and coverage are calculated only for non-abstained evaluation candidates.

Training and evaluation pools are disjoint by family, prototype, reduced formula and source paper. The target track is fixed to conventional electron-phonon superconductor classification in v0.1. Other mechanism and label tracks require separate cohorts; in particular, this audit does not turn BEE-NET's DFT/Allen–Dynes regression error into measured-Tc accuracy. Out-of-domain candidates must abstain with no probability and cannot be selected by the model policy.

The fixture is fully synthetic. The oracle is a scoring ceiling over the hidden synthetic truth, not a deployable model. The audit validates evaluator arithmetic and semantics only; it does not reproduce BEE-NET or demonstrate a real discovery advantage.


The discovery registry schema treats `arxiv_id` as optional nullable metadata: a source may be identified by its required DOI/URL and source ID without an arXiv identifier. This preserves source identity without inventing an arXiv record. The synthetic discovery fixture must also distinguish `manual_curated` from `database_import`; only actual external-catalog imports may claim catalog record/snapshot provenance.
