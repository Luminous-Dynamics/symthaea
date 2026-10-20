# Symthaea Landscape Review Profile v0

**Status:** first executable read-only review contract; not a physical landscape solver.
**Owner:** Symthaea.
**Parent:** [LAND-REVIEW-001 #8196](https://github.com/Luminous-Dynamics/symthaea/issues/8196).
**Upstream contracts:** Sol-Atlas snapshot PR #191, evaluation plan PR #210 and observation PR #216; Symtropy scenario PR #2322; Mycelix stewardship PR #5298.

## Intent and paradigm shift

Treat a landscape review as a reproducible audit of what can be said from a pinned set of project/evidence artifacts, not a generative recommendation or a scalar sustainability score. The first implementation reads exact bytes supplied by the caller, verifies their SHA-256 references, validates a few cross-system bindings, and reports coverage gaps and metric comparability.

It makes no network calls, mutates no source artifact, has no actuator authority, does not calculate environmental metrics, and never ranks alternatives. It is useful even when the correct conclusion is “unknown, not computable, or synthetic-only.”

## Rust API

The isolated workspace crate is crates/domains/symthaea-landscape-review.

The caller supplies:
- a typed ReviewRequest with two or more alternative IDs, the scope, required metric IDs, and references to project/evaluation/scenario/observation/stewardship/source artifacts;
- a BTreeMap keyed by lowercase SHA-256, where each value is the exact raw bytes for one referenced artifact.

Call validate_and_review(request, payloads_by_sha256). It returns either a deterministic ReviewReport or a sorted collection of ReviewViolation values. There is no filesystem/network loader in this crate: retrieval, access controls, signatures, consent verification and source-specific validators remain with the owning repositories.

## Verified cross-system bindings

- Project snapshot: exact bytes hash to the digest; schema URI/version, scope, frame, georeferenced flag and geometry-status are checked. v0 is LOCAL_CARTESIAN_DRAWING_UNITS with georeferenced=false.
- Evaluation plan: exact bytes hash to the reference; project_snapshot_sha256 and project_scope match the alternative. A scenario manifest's optional evaluation_plan object must match the pinned digest, schema URI and evaluation ID.
- Scenario manifest: exact bytes hash to the reference; the design snapshot digest, schema, scope, geometry status, frame, georeferencing and feature count match the project snapshot.
- Observation batch: exact bytes hash to the reference; any observation with a non-null project_snapshot_sha256 must match this project's exact revision. An observed metric must cite a batch containing at least one observation bound to that revision.
- Mycelix stewardship record: exact bytes hash to the reference; its project_snapshot.sha256 must match the reviewed project, and a linked scenario manifest must be present in the same alternative.
- Source dataset manifest: candidate metadata is not enough. A metric cannot cite a source as curated evidence unless the referenced payload is explicitly marked ingestion_state=ingested and has a stable source/dataset ID.
- Scenario result: the result references a verified scenario manifest digest and has a matching schema URI. Under the current Symtropy v0 contract, a schema-qualification or synthetic-benchmark manifest cannot qualify a modelled environmental outcome.

These are cross-reference checks, not replacements for Sol-Atlas, Symtropy or Mycelix validation. In particular, this profile does not prove that a Mycelix consent/authority reference is valid in law, nor that an observation is true.

## Evidence classes and output limits

The metric evidence vocabulary is observed, curated, synthetic, modelled, assumed and missing, with an explicit state of available, missing, not-computable or conflicting. An absent required metric produces a finding and an explicit missing entry with no value imputed. A unit/method/evidence-class mismatch keeps two metrics non-comparable.

The output has fixed flags:
- automated_ranking_allowed = false
- environmental_claims_allowed = false
- causal_claims_allowed = false

The v0 output level is schematic-evidence-review. Even a fully structurally verified project/evidence chain remains limited to coverage and human inspection because Sol-Atlas v0 geometry is schematic local drawing units. A future version can extend the claim envelope only after a versioned geospatial/model qualification contract and independent validation exist.

## Reproducibility and benchmark plan

The same typed request and exact payload bytes must produce byte-identical serialized report bytes. A byte mutation must produce a digest violation before semantic interpretation. Input vector order is normalized in the output; findings, alternatives, metric records and digest lists are sorted.

Recommended review-harness benchmark cases:
- 2, 4 and 8 alternatives;
- 8, 16, 32 and 64 required metrics;
- pinned snapshot only; snapshot + evaluation plan + scenario manifest; full observation/stewardship/source bundle;
- cold and repeated validation on identical bytes;
- 1-byte changes to every referenced artifact, missing payload, stale digest, scope mismatch, frame mismatch, source marked candidate instead of ingested, observation tied to another snapshot, synthetic→modelled relabel, mismatched metric unit/method, and stale scenario reference.

Record toolchain, OS/CPU, commit, fixture digests, input bytes, raw per-run latency and memory. Report p50/p95 only for matched environments. These are workload targets, not measured performance numbers. Parsing throughput does not establish the truth of source data or the scientific validity of a solver.

## Fixtures and upstream provenance

The crate pins a copy of the Sol-Atlas starter schematic fixture from source blob d16de158b1a8e614ef7b706f996e632a7086088f. It also vendors the upstream evaluation-plan fixture from source blob 55b413220387b8c8ca9a8b420d8e54afc0f7d5a9 and observation-batch fixture from source blob b9ef6aec0ca75ec6814c8bda7292001f3b4d2171 as test materials. Those payloads are used as fixtures; the owner repositories remain the source of truth. Updating a fixture must identify the new upstream blob SHA and prove that expected schema/digest behavior is unchanged or versioned.

## Build and tests

From the Symthaea workspace:

    cargo fmt --check --package symthaea-landscape-review
    cargo test -p symthaea-landscape-review

Tests cover deterministic report serialization, exact-byte digest mutation, unknown metrics as gaps rather than zero, wrong scope, unreferenced payload rejection and metric unit mismatch. Further cross-contract tests should be added alongside the authoritative upstream validators, not by re-implementing their entire schemas here.
