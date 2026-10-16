# Enzyme evidence graph verifier V0

**Status:** initial structural/provenance verifier; policy and scientific semantics still need expert review.

The Rust command enzyme-evidence-graph-check is the first executable subset of the evidence-graph contract (ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json). It aims to stop internally inconsistent provenance records from being mistaken for reproducible science. It does not duplicate EnzymeML/SBML, fit kinetic models, validate chemical balance, clear data licenses, or score an enzyme predictor.

## Run it

For a workspace-local graph:

    cargo run -p xtask -- enzyme-evidence-graph-check --input path/to/graph.json

The checker confines the graph file and acquired artifact paths to the workspace, including symlink targets. It never fetches an HTTPS locator. A remote artifact must be staged deliberately, then represented by a workspace-relative locator with acquisition_status: "available" so its exact bytes can be checked.

For a frozen graph, compute the SHA-256 over the exact serialized JSON bytes and pass the external pin:

    sha256sum path/to/frozen-graph.json
    cargo run -p xtask -- enzyme-evidence-graph-check --input path/to/frozen-graph.json --sha256 EXPECTED_64_HEX_DIGEST

The digest must be stored in the separate run receipt or publication record; embedding a file's own digest into that same file would change the bytes being hashed. A supplied digest is checked for draft and frozen graphs alike; status "frozen" requires one.

## V0 graph record layout

The top-level JSON object contains:

- schema_version: 1, a non-empty graph_id, and status (draft or frozen).
- rights_records: unique source-specific records with rights_record_id, source_id, license_id, attribution, and status. Accepted statuses are approved_for_this_use, review_pending, restricted, rejected, and unknown. These labels are declarations to audit, not legal determinations. A successful transformation is blocked unless all its input and output artifacts have status approved_for_this_use; the checker enforces the recorded label, not whether the review itself was legally or scientifically correct.
- artifacts: unique records with artifact_id, role, media_type, sha256, byte_length, locator, rights_record_id, and acquisition_status.
- nodes: unique node_id, contract-listed type, its matching evidence_class, and artifact_ids (use an empty array when none are attached).
- edges: unique edge_id, contract-listed type, from_node_id, and to_node_id.
- transformations: unique transformation_id, name, version, SHA-256 values for code, configuration, and environment lock, explicit input/output node and artifact IDs, deterministic, terminal_status, and a non-empty terminal_reason for every status except success. For success, code_artifact_id, configuration_artifact_id, and environment_lock_artifact_id are required and each must reference an available artifact of the matching role whose bytes match the associated SHA-256. Those three identities are separate from scientific input/output artifacts. Failed/blocked records may omit them if execution never started.
- parameter_claims: an optional array. When present, every claim records the contract's parameter name/endpoint/value/unit, enzyme form basis, reaction and substrate identifiers, assay context object, kinetic-equation identifier, source-or-fit node ID, and uncertainty status.

For artifacts, acquisition_status "available" requires locator to be a workspace-relative path that resolves to bytes whose length and SHA-256 match the record. Paths with absolute roots, dot/parent traversal, or symlink escapes fail closed. acquisition_status "not_acquired" requires an HTTPS locator and is metadata-only; its declared digest is shape-checked but not presented as verified.

Terminal transformation statuses are success, null_result, failed, excluded, out_of_domain, and blocked. A non-success status and its reason must remain in the graph. null_result must not claim output nodes or artifacts. A successful transform must declare input and output nodes, and every input/output artifact it declares must be locally verified. Transform dependencies must be represented by edges, endpoints must resolve, edge types must match the contract's endpoint types, and the graph must be acyclic.

Parameter claims distinguish source-reported values from fit outputs. The raw source value must be a JSON number or a non-empty verbatim string; objects, arrays, booleans, and null are rejected rather than silently reinterpreted. A normalized value and unit must be supplied together, and a normalized value must be numeric. At least one substrate ID is required; unknown substrate identity must be represented explicitly rather than by an empty list. Duplicate substrate IDs are rejected. An unknown uncertainty cannot be encoded as zero uncertainty; the allowed statuses are reported, recomputed, not_available, not_estimable, and not_applicable. When source_or_fit_node_id points to a source_record, that record must have a reported_by edge to a reported_parameter_claim node; when it points to a fit_run, that run must have an estimates edge to a reproduced_parameter_estimate node. Direct references to reported_parameter_claim or reproduced_parameter_estimate nodes remain valid. These checks require graph-level lineage instead of accepting an otherwise valid ID with no relevant claim edge. A value is not admitted merely because it is syntactically valid.

## What V0 checks—and does not check

V0 checks graph/node/artifact/edge/transformation/parameter identifier integrity; rejects unknown fields on every typed graph record rather than silently discarding misspelled keys; requires every declared artifact to be referenced by an evidence node or transformation; evidence-class and edge endpoint typing; dangling references; directed cycles; workspace confinement; local artifact byte length and SHA-256; required transformation identity fields and exact code/configuration/environment artifacts; input/output linkage; terminal-status reasons; parameter-claim uncertainty vocabulary; and frozen-graph byte pinning. A test loads the machine-readable contract and compares the Rust node/evidence-class, edge, artifact-role, graph-status, rights-status, acquisition-status, terminal-status, and uncertainty vocabularies against it so ordinary policy drift fails the test suite. Synthetic tests exercise these conditions.

V0 does not yet verify a JSON Schema, a semantic-unit ontology, reaction balance, source-specific license interpretation, remote downloads, actual computational replay, tolerances between pipeline stages, uncertainty calibration, causal validity, fit identifiability, or model generalization. For successful transforms, V0 binds the code, configuration, and environment-lock SHA-256 fields to separate, locally available artifacts, verifies their exact bytes, and requires their recorded rights status to be approved_for_this_use. That verifies byte identity, not that the code was executed as claimed: independent stage-wise computational replay, software/toolchain semantics, and output-tolerance checks remain unimplemented. A hash proves byte identity, not scientific truth or rights clearance. rights_reviews_unresolved in the report is informational; it is not a license approval. deterministic is recorded metadata, not a proof of determinism.

The command's current success message therefore means **the submitted graph passed this structural and local-byte verifier only**. It is not a claim that any enzyme parameter was independently reproduced or that a predicted mutation will improve activity.

## Adversarial tests

The unit-test fixtures are synthetic, never sourced biological observations. Regression cases include unreferenced-artifact rejection; unknown-field rejection at the graph, node, edge, artifact, rights-record, transformation, and parameter-claim levels; valid paired raw/normalized parameter values; malformed parameter value types; incomplete normalized value/unit pairs; duplicate substrate IDs; and valid local hashes plus metadata-only remote links, dangling references, directed cycles, tampered bytes, attempted use of unacquired remote data in a successful transformation, evidence class/type mismatch, missing output-artifact linkage, missing failure reason, a null result with inputs but no output node, invalid uncertainty status, workspace path traversal and symlink escape, malformed transformation fingerprints, and required frozen-graph hash pinning.
