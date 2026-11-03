# PITB typed proof-graph evidence verifier

A standalone Rust CLI for validating the **identity, relation typing, provenance references, and local byte integrity** of a PITB proof-graph snapshot.

This is a companion contract to the current two-level proof-graph scheduler. It does not replace or mutate the existing `pitb.proof-graph-state.v0` schema. The new contract distinguishes the kinds of graph edges that a single `dependency_ids` list cannot express.

## Run

From the repository root:

```sh
cargo fmt --check --manifest-path tools/pitb-graph-evidence/Cargo.toml
cargo test --locked --manifest-path tools/pitb-graph-evidence/Cargo.toml
cargo run --locked --manifest-path tools/pitb-graph-evidence/Cargo.toml -- \
  --self-test \
  tools/pitb-graph-evidence/fixtures/graph-evidence.synthetic.json \
  tools/pitb-graph-evidence/fixtures/evidence
cargo run --locked --manifest-path tools/pitb-graph-evidence/Cargo.toml -- \
  --verify path/to/graph.json path/to/evidence-root
cargo run --locked --manifest-path tools/pitb-graph-evidence/Cargo.toml -- \
  --verify-attested path/to/graph.json path/to/evidence-root \
  path/to/attestation.json path/to/key-registry.json EXPECTED_REGISTRY_SHA256
```

The crate includes Rust unit tests for path policy, relation/evidence mapping, confidence semantics, formal-vs-candidate roles and cycle policy. The synthetic fixture is also checked end-to-end by the CLI self-test. Its reviewer note and extractor receipts are deliberately synthetic and must never be mistaken for real Lean evidence or reviewer attestations.

## Edge ontology

All edges have a source and target. The relation determines the direction semantics; formal declaration edges represent the source declaration's dependency on the target.

| Relation | Evidence class | Derived role | Can become a hard scheduler edge in v0? |
|---|---|---|---|
| `signature_reference` / `type_reference` | `lean_elaborator` | Elaborated declaration type/signature references | No |
| `proof_term_reference` | `lean_elaborator` | Observed proof support, not logical necessity | No |
| `definition_body_reference` | `lean_elaborator` | Non-Prop definition-body references | No |
| `structure_field_reference` | `lean_elaborator` | Structure field type references | No |
| `inheritance_reference` | `lean_elaborator` | Structure/class extension and inheritance | No |
| `instance_synthesis` | `lean_instance_trace` | Environment/instance dependency observation | No |
| `documentation_reference` | `lean_doc_parser` | Observed docstring reference; retrieval only | No |
| `module_environment_dependency` | `lean_build_graph` | Build/environment dependency observation | No |
| `blueprint_dependency` | `blueprint_curated` | Candidate or reviewed search prior | No |
| `informal_candidate_dependency` | `informal_extractor` | Candidate retrieval hint | No |
| `cross_formality_match` | `llm_alignment` | Candidate retrieval hint | No |

The verifier requires relation/evidence-class compatibility, compatible node kinds, resolvable node/artifact IDs, unique edge identities, and an acyclic module/environment dependency graph. It deliberately permits cycles among observed type/proof/instance declaration references because Lean mutual declarations can produce valid strongly connected components; those observations must not be confused with the separate acyclic helper-plan DAG used by the scheduler. It rejects malformed confidence values, unsupported status/relation combinations, missing review artifacts for `human_reviewed` edges, unknown fields, unsafe paths, symlink traversal, digest mismatches, duplicate paths and unreferenced artifacts.

## Trust boundaries

### Signed snapshot envelope

The optional `--verify-attested` mode verifies an Ed25519 attestation over the SHA-256 of the exact graph-manifest bytes and binds it to the repository commit, environment digest, snapshot ID, extractor ID/version and extractor configuration digest. The attestation's signer and key IDs must resolve to an active entry in a strict registry whose SHA-256 is supplied separately on the command line. The attestation uses the versioned compact JSON canonicalization `serde-json-map-sorted-compact-v0`.

A valid signature authenticates the issuer and the exact snapshot bytes; it does **not** prove that the extractor's semantic claims are true. The verifier still hard-codes `scheduler_admission: false`. Production use needs independently rerun Lean extraction and exact edge-set comparison under the pinned source and environment. The in-memory signature self-test uses a deterministic synthetic key only; never use its seed for real signing.

A passing run means only:
- the graph envelope matches the structural contract;
- declared artifact paths stay inside the canonical evidence root and contain no symlink component;
- statement bytes match their declared statement digests;
- edge evidence and review artifact bytes match their declared SHA-256 digests;
- typed relations, endpoint kinds, status fields and reference joins are internally consistent.

In `--verify` mode, the graph manifest itself is not authenticated. `--verify-attested` adds an external-key-registry-bound signature check, but still authenticates only the issuer and exact manifest bytes, not the semantic truth of the extractor's claims.

A passing run does **not**:
- prove that an extractor receipt truthfully describes a Lean elaboration event;
- prove source-to-statement or informal-to-formal semantic correspondence;
- authenticate reviewer signatures or reviewer-key revocation;
- prove that a proof-term dependency is logically necessary;
- certify any Blueprint or informal candidate edge as a theorem dependency;
- check a Lean proof, theorem, axiom closure, or target integrity;
- authorize any edge as a scheduler dependency.

The summary therefore hard-codes `scheduler_admission: false`. Even a byte-verified `type_reference` is only a hash-bound graph assertion until a separate semantic receipt verifier authenticates the source, exact elaborated declaration and pinned Lean environment. Human-review metadata is similarly reported with an explicitly unverified signature status. The existing benchmark-integrity verifier's externally pinned Ed25519 reviewer registry remains the route for authenticated review.

## Architecture guidance

Use this graph as a **typed evidence layer** alongside the intra-declaration AND/OR proof hypergraph and cross-declaration helper DAG:

- type references describe elaborated type dependencies;
- proof-term references record observed support, not necessity;
- instance-synthesis edges bind to the exact environment and local/global instance state;
- module/environment edges capture build prerequisites;
- Blueprint edges provide curated research priors;
- informal candidate edges and cross-formality matches can rank retrieval candidates but cannot gate proof progress.

A proof plan is an AND/OR hypergraph over compatible premise bundles, not just a ranked list. Several alternative bundles may solve the same obligation. PITB's support-antichain experiments can estimate sufficient bundles under a fixed target, prover policy, context and budget; a proof-term reference alone does not establish universal necessity.

## Research motivation and baselines

DAPS reports dependency-aware group-level premise selection and two-hop bidirectional graph context; TheoremGraph/LeanGraph publishes a large elaborator-derived typed declaration graph and lower-trust informal candidate links. PITB should adapt these as retrieval baselines, preserving their provenance and original split definitions. DAPS's public code is Apache-2.0 but uses a Python/PyTorch/PyTorch-Geometric reproduction stack. This crate keeps the provenance verifier Rust-first; it does not import DAPS's training stack.

- DAPS: https://arxiv.org/abs/2610.41536 and https://github.com/DAPS-ETH/DAPS
- TheoremGraph: https://arxiv.org/abs/2606.25363
- LeanGraphSearch: https://arxiv.org/abs/2610.22339
- Existing PITB two-level scheduler: `docs/research/pitb-proof-graph-scheduler-2026-11-01.md`

## Evidence status

The checked-in sample is a synthetic contract fixture, not a measured snapshot. No live Lean graph has been imported, no DAPS/TheoremGraph retrieval run has been compared, and no scheduler has consumed edges from this tool. The dependency versions are exact-pinned in `Cargo.toml`, the generated `Cargo.lock` is committed, and the isolated exact-head workflow has qualified the previous seven-relation fixture; the expanded TheoremGraph-compatible relation set and unit tests are awaiting their own exact-head run.
