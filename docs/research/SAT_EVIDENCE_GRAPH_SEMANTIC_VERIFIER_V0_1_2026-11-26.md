# SAT Evidence-Graph Semantic Verifier v0.1

**Date:** 2026-11-26  
**Status:** Rust reference implementation and 16 source-authored unit tests; compilation and exact-head execution pending.  
**Crate:** `crates/core/symthaea-sat-evidence-graph-verifier`  
**Purpose:** independently verify the semantic integrity and canonical commitment of SAT benchmark-augmentation evidence graphs.

## 1. Why this implementation exists

JSON Schema validates local shape. It cannot, by itself, prove that identifiers are globally unique, evidence references resolve, dependency edges agree with declared dependencies, or a dependency graph is acyclic. A content digest also does not establish truth or authenticity.

The semantic verifier fills that gap with a deterministic, read-only Rust implementation. It is a separate verification layer, not a new benchmark runner, authority service, evidence collector or SAT qualifier.

## 2. Verification contract

The verifier:
1. Parses a restricted JSON profile and rejects duplicate object keys, floating-point numbers and integers outside the safe-integer range.
2. Enforces an input-size and graph-size ceiling before graph processing.
3. Checks uniqueness of observation IDs and claim IDs.
4. Requires every claim evidence reference to resolve to exactly one observation.
5. Requires every typed edge endpoint to resolve to a claim.
6. Requires each claim's `depends_on` set and its typed `depends_on` edges to agree exactly, with one edge per dependency.
7. Rejects cycles in the dependency subgraph.
8. Rejects supported claims with stale validity, missing evidence, unknown/invalid authenticity, or unverified/unsatisfied independence.
9. Requires critical claims to use observations with verified authenticity and satisfied independence.
10. Requires effect-correspondence claims to include a verified, independent `external_effect_readback` from an `independent_sink`.
11. Checks the SAT v0.1 vocabularies for observation kinds, source roles, authenticity/independence states, claim types/states/consequences and edge relations.
12. Recomputes the graph root under SAT's restricted canonicalization profile and compares it with the recorded digest.

The report always includes:
- `graph_integrity_passed`, explicitly scoped to graph semantics and root equality;
- `json_schema_validation_performed=false`;
- `qualification_assessed=false`;
- a computed and recorded graph root;
- a structured issue list;
- a separate semantic-validity result and root-match result.

A passing graph verification is not cryptographic authentication, external-world effect verification, a safety certification or system qualification.

## 3. Canonical graph-root profile

The committed graph payload contains exactly:

```json
{
  "schema_version": "...",
  "augmentation_id": "...",
  "native_benchmark": {},
  "subject": {},
  "run_context": {},
  "observations": [],
  "claims": [],
  "claim_edges": []
}
```

The root is:

```text
SHA-256(UTF8("SAT-EVIDENCE-GRAPH-v0.1\n") || canonical_json(graph_payload))
```

Normalization rules:
- observations sort by `observation_id` using UTF-16 code-unit order;
- claims sort by `claim_id` using UTF-16 code-unit order;
- edges sort by `(from_claim_id, to_claim_id, relation, rule_ref)` using UTF-16 code-unit order;
- `evidence_refs`, `depends_on` and `invalidation_triggers` are unique, UTF-16 sorted sets;
- an omitted optional `invalidation_triggers` field is normalized to an empty array;
- all JSON object keys sort by UTF-16 code-unit order;
- other arrays preserve order;
- only safe integers are allowed; strings are not Unicode-normalized.

This is the explicitly versioned SAT restricted canonicalization profile. It is not represented as complete RFC 8785 conformance.

A Python reference calculation independently produced the following golden vector for the minimal unit-test graph:

```text
sha256:e2b677dd4ad89972b5a306558a4f1fd32680749cfafd9aeb3a25102343851b82
```

The Rust unit test pins this value. The separate UTF-16 test distinguishes surrogate-pair ordering from Unicode scalar-value ordering.

## 4. Dependency semantics

A claim record's `depends_on` array is the claim-level declaration. The `claim_edges` array is the typed graph representation. They must agree exactly:
- every declared dependency has one `depends_on` edge;
- every `depends_on` edge appears in the source claim's array;
- duplicate edges do not count as extra support;
- the directed dependency subgraph must be acyclic.

The verifier uses iterative Kahn topological processing rather than recursive DFS, preventing attacker-controlled graph depth from consuming the call stack.

## 5. Supported-claim admissibility

The reference verifier applies conservative base rules:
- supported claims need current validity and at least one evidence reference;
- evidence with unknown or invalid authenticity cannot support a claim;
- unverified or unsatisfied independence cannot silently support a claim;
- every piece of evidence supporting a critical claim must have verified authenticity and satisfied independence;
- claim-specific evidence kinds must exist for authority, disclosure, effect correspondence, transparency continuity and witness-independence claims;
- an effect-correspondence claim requires an independent sink read-back;
- a supported claim cannot depend on a claim that is not both supported and current.

These are baseline semantic checks. Adapter-specific policies may impose stronger requirements, including stricter source roles, time windows, trust roots, witness cut sets, and domain-specific effect projections. This module does not infer missing requirements from absence of evidence.

## 6. Resource bounds and failure behaviour

- Maximum input: 32 MiB.
- Maximum observations: 100,000.
- Maximum claims: 100,000.
- Maximum claim edges: 250,000.
- Dependency processing is linear in vertices plus edges after parsing.
- Duplicate keys, malformed types, unresolved references, unknown endpoints, duplicate identities, duplicate dependency edges and cycles fail closed.
- Root mismatch is reported separately from semantic graph validity.

These are initial implementation limits, not benchmarked capacity claims. Performance thresholds must be set only after measurements on frozen hardware and graph distributions.

## 7. CLI

From the workspace root:

```bash
cargo test -p symthaea-sat-evidence-graph-verifier
cargo clippy -p symthaea-sat-evidence-graph-verifier --all-targets -- -D warnings
cargo run -p symthaea-sat-evidence-graph-verifier --bin sat-evidence-graph-verify -- augmentation.json
```

To calculate the root for a parseable graph whose recorded root is still a placeholder:

```bash
cargo run -p symthaea-sat-evidence-graph-verifier --bin sat-evidence-graph-verify -- --print-root augmentation.json
```

Exit codes:
- 0: semantic checks pass and recorded root matches;
- 2: malformed input or semantic failure;
- 3: semantics pass but root mismatch;
- 64: usage error;
- 66: input cannot be read.

`--print-root` is a canonicalization aid only. It does not run the semantic verifier.

## 8. Source-authored regression matrix

The current implementation contains 16 unit tests covering:
- minimal valid graph and pinned golden root;
- graph-array order invariance;
- omitted versus explicit-empty invalidation triggers;
- dangling evidence references;
- duplicate observation identities;
- unknown edge endpoints;
- dependency cycles;
- claim-dependency / typed-edge mismatch;
- duplicate dependency edges;
- critical evidence authenticity and independence;
- supported claims with inconclusive dependencies;
- duplicate JSON keys;
- non-integer numeric rejection;
- UTF-16 ordering;
- effect claims without independent sink read-back;
- recorded-root substitution.

The tests are authored but have **not** been compiled or executed in this authoring environment. They must run against the exact PR head before any implementation claim is promoted to executed evidence.

## 9. Benchmark and SOTA programme

The verifier should be compared as a semantic/evidence control, not as an agent task-completion model.

Measure:
- false acceptance across public and sealed graph mutations;
- false rejection across paired benign graph changes;
- dependency-cycle and dangling-reference detection;
- root-recomputation agreement across independent Rust and non-Rust implementations;
- canonicalization disagreement on Unicode, integer, duplicate-key and array-order edge cases;
- supported-claim admissibility confusion matrices;
- P50/P95 latency and peak memory by graph size;
- resource-exhaustion resistance under maximum-size and adversarially shaped graphs;
- deterministic report equality across machines and builds.

Research targets after executable baselines:
- zero critical false-accepts on a predeclared holdout;
- 100% deterministic agreement between independent implementations on public fixtures;
- zero accepted cycles, duplicate IDs, dangling evidence refs or dependency-edge inconsistencies;
- 100% rejection of altered roots;
- benign false-rejection rate no greater than 1% on matched controls;
- P95 latency and memory targets only after frozen-hardware measurement.

Do not combine these into one weighted score. A critical false acceptance must remain visible regardless of the mean.

## 10. Integration boundaries

- The benchmark-augmentation schema remains responsible for versioned local structure.
- The verifier is responsible for cross-record graph semantics and canonical-root equality.
- The trace/effect verifier remains responsible for attempt and effect receipt structure.
- Signature and trust adapters remain responsible for authenticity and revocation.
- The independence evaluator remains responsible for common-cause dependencies and source independence.
- The native benchmark remains authoritative for its native task score.
- The evaluator, not the subject, owns expected identities, policies and qualification decisions.

No action is authorized by this verifier. Unknown or contradictory evidence remains a hold or an adverse finding under the caller's policy.

## 11. Execution status

Source and documentation are committed to the PR branch. The authoring environment has no Cargo, rustc or rustfmt, so no compile, unit test, Clippy, executable CLI run, performance benchmark or independent Rust replay is claimed. A skipped CI job is not a pass.
