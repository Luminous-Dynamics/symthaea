# Financial Research Semantic Verifier v0.1

**Status:** focused implementation draft under #14203. Rust source and mutation tests are committed; local compilation is unavailable in the current authoring environment. The exact-head CI workflow is the next execution gate.

## Purpose

The crate consumes the v1 task manifest, evidence manifest and claim-level audit from the adjacent `financial-agent-v1` contract. It performs offline JSON Schema validation and a separate semantic pass over task/evidence/claim relationships.

It does not create a task schema, evidence store, portfolio ledger, broker, source registry, authority model, generic receipt system, or LLM judge.

## Verification vector

The report preserves independent outcomes:

- `schema`: committed Draft 2020-12 task, evidence-manifest and claim-audit schemas;
- `manifest_integrity`: SHA-256 of the exact evidence-manifest bytes bound by the task;
- `payload_integrity`: RFC 8785 JCS digest for JSON payloads and exact-byte SHA-256 for non-JSON payloads;
- `temporal`: typed evidence dispositions supplied by the shared #11253 temporal kernel adapter;
- `semantic`: task/manifest closure, evidence identity, source snapshot/family binding, subject mapping and derivation graph;
- `candidate_claims`: independent claim fact-assessment, trace-support and hard-gate outcomes.

The vector is normative. `overall()` is only a conservative exit-status helper; no dimension compensates for another.

## Important composition boundaries

### Temporal truth

This crate does **not** parse or decide `known_at`, `published_at`, `retrieved_at`, validity intervals, revision ancestry, date-only precision, rights or source availability. It consumes typed results from the shared #11253 kernel.

The v1 evidence schema currently contains a `known_at` string and a `rights_profile` label, but no principal-scoped availability or rights-proof references. That gap is owned by #11264. A v1 `known_at` value by itself must therefore not be promoted to temporal PASS for historical confirmation or prospective shadow. Legacy evidence without the proof-bound bridge remains INDETERMINATE for qualified temporal use.

The `temporally_admissible` Boolean in a claim audit is untrusted candidate/audit metadata. The verifier checks it for consistency with the typed temporal-kernel result; it never uses that Boolean as the admission source.

### Artifact bytes

Inline JSON payloads in the v1 development fixture resolve only inside their own evidence-manifest artifact entry. They are canonicalized with the pinned RFC 8785 implementation before hashing. External CAS/object payloads require an explicit bounded `PayloadResolver`; this crate has no URL, HTTP or filesystem auto-fetch path. The resolver caller remains responsible for an immutable allowlist and locator-to-artifact binding.

The shared artifact-byte receipt work tracked by #9608/#9429 remains the canonical integration destination. This crate's simple task-manifest and payload digest checks do not create another artifact store or receipt format; future integration should use the canonical receipt for external artifact bundles.

### Source independence and identity

The verifier checks relation family IDs against the manifest's `provenance_family_id` and reports distinct declared family counts. Those strings do not prove genuine source independence or ancestry. Identity/rights/entitlement assurance remains with Mycelix and the proof adapter.

### Claim and action boundaries

Fact assessment and trace support remain separate. A true-but-uncited claim is not converted into a supported trace. A supported fact requires an admissible relation of the correct polarity. Mixed/contested, insufficient, stale and not-evaluable dispositions remain distinct.

The verifier does not grant action authority. Proposal-versus-authorization and provider write controls remain owned by the Mycelix authority/effect contracts; no broker, order or provider write surface is added here. Financial research quality is not alpha, investment advice, portfolio return, or permission to act.

## Determinism and safety

- JSON object duplicate keys are rejected before conversion to `serde_json::Value`.
- JSON Schema references are resolved from a local in-memory registry in offline mode; remote/file schema retrieval is disabled.
- JSON payload digesting uses RFC 8785 JCS. Integers outside the IEEE-754 safe integer range are rejected at the strict parser boundary and should be encoded as strings.
- External payload resolution is opt-in and bounded to 16 MiB per payload.
- Claim audit input is bounded to 8,192 claims; derivation-cycle analysis is iterative rather than recursive.
- Inline claim locators must resolve within the exact evidence payload; external-format locators are not silently accepted.
- Findings are sorted deterministically by gate, path, code, disposition and message.
- Unknown or missing temporal and external-payload inputs remain INDETERMINATE, not PASS.

## Run

```sh
cargo fmt --manifest-path benchmarks/financial-agent-verifier/Cargo.toml --check
cargo test --manifest-path benchmarks/financial-agent-verifier/Cargo.toml
```

The package uses exact top-level versions for the JSON Schema validator and JCS canonicalizer. A generated Cargo.lock and successful Rust 1.96 build/test result are still required before qualification; the current authoring environment has no Rust toolchain.

## Claim ceiling

A passing synthetic test establishes only that this verifier applies the implemented schema/closure/digest/claim-reference rules to the supplied fixtures and trusted typed temporal result. It does not establish source truth, legal rights, genuine source independence, investment performance, causal validity or financial action authority.
