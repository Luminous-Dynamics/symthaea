# symthaea-rea-compat

A Rust-native compatibility core for the provider-neutral REA Evidence envelope.

## Scope of this first slice

This package implements:
- strict admission of the generic REA Evidence envelope;
- validation of the analysis-profile digest and provider binding;
- the semantic projection used by REA's Evidence ID;
- RFC 8785 JSON Canonicalization Scheme (JCS) serialization;
- computation and verification of `ev_<sha256>` semantic identifiers;
- separate SHA-256 identity for the exact raw JSON file;
- duplicate-key rejection and a 64 MiB bound for raw-file ingestion.

It intentionally does **not** validate operation-specific `normalized_result` schemas, Evidence bundles, residual-unknown graphs, provider signatures, device attestation, or whether an observation actually occurred.

## Use from Symthaea

Enable the optional root-crate feature:

```toml
symthaea = { version = "2", features = ["rea-compat"] }
```

Then call the direct Rust API; MCP is not required for in-process use:

```rust,ignore
use symthaea::rea_compat::verify_evidence_json;

let input = std::fs::read("evidence.json")?;
let verified = verify_evidence_json(&input)?;
println!("semantic Evidence ID: {}", verified.evidence_id);
println!("raw file SHA-256: {:?}", verified.raw_file_sha256);
```

You can also call the crate directly using `symthaea_rea_compat::verify_evidence_value` when the input is already a parsed `serde_json::Value`.

## Identity contract

The semantic projection includes:
- subject artifact digest, format and architecture (not display name or local path);
- provider and analysis-profile commitment;
- predicate type, operation and parameters;
- raw and normalized results;
- confidence, authority and environment;
- limitations and relationship coverage;
- source locations and Evidence links.

It excludes:
- `subject.name`;
- `subject.local_path`;
- `caller_input_path`.

Therefore, changing a display path must not change the Evidence ID. Changing a semantic field must change it. The raw-file digest is always a different identity over the exact input bytes.

The canonicalizer dependency is pinned through the workspace to `serde_json_canonicalizer = 0.3.2`. The adapter wraps serde_json values in an ECMAScript number serializer so integer tokens retained by Rust are converted to IEEE-754 binary64 before JCS serialization. This is necessary because JavaScript JSON.parse converts every JSON number to a double before JSON.stringify: for example, 9007199254740993 becomes 9007199254740992. The JCS serializer sorts object property names by UTF-16 code units and applies ECMAScript-compatible number formatting. The dedicated CI workflow executes pinned upstream REA 6.8.0 source and generates a bounded differential corpus: 9 valid records, display-only and key-order metamorphisms, 6 upstream-rejected negative controls, and an explicit isolated-surrogate importer boundary. This is a first cross-language matrix, not exhaustive schema/property-fuzz parity; operation-specific normalized-result schemas remain outside the single-record verifier. The optional bundle module is described below and has its own upstream differential fixture path.

## Raw-file admission versus parsed-object parity

Two entry points deliberately have different purposes:

- `verify_evidence_value(&Value)` validates an already-parsed object and verifies its semantic ID. This is the appropriate entry point for object-level disposition comparisons with upstream REA.
- `verify_evidence_json(&[u8])` additionally rejects duplicate object keys and files larger than 64 MiB. This is a stricter native file-ingestion policy. JavaScript `JSON.parse` resolves duplicate object keys before REA's Zod schema sees the object; this Rust importer refuses that ambiguous raw representation instead.

The file importer currently uses Serde JSON's default recursion guard. Inputs rejected by size, duplicate-key or nesting limits are explicit import failures, not successful or complete Evidence.

Rust strings represent Unicode scalar values and cannot represent isolated UTF-16 surrogate code units. JavaScript strings can contain isolated surrogates, and JSON.parse/JSON.stringify can preserve them using escaped surrogate code units. The current Rust raw importer therefore rejects such inputs rather than pretending to provide full string-domain parity. The upstream-generated corpus explicitly demonstrates that REA accepts an object containing an isolated surrogate while the Rust raw importer rejects its escaped JSON representation as invalid. This is a documented stricter-ingestion boundary; do not normalize, replace or silently drop those code units.

## Validation

The focused, standalone harness compiles the exact crate source and integration test while avoiding unrelated root-workspace glob entries that currently lack Cargo manifests:

```bash
cargo fmt --manifest-path tools/rea-evidence-compat-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-evidence-compat-harness/Cargo.toml --locked
```

The harness has its own minimal lockfile and names the same library package. It does **not** establish full Symthaea root-workspace metadata or root feature integration. Those remain separate repository-level checks.

The committed unit fixture is a source-derived reference vector; its semantic digest and profile digest were independently recomputed using ECMAScript `JSON.stringify` behavior and the pinned REA field projection. It is **not** described as a fixture generated by executing upstream REA. Separately, the dedicated CI workflow executes upstream REA at commit `8250bf2320ea57aab4e664db4b4d44b8abd2004e` under Node 22.23.4 / npm 11.16.0 and generates a 9-record single-Evidence corpus, display-only and key-order metamorphisms, six upstream-rejected controls and one stricter importer case. A second integration test now consumes an upstream-generated EvidenceBundle with revisioned residual unknowns, deterministic manifests, evidence references, dependencies and verified resolution, plus duplicate-record, noncanonical-order, missing-reference, history-gap and dependency-cycle controls. The bundle workflow is not yet run; no Bundle parity pass is claimed until the exact stacked PR head succeeds. Broader property-based/fuzz coverage and operation-specific normalized-result schemas remain future work.

## Claim ceiling

A matching Evidence ID proves that the supplied object is internally consistent with this implementation's semantic projection and canonicalization. It does not prove producer authorship, provenance authenticity, observation truth, hardware identity, runtime behavior, or security compliance. Use HARDSEC and independently trusted signed/anchored receipts for those claims.
