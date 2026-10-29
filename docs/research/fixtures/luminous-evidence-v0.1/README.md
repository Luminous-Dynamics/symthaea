# Luminous Evidence Profile v0.1 fixtures

These fixtures exercise only the **predicate schema** at `schemas/luminous-evidence-predicate-v0.1.schema.json`. For each full in-toto Statement, validate its `predicate` member against that schema; use a separate in-toto validator for the outer Statement and any signature/envelope.

- `valid-passed.json`: expected schema acceptance. Values and digests are synthetic placeholders; this is not a real run receipt.
- `invalid-queued-but-passed.json`: expected schema rejection. A queued run cannot carry a passed execution result or passed verification.
- `invalid-missing-target.json`: expected schema rejection. The required target identity is omitted.
- `semantic-stale-target.json`: expected schema acceptance but semantic rejection (`LEP_TARGET_MISMATCH`); the declared target and checked target use different source revisions.
- `conformance-cases.json`: machine-readable expected outcomes, including the semantic case that deliberately remains unimplemented.

Schema validation is not semantic verification. A semantic checker now exists in `tools/lep-conformance`; it compares `target` with `verification.checkedTarget` and checks status, digest-shape, and recorded-check consistency. Its Rust unit tests are checked in but have not been run in this environment because Rust tooling is unavailable. A production verifier must still resolve and re-hash referenced artifacts, verify signatures and signer authority, and independently establish whether the named checks actually ran. Neither JSON Schema nor this initial semantic checker proves the scientific claim true.

The initial local structural check used Python `jsonschema` 4.26.0 with Draft 2020-12 and a format checker. This is a development check only; CI and cross-implementation conformance remain pending. Validators must enable format checking explicitly where the implementation treats `format` as annotation-only.
