# EQC Trusted Observer Protocol v0.1

**Status:** Verification prototype on a feature branch; exact-head Rust validation is still required. Not a production attestation service, certification, or proof of physical independence.

**Tracking:** [Issue #14159](https://github.com/Luminous-Dynamics/symthaea/issues/14159).  
**Base evaluator:** [EQC v0.1 standalone PR #14158](https://github.com/Luminous-Dynamics/symthaea/pull/14158).

## 1. Security boundary

EQC v0.1 deliberately separates two propositions:

1. `contract_checks_pass`: the adapter-submitted observation is structurally and internally consistent with the frozen task.
2. `independent_evidence_verified`: a separately provisioned observer key signed a receipt that binds the task, run, challenge, exact observation bytes, system identity, state/effect evidence, resource measurements and replay result.

An adapter must never be allowed to set its own trust root, expected challenge, trust-store digest, verifier policy or qualification bit. The evaluator operator supplies those values out of band. An adapter's `verifier.independent=true` is only a claim; it does not verify independence.

A verified observer receipt is necessary but not sufficient. The seven contract dimensions and resource-budget gate remain conjunctive. If the observer signature is valid but any task dimension fails, the report records `independent_evidence_verified=true` and `qualified=false`.

## 2. Exact-byte signed payload

The signed payload is carried as lowercase hexadecimal in `eqc-observer-envelope-v0.1.schema.json`. The verifier:

1. decodes the exact payload bytes;
2. checks the envelope's SHA-256 against those bytes;
3. validates the parsed payload against the offline Draft 2020-12 payload schema;
4. prepends the fixed domain-separation bytes `LUMINOUS-EQC-OBSERVER-ATTESTATION-V0.1\\0`;
5. verifies the Ed25519 signature with `VerifyingKey::verify_strict`.

The verifier does not parse and re-serialize the JSON to verify the signature. This avoids signature ambiguity from whitespace, member order or numeric rendering. A future interoperable producer may use RFC 8785 JCS, but that canonicalization profile and its test vectors must be qualified independently before it is used for signing.

The signed payload binds:
- task ID, run ID, one-use challenge, issue and expiry time;
- SHA-256 of the exact frozen task-corpus bytes and exact adapter-observation bytes;
- observer ID, key ID, observer binary digest and observer environment digest;
- actor, subject, authorization decision, system source/environment and authority snapshot digests;
- input digests, state-before/state-after commitments, mutation count and conditional effect witness;
- resource usage, network bytes and endpoint observations;
- replay status, verifier identity/digest and replay-result digest.

## 3. Trust store and key lifecycle

`eqc-observer-trust-store-v0.1.schema.json` is the evaluator-operator trust store. Each key entry binds:
- key ID and observer identity;
- Ed25519 public key;
- approved observer binary and environment digests;
- allowed task domains;
- active/revoked/disabled status and validity window;
- optional revocation timestamp.

The verifier receives both the trust-store bytes and an **expected trust-store SHA-256** from a separate operator-controlled channel. It refuses a mismatch and refuses missing, duplicate, disabled, revoked, expired, not-yet-valid or out-of-scope key entries. A test key created inside unit tests is never a production trust anchor.

A signature proves that the corresponding private key signed the exact bytes. It does not prove the key was uncompromised, that the observer software was honest, or that the observation was physically independent. Those remain trust-profile and operational qualification questions.

## 4. Challenge and replay resistance

The evaluator/harness supplies a fresh 256-bit challenge nonce bound to one task and one run. The observer payload must echo it exactly. The verifier also checks:
- task ID and run ID;
- frozen corpus digest;
- exact observation digest;
- issue, expiry and verification times;
- observer key validity/revocation state.

After every schema, binding, trust and signature check succeeds, the verifier atomically creates a `<sha256(challenge)>.consumed` marker with create-new semantics, writes the run/task/payload binding, fsyncs the marker and fsyncs the ledger directory. Duplicate consumption fails.

The current ledger is a local filesystem prototype. It supports atomic one-use creation on the single-host filesystem used by the focused Ubuntu CI lane; it is **not** a distributed multi-host challenge-consumption service. Production multi-host use requires a durable transactional store or consensus-backed single-use primitive with explicit crash/recovery qualification.

Verification failures before the final consume step do not create a marker. A process or filesystem failure during marker creation can leave a partial marker; this is fail-closed and requires explicit reconciliation rather than automatic reuse.

## 5. Resource and state evidence

The observer receipt repeats the adapter observation's input digests and resource usage and must match them exactly. It also binds:
- state-before and state-after SHA-256 commitments;
- actual mutation count;
- an effect-witness digest when state changed;
- zero-mutation equality when no state change occurred;
- successful pristine replay with exact verifier identity and result digest.

Energy and monetary cost may be null when the observer has no qualified meter. Missing measurements are not coerced to zero. A future qualification profile should state meter calibration, sampling interval, isolation and uncertainty, and should not compare unlike hardware meters as though they were equivalent.

## 6. CLI invocation

The adapter observation alone remains unqualified:

```bash
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \\
  --corpus docs/research/eqc-seed-corpus-v0.1.json \\
  --task-id eqc-health-wrong-subject-deny-001 \\
  --observation run/observation.json \\
  --out run/contract-only-report.json
```

Independent verification requires all evaluator-supplied inputs:

```bash
cargo run --locked --manifest-path tools/eqc-reference/Cargo.toml -- \\
  --corpus docs/research/eqc-seed-corpus-v0.1.json \\
  --task-id eqc-health-wrong-subject-deny-001 \\
  --observation run/observation.json \\
  --observer-envelope run/observer-envelope.json \\
  --observer-trust-store operator/trust-store.json \\
  --trust-store-sha256 OPERATOR_PINNED_TRUST_STORE_SHA256 \\
  --run-id RUN_ID_FROM_HARNESS \\
  --challenge-nonce 64_HEX_CHARACTERS_FROM_HARNESS \\
  --verification-time 2026-11-28T12:00:00Z \\
  --challenge-ledger /var/lib/eqc/consumed-challenges \\
  --out run/qualification-report.json
```

The trust-store pin, challenge and verification time must be provided by the evaluator/operator outside the adapter. Do not populate them from adapter-submitted fields.

## 7. Threat model and limitations

- **Out of scope for v0.1:** TEE/TPM remote attestation, measured boot, hardware roots, protected key storage, network-consensus replay ledgers, external timestamp authorities, and a production observer daemon.
- **Shared-host risk:** an observer sharing the SUT's kernel, root account, hypervisor, storage controller or physical host may share common-cause compromise. Separate process and separate signing key are not enough to claim physical independence.
- **Observer honesty:** the verifier checks signatures and bindings; it cannot prove that a privileged observer measured the real world truthfully. Observer binary/configuration provenance, custody and independent review remain necessary.
- **Measurement quality:** reported resource numbers require a meter profile, calibration and uncertainty before they can support cross-system performance rankings.
- **Time:** the CLI currently accepts evaluator-supplied UTC verification time. A production harness must bind this to a trusted clock/time service and preserve that evidence.
- **Local ledger durability:** the `create_new` + fsync implementation is a single-host prototype and must not be represented as a distributed anti-replay guarantee.

## 8. Required adversarial tests

The focused Rust test suite covers valid exact-byte signatures, duplicate challenge consumption, wrong challenge with no consumption, payload tampering, wrong trust key, expiry and revocation. Before claiming the observer verifier is qualified, the matrix must also cover malformed signature encodings, future key validity, duplicate key IDs, wrong domain scope, wrong run/task/corpus/observation digest, observer binary/environment mismatch, state/effect-witness inconsistency, replay mismatch, ledger symlink/permission failure, partial marker write/restart and concurrent duplicate consumption.

## References

- [ed25519-dalek 3.0.0](https://docs.rs/ed25519-dalek/3.0.0/) and its strict verification API. The crate declares Rust 1.85 as its MSRV; the focused workflow uses the repository's pinned Rust 1.96.0 toolchain.
- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html), considered but not used for the v0.1 signature path; exact payload bytes are signed instead.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), as governance context rather than a technical attestation.
- [ASME VVUQ resources](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty), for domain-specific physical validation and measurement uncertainty.
