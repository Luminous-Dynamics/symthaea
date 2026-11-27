# MON-027 — Evidence Bundle Assembly and External Signing

**Status:** source for deterministic artifact assembly, signing-request generation and signed-manifest attachment added; exact-head Go execution, KMS/HSM adapter, MON-025 automation and independent custody qualification pending.

MON-027 joins the MON-025 lab and MON-026 custody verifier without placing signing keys in the collector.

## Pipeline

```text
MON-025 raw outputs
  | allowlisted artifact index
  v
MON-027 assembler
  | hash exact bytes, enforce exact inventory
  v
unsigned MON-026 manifest + signing request
  |
  | detached signing request crosses trust boundary
  v
external Ed25519 signer / HSM / KMS adapter
  |
  v
detached signature envelope
  |
  v
MON-027 attach + verify
  |
  v
signed manifest + MON-026 offline verification
```

The assembler never receives a private key. The external signer signs the exact domain-separated message for the manifest root and key identity. The attach step checks key ID, algorithm, signature and all artifact bytes against an out-of-band expected run ID, sequence, predecessor root and trusted public key.

## Contents

- `bundle/assemble.go` — allowlisted input index, exact directory inventory, path/symlink checks, streaming SHA-256 and manifest/signing-request construction.
- `bundle/signature.go` — detached signature envelope parser and signature attachment.
- `cmd/assemble` — CLI to create an unsigned manifest and external signing request.
- `cmd/attach-signature` — CLI to attach a signer response and independently verify it before writing the signed manifest.
- `INPUT-v1.schema.json` and `SIGNING-REQUEST-v1.schema.json` — strict input and detached signing request formats.
- `SCENARIOS-v1.json` — assembly/signing-boundary adversarial cases.
- `BENCHMARKS-v0.1.md` — assembly cost, signing latency and fault-detection targets.

## Trust boundary

The artifact index is an allowlist, not an independent truth source. Its entries must be derived from MON-025 run evidence and independently reviewed as part of the run. A signature authenticates the exact manifest; it does not validate PostgreSQL truth, CDC completeness, restore correctness or service safety.

The assembler and collector are not authorized to sign. The signer must independently validate the signing request's authorization and use a private key held outside the lab. A production KMS/HSM adapter and key lifecycle are not included.

## Execution status

No MON-025 run has been executed in this authoring environment, no signer/HSM/KMS service is connected, and no Go test/race/vet/fuzz/benchmark has been run on this branch. The assembler is source implementation only; downstream verification remains independently required.
