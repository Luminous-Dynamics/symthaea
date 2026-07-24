# Fabrication Secure Release Boundary

Version 0.10.0 adds a cryptographic and session-scoped release layer above the
v0.9.0 qualification and deterministic-provenance pipeline.

## Authority chain

```text
ManufacturingReadyMesh
  -> exact slices and toolpath
  -> ValidatedGCode for one exact MachineProfile
  -> FabricationManifest
  -> SHA-256 manifest digest
  -> detached external signatures
  -> VerifiedAttestation under an explicit trust policy
  -> MachineCapabilities + fresh MachineSession nonce
  -> NegotiatedMachine
  -> single-use AuthorizedPrintJob
  -> submission receipt
  -> ExecutionGuard containment decisions
```

Each stage retains the values that granted it authority. A same-name profile,
stale session, changed G-code program, changed manifest, insufficient signature
policy, or endpoint capability downgrade cannot silently cross the trusted path.

## Cryptographic scope

`Sha256Digest` is a real SHA-256 implementation tested against standard vectors.
It supplies tamper-evident byte identity, not signer identity. Private keys remain
outside this crate behind `ManifestSigner` and `ManifestSignatureVerifier`.
Supported algorithm labels include Ed25519, ML-DSA-65, and ML-DSA-87, but this
crate does not claim to implement those signature algorithms.

`AttestationPolicy` can require a minimum number of distinct valid signatures,
required algorithms, and an allowlist of key identifiers. Only a completely
clean report becomes `VerifiedAttestation`.

## Package verification

Attested 3MF packages contain:

- `/3D/3dmodel.model`
- `/Metadata/fabrication-manifest.json`
- `/Metadata/fabrication-manifest.sha256`
- `/Metadata/fabrication-attestation.json`
- OPC content types and relationships

The bounded inspector accepts the stored-entry ZIP profile emitted by this
crate. It rejects unsupported compression/flags, unsafe paths, duplicates,
truncation, CRC drift, package and entry budget overruns, malformed metadata,
manifest-copy drift, and digest drift.

It is not a general-purpose ZIP implementation and intentionally fails closed on
ZIP features that the exporter does not generate.

## Machine-session scope

A `MachineProfile` is a requested policy. `MachineCapabilities` are advertised
limits from a concrete endpoint. `negotiate_machine_profile` grants authority
only when every requested bound fits inside those capabilities and the session
has a non-empty freshness nonce.

`AuthorizedPrintJob` binds:

- exact validated G-code;
- exact manifest and valid signature policy;
- exact machine profile;
- machine identity;
- session nonce.

The type is not cloneable, and submission consumes it. This is an API-level
single-use property; persistent anti-replay across process restarts still
requires the machine gateway to store consumed nonces or job identities.

## Runtime containment

`ExecutionGuard` deterministically maps telemetry failures to monotonically
latched actions:

- thermal deviation or progress stall -> pause;
- stale heartbeat or regressed progress -> cancel;
- non-finite telemetry, regressed time, or absolute overtemperature -> emergency stop.

The guard produces decisions only. A real machine gateway must connect those
decisions to independently tested pause, cancel, power-cut, or emergency-stop
actuators.

## Replay contracts

`FabricationReplayContract` binds the manifest digest to a source revision,
kernel version, target triple, Rust compiler identity, Cargo lock digest,
feature set, deterministic seed, and explicit algorithm-version inventory.
This makes environment drift visible. It does not prove bit-identical replay by
itself; CI must rerun the pipeline and compare resulting manifest and contract
digests.

## Remaining physical authority requirements

Before real hardware authority:

1. Run formatting, compilation, tests, Clippy, and documentation builds in the
   canonical Symthaea workspace.
2. Integrate a reviewed cryptographic provider and protected key service.
3. Authenticate machine capability advertisements and session nonces.
4. Persist anti-replay state at the machine gateway.
5. Integrate and fault-inject pause/cancel/emergency-stop actuators.
6. Validate output on sacrificial hardware under supervised operating limits.


## Successor governance layer

Version 0.11.0 adds revocation-aware trust snapshots, monotonic snapshot
tracking, hash-chained audit evidence, durable execution checkpoints, governed
3MF packages, and explicit fresh-session restart re-authorization. See
`FABRICATION_GOVERNANCE_AND_RECOVERY.md`.
