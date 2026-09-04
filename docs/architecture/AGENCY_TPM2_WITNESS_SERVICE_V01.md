# Agency TPM2 witness service V0.1

## Status

Authored design and tests. **Not qualified yet.** This is an evidence/notarization service boundary only. It does not grant or exercise runtime authority.

## Problem

`#439` intentionally contains pure witness primitives, including a deterministic function that can sign a parsed `QualificationAcceptanceV1`. That is useful for composition and testing, but it is the wrong remote operational API for an independently administered witness.

A witness service must not behave like:

```text
caller supplies acceptance JSON
        ↓
witness key signs it
```

Even when the JSON schema is strict, that would make the key holder responsible for remembering to run the independent verifier first.

The production-shaped boundary is:

```text
archive
+
independently obtained archive SHA / Git HEAD / Git tree
        ↓
exact reviewed evidence-verifier runtime
        ↓
release-bound verifier stdout
        ↓
strict #439 acceptance parser
        ↓
exact binding re-check
        ↓
witness key
        ↓
attestation
```

There is no public acceptance-JSON argument in `verify_archive_then_sign_v1`.

## Exact verifier runtime identity

`QualificationVerifierRuntimePolicyV1` commits:

- runtime-policy ID;
- canonical absolute Python executable path;
- domain-separated BLAKE3 of exact Python executable bytes;
- canonical absolute evidence-verifier script path;
- domain-separated BLAKE3 of exact script bytes;
- whether both paths must be beneath `/nix/store`;
- runtime timeout;
- stdout/stderr bounds;
- fixed invocation profile;
- exact acceptance schema.

The resulting `implementation_digest()` is the verifier digest enrolled by `QualificationWitnessPolicyV1`.

Changing a timeout, output limit, interpreter, verifier script, invocation profile, Nix-store requirement, or acceptance schema therefore changes the witnessable verifier identity.

## Production Nix profile

`nix/agency-tpm2-evidence-verifier-runtime.nix` returns two immutable paths derived from locked nixpkgs and the exact repository verifier source:

- the real `${pkgs.python3}/bin/python3`;
- a `pkgs.writeText` copy of `verify-tpm2-qualification-evidence.py`.

No shell wrapper or mutable `PATH` lookup is required. Production policy should set `require_nix_store_paths = true`.

The Rust service canonicalizes both paths, requires regular files, measures them before execution, and measures them again after verifier completion.

## Process boundary

The system runner invokes:

```text
<exact-python> -I -B <exact-verifier> <archive>
    --release
    --expected-archive-sha256 <external>
    --expected-head <external>
    --expected-tree <external>
```

with:

- `env_clear()`;
- current directory `/`;
- stdin closed;
- bounded stdout;
- bounded stderr;
- hard runtime timeout.

`-I` places Python in isolated mode and `-B` prevents bytecode cache writes.

A successful process that writes anything to stderr is not signable. A timeout, output flood, nonzero verifier exit, malformed acceptance, changed runtime, or external-binding mismatch produces no witness signature.

Verifier failure stdout/stderr are represented by commitments in the Rust error rather than returned as arbitrary durable operational payloads.

## Archive semantics

This service inherits the corrected #431 verifier, where the release archive is opened once with `O_NOFOLLOW`, snapshotted as a regular bounded file, hashed, and semantically parsed from those exact same bytes. The gzip expansion is bounded before tar interpretation.

The service therefore does not create a second archive parser. It delegates evidence semantics to the independently reviewed #431 boundary and then verifies that the parsed acceptance still equals its external release bindings.

## Key custody

The current library API receives an already-instantiated `ed25519_dalek::SigningKey`. It defines no seed-file format, environment-variable key, development key, or persistence mechanism.

A production daemon should put key custody behind Xenia, TPM/HSM signing, or another separately administered signing provider. The critical operational rule is that remote callers receive only a **verify-then-sign** endpoint, never the lower-level generic signing primitive.

## Witness sequence

As in #439, `witness_sequence` is signed but durable monotonic storage remains outside these pure crates. A production witness daemon must allocate/increment the sequence in its own durable domain before releasing an attestation.

A later service tranche should make that ordering executable:

```text
verify release evidence
        ↓
reserve next witness sequence durably
        ↓
sign exact acceptance
        ↓
persist attestation / publication intent
        ↓
release attestation
```

A crash must not permit sequence rollback or duplicate release under ambiguous state.

## Tests authored

The V0.1 unit suite uses a private fake verifier runner while retaining the real runtime measurement/signing logic. It covers:

- verifier-owned release acceptance can be signed;
- acceptance that disagrees with external Git/tree/archive bindings cannot be signed;
- verifier script mutation during execution prevents signing;
- success-with-stderr prevents signing;
- production Nix-store policy rejects a fixture runtime before the runner is invoked.

The real Python/#431 process path still needs a Nix qualification lane before production claims.

## Non-claims

This layer does not prove:

- the qualification itself passed until the real verifier/capsule lanes run;
- physical TPM security;
- measured-boot/IMA correctness;
- uncompromised kernel/root state;
- witness key hardware protection;
- durable witness sequence monotonicity;
- publication/transparency inclusion;
- any right for Symthaea to execute an effect.

The output remains evidence only.
