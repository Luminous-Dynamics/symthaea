# Workbench Nix Closure Capture Verifier v1

Status: **candidate independent receipt-verification theorem; no Workbench execution or scientific transform is qualified by this verifier**

Verifier output schema: `symthaea-workbench-nix-closure-capture-verification-v1`

Receipt input schema: `symthaea-workbench-nix-closure-capture-receipt-v1`

## Purpose

The producer defined by `WORKBENCH_NIX_CLOSURE_CAPTURE_V1.md` can observe a real Nix realization and emit a receipt. A producer must not be allowed to establish the correctness of its own receipt merely by re-reading the values it wrote.

This verifier therefore treats every receipt field and every retained sidecar as hostile input and independently reconstructs the admissible facts.

The v1 theorem is:

```text
hostile receipt directory
    + exact #624 execution-capsule profile
    + exact root flake.lock
    + exact producer source bytes
    + exact #629 normalizer source bytes
        -> independent closed-world reconstruction
        -> verified receipt observation
```

with the critical boundary:

```text
verified receipt observation
    != closure execution qualification
    != Workbench scientific execution
    != atlas correctness
    != FMQ-010
    != neural alignment
    != consciousness evidence
```

## Verification success does not mean observation success

The verifier recognizes two valid receipt outcomes:

```text
observed-normalized-unqualified
    -> verified-complete-observation

observation-incomplete-unqualified
    -> verified-incomplete-observation
```

A failed Nix realization may therefore still produce a **verified incomplete observation** when its retained bytes and failure semantics are internally authentic and exactly bound.

This distinction is intentional:

```text
receipt is verified
    !=
underlying observation succeeded
```

The verifier never repairs an incomplete producer receipt and never manufactures missing normalized state.

## Independence boundary

The verifier:

- does not import `workbench_nix_closure_capture.py`;
- does not invoke Nix;
- does not query `/nix/store`;
- does not execute Workbench;
- does not use the producer's parsed facts as authoritative inputs;
- independently derives the selected installable from the profile and `flake.lock`;
- independently reconstructs command contracts and normalized closure identity from retained bytes.

The only shared semantic component is the already-qualified pure #629 closure identity normalizer. The verifier requires the `normalizer_path` whose digest is bound into the receipt to be the exact module actually imported for reconstruction:

```text
hash implementation A + execute implementation B -> reject
```

## Exact selection reconstruction

The verifier independently re-derives:

```text
profile root_input
+ profile locked_node/revision/NAR
+ flake.lock root input binding
+ selected GitHub owner/repository/revision
+ profile Workbench attribute
    -> exact installable
```

For v1 the platform remains exactly:

```text
x86_64-linux
```

Selection facts in the receipt must equal this reconstruction exactly. A self-rehashed profile digest, revision, node, NAR, attribute, installable, or platform substitution is rejected.

## Exact command reconstruction

The verifier fixes the four producer commands independently:

```text
nix --version

nix eval --raw --impure --expr builtins.currentSystem

nix build --no-link --print-out-paths --no-write-lock-file <derived-installable>

nix path-info --json --json-format 2 --recursive <root-derived-from-realization-stdout>
```

The path-info command is permitted only when realization stdout yields exactly one canonical Nix store root.

Every command record has a closed-world schema and must retain:

- exact argv;
- exact integer exit status (booleans are rejected);
- fixed stdout sidecar path;
- fixed stderr sidecar path;
- exact byte length;
- exact SHA-256 of retained bytes.

## Root derivation

The root is not trusted from `receipt.json`.

It is reconstructed from the retained realization stdout:

```text
Root_verified = ParseCanonicalSingleStorePath(raw/realization.stdout)
```

and then compared with the receipt's root field.

Therefore:

```text
receipt root substitution -> reject
caller-selected alternative root -> reject
multiple realized roots -> incomplete/reject according to retained producer semantics
```

## Nix protocol interpretation

v1 requires exact Nix version:

```text
2.33.6
```

and exact path-info JSON protocol:

```text
version = 2
storeDir = /nix/store
info = non-empty object
```

For each store object, the verifier independently consumes only:

```text
narHash
references
```

NAR hashes must be canonical 32-byte SHA-256 SRI values. They are independently translated into #629's canonical representation:

```text
sha256:<64 lowercase hex>
```

Other raw Nix metadata remains part of retained observation bytes but does not silently enter the normalized closure projection.

Therefore:

```text
RawObservationRoot != NormalizedClosureRoot
```

remains preserved by verification.

## #629 reconstruction

The verifier reconstructs #629 entries from raw path-info bytes and invokes the exact bound normalizer.

The reconstructed closure must satisfy all #629 invariants, including:

```text
SuppliedEntries = Reachable(root)
```

The retained `normalized/closure_identity.json` must then be byte-for-byte the canonical JSON representation of that independently reconstructed identity, including its final newline.

A semantically equivalent but differently serialized identity is not the same retained artifact.

## Canonical receipt bytes

`receipt.json` itself must be exactly:

```text
CanonicalJSON(receipt) + LF
```

This closes representation ambiguity where a receipt could be reparsed/reformatted after publication while continuing to claim the same producer artifact.

Duplicate JSON keys are rejected before semantic interpretation.

## Filesystem containment

The receipt directory is a closed evidence object.

Sidecar paths are fixed by the verifier and must be:

```text
raw/nix-version.stdout
raw/nix-version.stderr
raw/platform.stdout
raw/platform.stderr
raw/realization.stdout
raw/realization.stderr
raw/path-info.stdout       # only when path-info was observed
raw/path-info.stderr       # only when path-info was observed
normalized/closure_identity.json  # only for complete normalized observations
receipt.json
```

The verifier rejects:

- absolute sidecar paths;
- `.` / `..` traversal;
- missing files;
- symlinks;
- symlink escape;
- non-regular files;
- extra unbound files.

The receipt cannot redirect the verifier toward arbitrary host files.

## Fact reconstruction

Receipt `facts` are assertions to be checked, not authority inputs.

The verifier independently derives:

```text
selection_revalidated
nix_version_observed
platform_observed
realization_command_observed
realized_root_observed
path_info_observed
canonical_closure_identity_compiled
```

and requires exact equality with the retained fact object.

A self-rehashed fact mutation remains invalid.

## Digest reconstruction

The verifier independently recomputes:

```text
raw_observation_digest =
    SHA256(CanonicalJSON(commands))

capture_digest =
    SHA256(CanonicalJSON(receipt excluding capture_digest))
```

For successful normalized observations it additionally reconstructs and checks:

```text
closure_identity_sha256
closure_digest
```

Rehashing modified semantics does not restore authority.

## Authority boundary

Every authority field in the producer receipt must remain false:

```text
closure_capture_qualified          false
workbench_execution_qualified      false
transform_executed                 false
atlas_correctness_established      false
fmq010_established                 false
neural_alignment_established       false
consciousness_evidence             false
```

A successful hostile verification emits only:

```text
capture_receipt_verified           true
workbench_execution_qualified      false
transform_executed                 false
fmq010_established                 false
neural_alignment_established       false
consciousness_evidence             false
```

Thus:

```text
VerifiedReceipt != QualifiedExecution
```

This verifier does not add `closure_capture_qualified = true`. A later execution-capsule theorem must decide how a verified realization receipt becomes an admissible realization component under the #624 profile.

## Adversarial qualification

The focused suite contains 23 contracts covering, among other cases:

1. valid complete receipt verification;
2. valid incomplete/failure receipt verification without promotion;
3. raw sidecar tamper rejection;
4. self-rehashed argv mutation rejection;
5. sidecar path traversal rejection;
6. sidecar symlink escape rejection;
7. boolean exit-status laundering rejection;
8. boolean byte-length laundering rejection;
9. authority escalation rejection;
10. fact tamper rejection;
11. profile/selection digest substitution rejection;
12. realized-root substitution rejection;
13. normalized identity tamper rejection;
14. orphan raw-closure injection rejection through #629;
15. preservation of unconsumed raw metadata without altering normalized closure meaning;
16. capture-digest tamper rejection;
17. unknown receipt-field rejection;
18. unknown command-field rejection;
19. duplicate JSON-key rejection;
20. incomplete receipt normalized-claim laundering rejection;
21. extra unbound-file rejection;
22. noncanonical receipt serialization rejection;
23. normalizer source/executed-implementation substitution rejection.

## Hosted independence topology

The dedicated workflow deliberately separates observation and verification into different jobs:

```text
static verifier contracts
        ↓
producer job
    Ubuntu 24.04
    pinned Nix 2.33.6
    real Workbench realization/capture
        ↓
immutable Actions artifact
        ↓
fresh verifier job
    Ubuntu 24.04
    no Nix installation
    hostile receipt reconstruction only
```

The verifier job fails if a `nix` executable is unexpectedly present on its PATH.

This prevents verification from consulting the live store to fill gaps in, repair, or reinterpret the receipt.

## Promotion sequence

```text
#624 selected execution profile
        ↓
#629 pure closure identity
        ↓
#638 raw real-world capture producer
        ↓
this independent hostile verifier
        ↓
future verified Workbench execution capsule
        ↓
#576 snapshot-integrated Lineage-B derive()
        ↓
retained scientific execution evidence
        ↓
#525 independent archival reconstruction
        ↓
Lineage A/B FMQ-010
```

Only the independent receipt-verification step is in scope here.
