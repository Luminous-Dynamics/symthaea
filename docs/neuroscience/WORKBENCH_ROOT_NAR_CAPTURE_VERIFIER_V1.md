# Workbench Root NAR Capture + Independent Membership Verification v1

Status: **candidate real root-NAR observation and independent membership bridge; no Workbench invocation or scientific transform is qualified by this specification**

Schemas:

- producer receipt: `symthaea-workbench-root-nar-capture-receipt-v1`
- verifier result: `symthaea-workbench-root-nar-capture-verification-v1`

## Purpose

The qualified Workbench chain can now independently establish a real realized closure and can purely prove target membership inside a Nix Archive, but those two theorems are not yet connected by retained real root-NAR bytes.

The missing bridge is:

```text
#667 independently verified closure receipt
        -> verified closure root R
        -> verified root NAR SHA-256 H

untrusted Nix-bearing producer
        -> raw nix-store --dump R bytes N

fresh no-Nix verifier
        -> independently re-run #667
        -> recover R and H
        -> require SHA256(N) = H
        -> #690 canonical NAR membership theorem
        -> observe bin/wb_command node
```

The producer does not certify the root, the NAR, or the target. It only retains an observation.

## Parent theorem chain

This specification is stacked on the exact-head-qualified chain:

```text
#624 Workbench selection/execution-capsule profile
  -> #629 canonical Nix closure identity
  -> #638 raw real closure observation producer
  -> #667 independent closure-receipt verifier
  -> #681 root-main-program invocation-isolation profile
  -> #690 pure NAR membership verifier
  -> this real NAR observation/verification bridge
```

No parent theorem is widened here.

## Producer authority is intentionally weak

The producer accepts only:

```text
--closure-receipt-dir <#638 receipt directory>
--out <new create-only destination>
```

It does **not** accept:

- an arbitrary store root;
- an expected NAR SHA-256;
- an expected `wb_command` digest;
- a target path;
- an execution/scientific authority token.

It reads the observed root from the already-produced closure receipt and executes exactly:

```text
nix-store --dump <observed-root>
```

without a shell.

This root remains an untrusted producer-side selection until the fresh verifier independently reconstructs the #667 closure theorem.

Therefore:

```text
ProducerObservedRoot != VerifiedClosureRoot
```

until independent equality is established.

## Raw observation retention

The producer retains:

```text
receipt.json
raw/root.nar
raw/nix-store-dump.stderr
```

The receipt binds:

- the parent closure-capture digest it observed;
- the observed root string;
- exact dump argv;
- exact process exit code;
- raw NAR byte length and SHA-256;
- raw stderr byte length and SHA-256;
- exact producer implementation SHA-256;
- a self-contained capture digest;
- an all-false authority object.

The destination is create-only and assembled in a private temporary sibling before atomic rename. A later run cannot silently overwrite prior evidence.

## Failure remains evidence

If `nix-store --dump` returns nonzero, the producer still retains the raw stdout/NAR sidecar, stderr, exit status, and receipt and emits:

```text
status = root-nar-observation-incomplete-unqualified
```

A failed observation is not converted into a successful membership result.

The independent verifier may validate that a failed receipt faithfully represents failure when used as a unit contract, but the hosted real integration gate requires a successful dump before program membership can be promoted.

## Fresh-runner verification topology

The hosted qualification topology is:

```text
Nix-bearing producer job
  #638 closure observation
  root NAR observation
        ↓
immutable Actions artifact keyed by exact PR-head SHA
        ↓
fresh Ubuntu runner
  exact same PR-head checkout
  Nix required to be absent
        ↓
#667 independently re-verifies closure receipt
        ↓
verified closure identity projection
        ↓
raw NAR receipt verification
        ↓
#690 membership verification
```

Every job first establishes:

```text
CheckedOutCommit = QualificationSubject
```

The artifact name is keyed by the same qualification-subject SHA rather than GitHub's synthetic pull-request merge SHA.

## Verified closure-root projection

After #667 returns `verified-complete-observation`, the verifier re-opens the canonical closure identity that #667 has authenticated and requires its closure digest to equal #667's independently reconstructed digest.

It then projects exactly:

```text
VerifiedRootProjection = {
    root_store_path,
    root_nar_sha256,
    closure_digest
}
```

from the unique closure entry whose path equals the canonical closure root.

This projection is not a new realization theorem. It is a deterministic read of already-verified closure evidence.

## Dump receipt reconstruction

The fresh verifier requires the NAR capture receipt to be canonical closed-world JSON and reconstructs all of the following rather than trusting producer claims:

```text
receipt.root == VerifiedRootProjection.root_store_path
receipt.closure_capture_digest == #667 verified capture digest
receipt.command.argv == ["nix-store", "--dump", verified_root]
receipt.command.exit_code == 0
receipt.implementation_sha256 == SHA256(exact producer source)
retained sidecar lengths/digests == actual files
receipt.capture_digest == independently recomputed digest
all producer authority fields == false
```

The receipt filesystem is also closed-world. Symlinks, path escape, missing files, or extra unbound files are rejected.

## NAR bytes are authenticated before parsing

The decisive bridge is delegated to the already-qualified #690 theorem:

```text
#690.verify_membership(
    raw/root.nar,
    VerifiedRootProjection.root_nar_sha256,
    "bin/wb_command"
)
```

#690 hashes the entire NAR before structural parsing.

Therefore:

```text
SHA256(CapturedRootNAR) != VerifiedRootNarHash
    -> reject before parse
```

and only:

```text
SHA256(CapturedRootNAR) = VerifiedRootNarHash
    -> canonical nix-archive-1 parse
    -> target membership observation
```

The child verifier also requires #690's exact output schema, expected NAR hash, exact target spelling, and non-promoting authority object.

## Target-shape humility

This specification does not assume before observation that `bin/wb_command` is a regular executable file.

The real target may be:

```text
regular executable
regular non-executable
symlink
```

The verifier preserves the observed shape.

### Regular executable

If #690 proves:

```text
node_type = regular
executable = true
content_sha256 = sha256:<...>
```

then the child may establish only:

```text
root_main_program_regular_executable_verified = true
```

This authenticates the exact entry-program bytes and execute bit inside the verified root NAR.

It still does not prove that the entry program has executed or that a wrapper's descendant environment is scientifically reproducible.

### Symlink

If #690 observes a symlink:

```text
target_membership_verified = true
root_main_program_regular_executable_verified = false
symlink_resolution_verified = false
```

No regular-file digest is invented. Canonical in-NAR symlink resolution becomes the next explicit theorem.

This branch is essential because package layout must be observed, not assumed.

## Relationship to the #681 wrapper/environment boundary

Even if `bin/wb_command` is a verified regular executable, program membership is not invocation equivalence.

The selected package may use closure-owned wrapper behavior that transforms the runner-controlled entry environment before a descendant binary executes. #681 already freezes the distinction:

```text
RunnerEntryEnvironment
    + VerifiedProgramBytes
        -> ProgramDefinedEnvironmentTransition
        -> DescendantEnvironment
```

This PR therefore never promotes:

```text
ProgramMembership -> WorkbenchExecutionQualified
```

## Authority boundary

A fully verified successful result may establish:

```text
closure_receipt_verified = true
root_nar_capture_verified = true
target_membership_verified = true
```

and conditionally, only for a regular executable target:

```text
root_main_program_regular_executable_verified = true
```

It always preserves:

```text
symlink_resolution_verified = false
workbench_execution_qualified = false
transform_executed = false
fmq010_established = false
neural_alignment_established = false
consciousness_evidence = false
```

The core invariant is:

```text
VerifiedProgramMembership != ProgramExecution
```

and:

```text
ProgramExecution != ScientificCorrectness
```

## Qualification contracts

The authored adversarial suite covers at least:

- successful create-only producer capture;
- failed dump retention without promotion;
- incomplete parent closure receipt rejection;
- noncanonical producer root rejection;
- independently reconstructed root binding;
- self-rehashed root substitution rejection;
- self-rehashed argv mutation rejection;
- raw NAR sidecar tamper rejection;
- raw stderr tamper rejection;
- authority escalation rejection;
- unknown receipt-field rejection;
- bool/int exit-code laundering rejection;
- extra unbound file rejection;
- noncanonical receipt serialization rejection;
- producer implementation substitution rejection;
- duplicate JSON-key rejection;
- verified root projection;
- closure-digest substitution rejection;
- missing root-entry rejection;
- regular executable authority branch;
- symlink non-laundering branch.

The focused workflow additionally enforces:

- exact-head checkout in every job;
- no Nix/subprocess/network execution surface in the fresh verifier;
- shell-free producer process execution;
- exact inherited verifier source binding;
- fixed false scientific authority fields.

## Promotion sequence after this theorem

If the real target is a regular executable, the next layer can bind its authenticated bytes into a concrete execution-observation receipt together with #681's isolated invocation context.

If the real target is a symlink, the next layer must instead prove canonical in-NAR symlink resolution first.

In either case the later scientific sequence remains:

```text
verified entry program
  -> isolated real invocation observation
  -> same-host repeatability
  -> path perturbation equivalence
  -> CPU-dispatch qualification
  -> actual Lineage-B transform
  -> exact scientific-output commitment
  -> independent archival reconstruction
  -> real A <-> B FMQ-010
```

No later state is claimed here.
