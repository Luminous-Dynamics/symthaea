# HCP-MMP1 Lineage-B Atomic No-Replace Publication v1

Status: **candidate custody repair; hosted qualification pending**

## Purpose

The earlier Lineage-B bundle-custody theorem correctly introduced private staging, exclusive retained-file creation, fsync, cooperative publisher locking, and directory rename publication. Later adversarial review found one remaining check/publish race:

```text
check(final absent)
    ...
concurrent creation of empty final directory
    ...
os.rename(staging, final)
```

On POSIX systems, renaming a directory onto an existing empty directory may replace that directory. Therefore:

```text
PrePublishAbsenceCheck + OrdinaryRename
    !=
AtomicNoReplacePublication
```

This tranche closes exactly that gap.

## Required theorem

Lineage-B evidence publication needs both properties at once:

```text
AtomicVisibility
AND
NoReplace
```

Reserving the final directory with `mkdir()` and then moving files individually would satisfy no-replace but expose a partially populated final bundle. Keeping ordinary directory `rename()` preserves atomic visibility but permits replacement of an empty destination created in the race window.

Neither weakened construction is acceptable.

## Linux qualification primitive

The qualified Lineage-B execution platform is Linux. Publication therefore uses:

```text
renameat2(..., RENAME_NOREPLACE)
```

through libc with:

```text
AT_FDCWD         = -100
RENAME_NOREPLACE = 1
```

The entire staged directory moves into the final pathname atomically only if that destination does not exist.

If the destination exists, `EEXIST` is a hard **pre-commit** custody failure and the staged bundle is not substituted for it.

If libc/kernel/filesystem support for the atomic no-replace primitive is unavailable (`ENOSYS`, `EINVAL`, or `ENOTSUP`), publication fails closed. There is deliberately no check-then-rename fallback.

## Publication sequence and commit point

```text
cooperative publish lock acquired
        ↓
hidden private staging directory
        ↓
left.semantic.json       O_EXCL / 0600 / fsync
right.semantic.json      O_EXCL / 0600 / fsync
derivation-evidence.json O_EXCL / 0600 / fsync
        ↓
fsync(staging directory)
        ↓
renameat2(staging, final, RENAME_NOREPLACE)
        ↓
VISIBILITY COMMIT POINT
        ↓
fsync(parent directory)
        ↓
DURABILITY CONFIRMATION
        ↓
cooperative lock cleanup
```

The cooperative lock prevents two conforming publishers from racing each other. `RENAME_NOREPLACE` additionally protects the final pathname against a non-cooperating concurrent creator in the check-to-publish window.

The atomic rename is the visibility commit point. A later parent-directory `fsync` or cooperative-lock cleanup failure cannot truthfully be interpreted as proof that no final bundle is visible.

Therefore:

```text
VisibilityCommit != DurabilityConfirmation
```

and the v1 theorem must not collapse all non-success returns into `NoPublication`.

A future publication receipt should make at least these states explicit:

```text
visibility_committed
durability_confirmed
cooperative_cleanup_confirmed
```

without attempting to roll back an already visible complete bundle after the commit point.

## Failure semantics

Before successful `RENAME_NOREPLACE`, any failed publication attempt must satisfy:

```text
ExistingDestinationPreserved = true
StagingPublishedOverIt        = false
PartialFinalBundleExposed     = false
```

The normal pre-commit cleanup path removes the unpublished staging directory and cooperative lock while leaving a concurrently created destination untouched.

After successful atomic rename, the complete staged bundle has crossed the visibility boundary. Failures in later durability or cooperative-cleanup steps are a distinct state and require explicit reporting rather than the false statement `nothing was published`.

## Generator provenance consequence

This repair modifies `derive_hcpmmp1_neuromaps_lineage_b.py`.

That file is already part of the Lineage-B generator implementation root, so this change intentionally creates a new:

```text
GeneratorImplementationRoot
```

Even though the intended cortical transform and semantic mapping are unchanged, old generator roots and old retained evidence roots must not be silently reused for outputs produced by this implementation.

The existing local Lineage-B derivation and generator-provenance suites are rerun in the dedicated qualification lane.

## Archival-verifier topology

#525 is a sibling theorem, not an ancestor of this execution/custody stack. Its source files are therefore not present in #977 merely because both concern Lineage-B evidence.

The first #977 hosted attempt incorrectly tried to execute a #525 test path from the local checkout and failed before importing any archival verifier code. That was a CI topology error, not an archival compatibility result.

The correct relationship is:

```text
ArchivalVerifierCompatibility != LocalAncestry
```

#977 qualifies its local custody theorem independently. Cross-lineage archival compatibility must be established only after #525 itself has an exact-head hosted-green verifier theorem, using an explicit integration/replay boundary that identifies the exact verifier source root being applied.

This avoids both false assumptions:

```text
sibling PR exists -> sibling files are in current checkout
```

and:

```text
local custody tests pass -> archival reconstruction is qualified
```

## Adversarial contracts

The focused suite covers:

1. successful whole-directory atomic no-replace publication;
2. pre-existing empty destination rejection without replacement;
3. destination creation injected after staging but immediately before atomic publication;
4. complete private final-bundle shape after successful publication;
5. unavailable `renameat2` support fails closed with no pre-commit final publication.

The workflow additionally reruns the existing bundle-custody, Lineage-B derivation, and generator-provenance suites. It deliberately does not pretend the sibling #525 test files are local ancestry.

## Authority boundary

This repair strengthens evidence-bundle filesystem custody only.

It does **not** establish:

- authorized HCP/BALSA acquisition;
- immutable scientific-input snapshot consumption;
- isolated Workbench transform execution;
- same-host repeatability;
- path equivalence;
- cross-CPU equivalence;
- archival-verifier compatibility;
- atlas correctness;
- FMQ-010;
- neural alignment;
- consciousness evidence.

Core invariant:

```text
AtomicNoReplaceEvidenceCustody != ScientificValidity
```

## Relationship to the next Lineage-B tranche

This repair should precede snapshot/execution integration into `derive()` so that newly qualified scientific outputs are not committed through a known weaker publication primitive.

The later integration must still:

- consume #576 snapshot paths rather than mutable operator input paths;
- replace the current ambient `verify_inputs()` / `run_wb()` execution path with the qualified Workbench invocation boundary;
- bind the real #976 program/version/closure identity into run admission rather than trusting arbitrary run-manifest strings;
- expand the generator implementation root for every new output-affecting module;
- migrate the retained evidence schema and independently replay the qualified #525 archival verifier coherently;
- establish same-host repeatability separately before any broader execution-equivalence claim.
