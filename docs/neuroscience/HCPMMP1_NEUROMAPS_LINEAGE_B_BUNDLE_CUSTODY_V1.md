# HCP-MMP1 neuromaps Lineage-B Bundle Custody v1

Status: **evidence-publication custody qualification only**

This profile hardens publication of candidate Lineage-B evidence bundles without changing the scientific evidence schema or granting any new scientific authority.

## Theorem

`completed scientific execution != safely retained evidence bundle`

A successful transform is not enough. Publication must also avoid silent overwrite, partial visible bundles, permissive file modes, and accidental disclosure through CLI output.

## Publication contract

The derivation publishes exactly three retained artifacts:

- `left.semantic.json`;
- `right.semantic.json`;
- `derivation-evidence.json`.

Publication uses a sibling hidden staging directory and a cooperative publication lock. The destination must not already exist. Files are created with mode `0600`; the bundle directory is mode `0700`.

All staged files and the staging directory are fsync'd before the staging directory is renamed into the requested final path. The parent directory is fsync'd after publication.

If any staged write fails, the staging directory is removed and no final bundle is published.

This is a single-publisher custody mechanism for the supported POSIX execution environment; it is not a distributed locking protocol or a claim that arbitrary external processes cannot race the filesystem.

## No overwrite

An existing final destination is a hard error. The mechanism never deliberately truncates or replaces a prior evidence bundle.

A pre-existing publication lock is also a hard error.

## CLI disclosure boundary

The complete derivation evidence includes descriptive provenance such as `authorization_reference`, so neither `derive` nor `verify-evidence` prints the full evidence document.

Successful CLI output is a minimal receipt containing only:

- receipt profile;
- action (`derive` or `verify`);
- evidence content digest;
- retained evidence-file SHA-256.

Local paths and descriptive authorization/provenance text are not emitted in the success receipt.

Programmatic callers may still receive the validated evidence object directly; the restriction applies to the command-line logging surface.

## Relationship to other qualification layers

- #523 defines the Lineage-B scientific transform and generator provenance;
- #540 captures a pre-execution run manifest;
- this custody profile controls publication of the resulting candidate evidence bundle;
- #525 separately verifies a retained bundle against an externally retained root;
- #490/#491 separately compile and compare atlas mappings.

No custody property promotes evidence authority.

## Qualification gates

The dedicated suite covers:

1. restrictive directory/file modes;
2. programmatic evidence validation after publication;
3. existing-destination rejection without overwrite;
4. pre-existing publication-lock rejection;
5. staged-write failure cleanup with no final bundle;
6. digest-only derive CLI receipt;
7. digest-only verify CLI receipt.

The dedicated CI lane also preserves the full #523 derivation tests and generator-provenance tests.

## Non-goals

This profile does not establish:

- authorized HCP/BALSA acquisition;
- actual independent execution provenance;
- atlas correctness;
- external retained-root custody;
- FMQ-010;
- empirical neural alignment;
- consciousness evidence;
- benchmark de-quarantine authority.
