# Workbench Execution Capsule Profile v1

Status: **candidate profile only; no Nix closure has been realized or qualified by this profile**

Schema: `symthaea-workbench-execution-capsule-profile-v1`

## Purpose

The current Lineage-B run manifest binds the `wb_command` file hash and the exact `-version` output. That is useful, but a dynamically linked scientific program is not defined by one executable file alone. Its runtime behavior also depends on the realized library/resource closure and on mutable process state.

This profile defines the next execution theorem without claiming that theorem has already been achieved:

```text
Symthaea flake selection
        +
exact nixpkgs Workbench package
        +
realized recursive Nix runtime closure
        +
closed execution environment
        +
declared platform
        -> Workbench execution capsule identity
```

A capsule identity is still **not** atlas correctness, transform validity, lineage independence, neural alignment, or consciousness evidence.

## Selection root

The v1 profile is bound to Symthaea's root `flake.lock` input:

- root input: `nixpkgs`
- locked node: `nixpkgs_2`
- revision: `9ae611a455b90cf061d8f332b977e387bda8e1ca`
- NAR root: `sha256-md8WlXOlfnIeHeOScMTTHFyf2d6iaTwPl2apR5EQ3P4=`

At that exact nixpkgs revision, the selected package metadata is pinned as:

- attribute: `connectome-workbench`
- package path: `pkgs/by-name/co/connectome-workbench/package.nix`
- Git blob: `01e021edc0f7795f946015bb2a69103cbffaa0ba`
- package version: `2.1.0`
- main program: `wb_command`
- upstream tag: `v2.1.0`
- upstream source hash: `sha256-f1T0i4x7rr3u/3ZvJ4cEAb377e7YcaGMKa2uUslVqR0=`

This establishes a **selection root**. It does not prove that any particular `/nix/store/...` realization was produced from that selection.

## Future closure receipt

A later receipt must identify the realized runtime closure rather than only the executable.

For the Workbench store root it must retain:

1. the root store path;
2. the recursive runtime closure;
3. a NAR hash for every store path;
4. the reference set for every store path;
5. the SHA-256 of the actual `wb_command` file;
6. the SHA-256 of exact `wb_command -version` stdout+stderr;
7. the Nix version used to query the store;
8. execution-platform identity;
9. a canonical SHA-256 over the sorted closure representation.

The intended closure identity is therefore independent of human-readable package names alone.

Long-lived Nix commands such as `nix-store --query --requisites`, `--hash`, and `--references`, or an equivalently qualified `nix path-info --recursive --json` representation, may be used by the future capture implementation. The chosen command/output contract must be frozen by that implementation's own qualification profile.

## Execution environment

The v1 candidate freezes avoidable ambient process variability:

- `LANG=C`
- `LC_ALL=C`
- `TZ=UTC0`
- `OMP_NUM_THREADS=1`
- `OMP_DYNAMIC=FALSE`
- fresh private empty `HOME`
- fresh private empty `XDG_CONFIG_HOME`
- fresh private empty `XDG_CACHE_HOME`
- fresh private empty `TMPDIR`

The POSIX `C` locale avoids introducing a locale-archive availability dependency that the HCP-MMP label/JSON path does not require. `UTC0` similarly avoids relying on a mutable host timezone database for the execution contract. Fixed OpenMP count plus `OMP_DYNAMIC=FALSE` removes avoidable thread-count drift.

These controls do not prove numerical determinism. They remove mutable user configuration, locale/time-zone differences, and uncontrolled OpenMP parallelism from the qualified execution surface.

## Platform boundary

Profile v1 is limited to `x86_64-linux`.

Do not silently reuse a v1 closure qualification for `aarch64-linux`. The HCP-MMP resampling path contains floating-point computation; cross-architecture equivalence must be measured rather than assumed. A future architecture expansion requires either a new profile or explicit cross-platform equivalence evidence.

## Four distinct roots

The program must preserve these distinctions:

```text
selection root
  flake.lock + exact nixpkgs package metadata

realization root
  actual recursive Nix closure + NAR/reference census

execution root
  realization + process environment + platform + exact version output

scientific result root
  Lineage-B inputs + generator + execution root + transform outputs
```

A stronger later root may incorporate the weaker roots, but no weaker root is allowed to claim the authority of a stronger one.

## Authority state

The machine profile deliberately fixes:

- `flake_selection_bound = true`
- `nixpkgs_package_metadata_pinned = true`
- `closure_realized = false`
- `closure_qualified = false`
- `transform_executed = false`
- `scientific_execution_qualified = false`
- `atlas_correctness_established = false`
- `fmq010_established = false`
- `neural_alignment_established = false`
- `consciousness_evidence = false`

The profile verifier fails closed if those non-authorized fields are promoted.

## Qualification invariants

- **WEC-001 Lock binding:** the root flake's selected nixpkgs node must match the profile revision and NAR root.
- **WEC-002 Package binding:** package attribute, source path/blob, version, main program, upstream tag, and source hash are closed-world.
- **WEC-003 Platform scope:** v1 is `x86_64-linux` only.
- **WEC-004 Environment scope:** locale, timezone, OpenMP behavior, HOME/XDG/TMP policies are closed-world.
- **WEC-005 Closure completeness:** future realized receipts require the recursive runtime closure.
- **WEC-006 Per-object content:** every closure object requires a NAR hash.
- **WEC-007 Reference graph:** every closure object requires its reference set.
- **WEC-008 Program identity:** future receipts require the actual `wb_command` SHA-256 and exact version-output SHA-256.
- **WEC-009 No realization laundering:** this profile cannot mark a closure realized or qualified.
- **WEC-010 No scientific laundering:** profile validity cannot mark a transform or scientific execution qualified.
- **WEC-011 Cross-architecture humility:** no qualification transfers to another architecture without evidence.
- **WEC-012 Provenance migration:** when the execution capsule becomes part of Lineage-B execution, its implementation/receipt roots must be incorporated into the scientific generator/evidence chain rather than treated as invisible infrastructure.

## Promotion sequence

```text
profile/flake binding
        ->
realized closure receipt
        ->
independent receipt verification
        ->
platform/environment-qualified Workbench execution
        ->
snapshot-integrated Lineage-B derivation
        ->
retained evidence verification
        ->
Lineage A/B FMQ-010 comparison
```

Only the first step is in scope for this profile PR.
