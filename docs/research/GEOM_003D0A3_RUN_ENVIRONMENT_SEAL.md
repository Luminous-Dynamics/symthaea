# GEOM-003D0A3 — Run environment seal and evidence inventory

Status: implementation candidate. This tranche was defined without executing or inspecting a target GWT lesion outcome.

Parent authority: #3157  
Implementation issue: #3171  
Exact predecessor: `bba5e615d67f39b2d8bde90518980cb77c7922fd`

## Purpose

D0A2 answers: **is this process clean enough to construct the experimental agent?**

D0A3 answers two different questions after construction and execution:

1. **Did the scientific environment stay the same?**
2. **Exactly which evidence artifacts did the experiment produce?**

Those questions are deliberately not collapsed into one directory hash.

## Immutable run environment

The run snapshot binds:

- exact Git subject and tree identity;
- clean-source assertion;
- BLAKE3 identities for Cargo/Nix/toolchain lock inputs supplied by the harness;
- Rust/Cargo, host/target, OS/architecture, hardware identity, and thread policy;
- normalized Cargo feature set, with `scientific_method` required;
- canonical commitment of the full serialized `CognitiveLoopConfig`;
- canonical commitment of the accepted D0A2 preflight report;
- fixed UTC hour;
- campaign, arm, analysis-authority, input-schedule, and arm-order identities;
- one aggregate commitment over the complete UTF-8 process environment;
- the canonical isolated persistence root.

The pre-run and post-run snapshots must produce the same authoritative commitment.

Capture timestamps are explicitly metadata-only and do not participate in the commitment.

## Commitment encoding

The authoritative commitment is not pretty JSON.

D0A3 uses domain-separated BLAKE3 over schema-tagged, length-prefixed fields. JSON exists only as a readable/serializable view.

Full config/preflight values first pass through a local canonical JSON profile that sorts object keys lexicographically. This prevents map insertion order from changing scientific identity. The profile is intentionally described as Symthaea-local and does not claim RFC 8785 compliance.

A BLAKE3 digest is called a **commitment**, never a signature.

## Environment privacy

The full process environment is sorted and committed as key/value bytes, but environment values are not serialized into the run report.

D0A2 separately records presence/absence of the specifically forbidden external-state channels. D0A3 adds aggregate drift detection without publishing secrets.

Non-UTF-8 process environment entries fail closed.

## Persistence boundary

The first D0/D1 campaign permits **no persistent agent state**.

The isolated agent persistence root must:

- be a real directory rather than a symlink;
- be empty before execution under D0A2;
- remain empty after execution under D0A3.

If a subsystem writes anything there, the campaign fails. Scientific outputs belong in a separate evidence directory.

## Evidence inventory

The evidence directory is inventoried recursively after execution.

For every regular file the inventory records:

- normalized relative path;
- byte length;
- BLAKE3 file digest.

Entries are sorted by path before the inventory commitment is computed. Symlinks and non-regular special files are rejected.

The inventory commitment therefore changes on evidence file content, path, size, addition, deletion, or substitution.

## Controls

The candidate includes controls for:

- canonical config commitment under HashMap insertion-order changes;
- process-environment ordering and value sensitivity;
- Cargo feature sorting/deduplication;
- fixed-clock/preflight consistency;
- timestamp non-authority;
- source/tree/toolchain/config/input/arm/clock/process-environment drift;
- stale/tampered serialized commitments;
- empty-agent-root postflight enforcement;
- evidence path/content/addition/deletion sensitivity;
- symlink and Unix special-file rejection.

## Remaining boundary

D0A3 does **not** capture Git/rustc/cargo by invoking subprocesses. The execution harness must supply those observed identities. A later harness tranche may standardize that capture path, but it must consume the D0A3 commitment contract rather than redefining it.

No target GWT lesion execution is authorized by this tranche.
