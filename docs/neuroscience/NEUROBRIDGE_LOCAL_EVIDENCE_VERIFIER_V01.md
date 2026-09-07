# NeuroBridge Local Qualification Evidence Verifier v0.1

Status: **candidate verifier; verifier tests do not themselves prove a qualification run occurred**

Verifier profile: `symthaea-neurobridge-local-evidence-verifier-v0.1`

Producer profile accepted: `symthaea-neurobridge-local-qualification-v0.1`

## Purpose

The local/Nix qualification capsule and the program that accepts its retained evidence must not be the same authority surface.

This verifier treats a purported NeuroBridge local PASS archive as hostile input and decides only whether the archive satisfies the closed-world v0.1 PASS evidence contract.

It never executes Nix, Cargo, Rust tests, Workbench, the qualification producer, or real neuroscience-data processing.

## Never extract hostile evidence

The verifier reads gzip/tar members directly in memory. It never calls archive extraction APIs.

It rejects:

- oversized compressed archives;
- excessive expanded size or oversized members;
- non-normalized gzip timestamps/flags;
- truncated, trailing or concatenated gzip streams;
- non-normalized tar ownership or timestamps;
- PAX metadata;
- absolute, nested or traversal member names;
- duplicate normalized member names;
- symlinks, hardlinks, devices, FIFOs and other special members;
- unexpected files;
- missing PASS files;
- retained files whose mode is not `0600`;
- a root evidence directory whose archived mode is not `0700`.

The exact PASS archive allowlist is encoded in the verifier. Failure archives are intentionally outside v0.1 acceptance because they can legitimately stop before later PASS files exist.

## Manifest integrity

`MANIFEST.sha256` must cover every retained PASS evidence file exactly once except itself.

Every covered digest is recomputed from archive member bytes.

A self-consistent manifest is necessary but not sufficient for acceptance.

## Independent PASS semantics

The verifier independently requires:

- the exact ordered set of 22 authored qualification phases, each marked `PASS`;
- empty pre-qualification and post-qualification Git status files;
- well-formed exact Git HEAD and tree identities;
- the retained Git commit record to agree with those identities;
- closed-world `STATUS.env` with the exact producer profile;
- `EXECUTION_RESULT=PASS`;
- `EXECUTION_EXIT_CODE=0`;
- `LAST_PHASE=complete`;
- status HEAD/tree to agree with the retained source identities;
- exact source-lock path commitments for `Cargo.lock`, `flake.lock`, and `rust-toolchain.toml`;
- exact qualifier-file commitment paths;
- exact hosted-focused-workflow commitment paths;
- Rust 1.96.0, Cargo 1.96.0, Python 3.11 and Nix identity markers in retained tool evidence.

A producer-authored `PASS` string therefore cannot establish acceptance by itself.

## Release-mode external bindings

Ordinary verification recomputes and reports the archive SHA-256.

For stronger use, `--release` requires independently supplied:

- expected archive SHA-256 in `sha256:<64 lowercase hex>` form;
- exact expected Git HEAD;
- exact expected Git tree.

Those values are compared against the inspected archive.

This is an external commitment boundary, not producer authentication. An attacker who controls both the execution environment and the externally trusted commitments is outside the theorem.

## Acceptance statement

A successful verifier emits canonical JSON containing:

- verifier profile;
- accepted PASS status;
- archive SHA-256;
- manifest SHA-256;
- source HEAD/tree;
- source lock commitments;
- qualification-capsule file commitments;
- focused hosted-workflow commitments;
- explicit negative authority flags.

The current negative authority flags include:

- `producer_authenticity_established = false`;
- `hosted_ci_agreement_established = false`;
- `real_neuroscience_evidence_established = false`.

Acceptance therefore means only:

> the inspected archive is a closed-world, internally coherent NeuroBridge local PASS archive consistent with any supplied external commitments.

It does **not** mean a trustworthy machine necessarily executed the qualification.

## Hostile regression suite

The dependency-free verifier suite currently covers:

1. valid synthetic normalized PASS acceptance;
2. retained member tampering without manifest repair;
3. path traversal;
4. symlink injection;
5. duplicate member injection;
6. self-consistent `STATUS=FAIL` after manifest repair;
7. reordered PASS phase sequence after manifest repair;
8. dirty post-qualification source after manifest repair;
9. release mode without external commitments;
10. wrong external archive root;
11. nonzero gzip mtime;
12. non-normalized tar mtime;
13. an unexpected manifest-covered file.

These fixtures validate the verifier logic only. They are not local NeuroBridge qualification evidence.

## Relationship to hosted CI and real-data evidence

The intended evidence structure remains:

```text
exact source head
    ├── hosted focused Actions
    ├── local/Nix qualification archive
    │       └── hostile-input verifier
    └── later authorized real-data execution
```

The verifier does not turn local evidence into hosted evidence and does not qualify a real HCP/BALSA/Workbench run.

A future witness layer may sign the verifier acceptance statement or bind it into Xenia/SCITT-style provenance. That is the correct place to add producer/witness authenticity rather than teaching this verifier to trust its own archive producer.

## Non-claims

Verifier acceptance does not establish:

- producer authenticity;
- uncompromised host/kernel/Nix daemon;
- hosted CI agreement;
- authorized HCP/BALSA/Mills acquisition;
- Connectome Workbench scientific correctness;
- Lineage-A/Lineage-B independence;
- FMQ-010;
- human neural representational alignment;
- substrate consciousness;
- Symthaea consciousness.
