# Agency TPM2 Evidence Verifier v0.1

## Purpose

`verify-tpm2-qualification-evidence.py` is an independent, pass-only verifier for the evidence archive emitted by `qualify-tpm2-local.sh`.

The producer and verifier are intentionally separate programs. The producer decides what happened during qualification and writes evidence. The verifier later treats the resulting archive as hostile input and decides only whether that archive satisfies the closed-world V1 evidence contract.

This distinction prevents a release process from reducing qualification review to `tar -xf` plus trusting a producer-written `RESULT=PASS` file.

## V1 acceptance claim

A successful verifier result means:

1. the archive SHA-256 matches the supplied expected value or accompanying sidecar;
2. gzip/tar metadata matches the normalized producer format;
3. the archive contains exactly the V1 evidence file set;
4. the archive contains no links, devices, FIFOs, traversal paths, nested arbitrary files, or duplicate normalized members;
5. `MANIFEST.sha256` covers every retained evidence file exactly once and every retained digest recomputes;
6. all PASS/result/phase/exit/lock-state surfaces agree;
7. the detached qualification worktree status was empty;
8. retained Cargo lock before/candidate data independently satisfies the same no-removal/no-change/no-new-sourced-package rule as the producer;
9. PASS additionally requires the candidate lock to be byte-identical to the checked-in lock;
10. retained flake metadata and retained locked nixpkgs object agree;
11. Rust/Cargo evidence reports the V1 pinned `1.96.0` release;
12. TPM verifier paths are exactly beneath the recorded Nix-store verifier output;
13. retained launcher digest paths agree with those exact executables;
14. retained ELF evidence shows no dynamic interpreter;
15. retained verifier references are Nix-store paths;
16. TCTI/PCR-format override rejection evidence is present;
17. the reviewed PCR profile is a nonzero 32-byte commitment;
18. fresh TPM verification contains the exact V1 digest field set and its verified PCR profile equals the reviewed profile;
19. adversarial PCR mutation is retained as a fail-closed rejection.

The verifier outputs a canonical JSON acceptance summary containing the archive and manifest hashes, exact Git HEAD/tree claim, nixpkgs lock object, TPM policy/challenge/profile commitments, probe hash, and verifier-launcher identities.

## Pass-only profile

V1 intentionally accepts only complete `PASS` capsules.

The producer archives failures too, but a failure can occur before later evidence files exist. Interpreting partial failure archives safely therefore requires a phase-aware schema. V1 does not pretend that the success schema is also such a forensic schema.

A separate failure-forensics profile can be added later if needed.

## Hostile archive handling

The verifier never extracts the tarball.

It rejects:

- absolute paths;
- `..` traversal;
- nested unexpected paths;
- duplicate raw or normalized names;
- symlinks and hard links;
- character/block devices;
- FIFOs and other special members;
- unknown evidence filenames;
- missing V1 files;
- oversized compressed, member, or expanded evidence;
- non-normalized UID/GID/mtime;
- non-normalized gzip timestamp/filename/comment metadata.

This allows an evidence archive to be inspected without first trusting its filesystem behavior.

## External bindings

Normal verification can consume the producer-created `ARCHIVE.tar.gz.sha256` sidecar. This detects accidental corruption but **does not authenticate the producer**, because an attacker who can replace both archive and sidecar can make them agree.

For stronger release use, invoke with independently obtained values:

```text
--release
--expected-archive-sha256 <sha256>
--expected-head <40-hex commit>
--expected-tree <40-hex tree>
```

`--release` requires all three values.

Even then, the verifier proves only that the inspected archive is internally consistent with those external commitments. It does **not** cryptographically prove that the qualification execution itself occurred on a trustworthy machine. The archive commitment must therefore come from an independently trusted publication/witness path if producer authenticity matters.

A future Xenia/SCITT/witness signature over the acceptance statement is the appropriate place to add that property.

## Evidence lineages

The intended qualification model is not "local instead of CI."

It is:

```text
local exact-head Nix/swtpm capsule
            +
hosted GitHub source/swtpm qualification
            +
independent evidence verification
            +
future physical-TPM/measured-boot lane
```

Each lane has different failure and compromise modes. Agreement is stronger than treating any one lane as the universal truth source.

## Tests

`test_verify_tpm2_qualification_evidence.py` builds a complete synthetic normalized PASS archive and includes hostile regressions for:

- manifest/content tampering;
- path traversal;
- symlink injection;
- a self-consistent but stale Cargo candidate presented as PASS;
- a TPM success record whose PCR profile differs from the reviewed profile.

`.github/workflows/agency-tpm2-evidence-verifier.yml` is deliberately a small, dependency-light lane: Python syntax compilation plus the hostile archive tests. It does not itself claim that TPM qualification occurred.

## Non-claims

V1 does not prove:

- who produced an archive when only its adjacent sidecar is supplied;
- physical TPM authenticity;
- measured-boot or IMA correctness;
- trustworthy-kernel execution;
- remote-attestation freshness;
- Xenia authorization semantics;
- that retained command logs could not have been fabricated by a malicious producer.

Those properties belong to separate trust layers rather than being implied by archive parsing.
