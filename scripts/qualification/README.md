# Qualification capsules (prototype)

This directory implements issue #242's executor-neutral evidence prototype.

It exists because exact-head GitHub Actions jobs can be accepted into the queue
without receiving a runner. A local/Nix replay should make that state
diagnosable and reproducible without changing the scientific branch just to
manufacture another CI event.

## Evidence boundary

A capsule records:

- exact Git commit and tree identity;
- clean-worktree status;
- GitHub repository identity from `origin`;
- profile identity;
- runner identity;
- selected executor class;
- pinned Rust/tool/environment evidence;
- hashes of manifests/locks declared by the profile;
- exact reviewed step IDs;
- per-step argv hash, exit code, stdout/stderr, and log hashes;
- one hash-addressed capsule manifest.

`CAPSULE.sha256` is a content digest. It is **not a signature** and does not
authenticate the human or machine that executed the profile.

`PASS_EXACT_HEAD@LOCAL_NIX` is not interchangeable with
`PASS_EXACT_HEAD@GITHUB_HOSTED`. Executor identity remains explicit so two
independent executions can be compared rather than collapsed into one Boolean.

## Why profiles cannot contain shell

Profile files select a closed, versioned sequence of step IDs. The runner owns
the reviewed `step ID -> argv` mapping and rejects a profile whose step sequence
differs from the compiled contract.

There is no `eval`, no shell command field, and capsule contents are never
executed.

Changing command semantics therefore requires changing reviewed runner code
and, where semantics change, versioning the profile contract.

## Current profiles

- `statistics-active-test-surface-v1`
- `statistics-core-v1`
- `epidemiology-surveillance-v1`

Their step sequences mirror the focused workflows that existed when issue #242
was opened.

The statistics active-surface profile intentionally omits Clippy. Its parent PR
only claims coherent active 0.1 package/test discovery. The child statistics
core profile adds `clippy --all-targets -- -D warnings`.

All profiles use complete:

`cargo metadata --locked --format-version 1`

They deliberately do not use `--no-deps`.

## Running from a separate worktree

The prototype can live in an infrastructure worktree while the target
scientific worktree remains frozen and clean.

Example shape:

```bash
cd /path/to/exact-target-worktree

python3 /path/to/qualification-infra/scripts/qualification/run_capsule.py \
  --profile /path/to/qualification-infra/scripts/qualification/profiles/statistics-core-v1.profile \
  --expected-head 0123456789abcdef0123456789abcdef01234567 \
  --executor LOCAL_NIX \
  --output /tmp/statistics-core-capsule
```

For qualification-grade evidence:

- `--expected-head` must be the full 40-hex target SHA;
- the target worktree must be clean;
- the target `origin` must canonicalize to the repository declared by the
  profile;
- output must be outside the target worktree;
- the active `rustc` release must exactly match the profile's Rust channel.

On NixOS, enter the target repository's pinned development environment before
running the command when practical. The runner records environment identity
rather than pretending Nix and GitHub Ubuntu are identical.

## Capsule layout

```text
qualification-capsule/
  PROFILE
  SOURCE
  ENVIRONMENT
  RESULTS
  failure.txt          # failure capsules only
  logs/
  SHA256SUMS
  CAPSULE.sha256
```

`SHA256SUMS` is lexical by relative path. `CAPSULE.sha256` is SHA-256 over a
domain-separated `SHA256SUMS` payload, not over a tar archive.

## Fail-closed source rules

Qualification fails before Cargo execution when:

- HEAD differs from the expected SHA;
- the worktree is dirty;
- the origin repository differs;
- a required hashed input file is missing;
- the profile is malformed or its step sequence is substituted;
- Rust does not match the profile channel.

A failure still emits a hash-addressed capsule when possible.

## Tests

Run:

```bash
python3 scripts/qualification/test_run_capsule.py
```

The prototype tests cover:

- GitHub origin canonicalization;
- profile step-substitution rejection;
- exact-HEAD mismatch rejection;
- dirty-tree rejection before toolchain execution;
- deterministic content hashing for identical evidence payloads.

Passing these unit tests is **not** qualification of the scientific profiles.
The actual profile must still run against the intended exact source head.

## Next hardening

Before this becomes a merge/release gate:

1. add a canonical profile-schema document and escaping rules;
2. add profile/runner golden hashes;
3. add a Nix wrapper that selects the intended dev shell explicitly;
4. run an exact frozen head locally and inspect the capsule;
5. add deliberate repository/profile substitution tests;
6. refactor GitHub focused workflows to invoke this runner after hosted runners
   recover;
7. compare LOCAL_NIX and GITHUB_HOSTED capsules for the same source/profile;
8. decide whether authenticated signing is needed as a separate layer.

This mechanism proves execution of declared software checks under a recorded
environment. It does not prove scientific truth, epidemiological validity,
source authenticity beyond Git source identity, or public-health authority.
