# Agency TPM2 Local Qualification Capsule v0.1

Status: **authored; not qualified until this capsule itself has executed successfully**.

This document defines the local/self-hosted qualification path for the bounded-agency TPM2 boundary. It exists so GitHub-hosted runner availability is not the only way to discover whether an exact Agency Kernel head compiles and whether its swtpm protocol lane behaves as designed.

It does **not** weaken or replace independent CI. A local result and a hosted result are separate evidence lineages.

## Invocation

From any working tree whose committed `HEAD` contains the capsule:

```bash
bash scripts/agency/qualify-tpm2-local.sh
```

An alternate evidence destination may be supplied with:

```bash
bash scripts/agency/qualify-tpm2-local.sh --out /path/to/evidence
```

The script may be launched from a dirty developer working tree. Those uncommitted bytes are **not** qualified.

## Exact-source rule

The bootstrap records the caller repository's exact committed `HEAD`, creates a detached Git worktree at that commit, and then re-enters the copy of the qualification script contained in that detached worktree.

Consequently:

```text
caller working tree
     |
     | determines HEAD only
     v
detached exact-HEAD worktree
     |
     +-- flake.lock
     +-- rust-toolchain.toml
     +-- Cargo.lock
     +-- qualification script
     +-- TPM adapter source
     +-- Nix verifier source
     |
     v
qualification
```

Uncommitted source, uncommitted lock changes, and an uncommitted modification to the qualification script cannot silently enter a passing result.

## Exact tooling rule

The detached worktree evaluates a minimal Nix shell from its own locked flake:

- nixpkgs comes from the exact `flake.lock`;
- rust-overlay comes from the exact `flake.lock`;
- the Rust channel comes from the exact `rust-toolchain.toml`;
- Cargo, rustc, Clippy, and rustfmt come from that Rust toolchain;
- swtpm, tpm2-tools, binutils, file, Python, coreutils, and the inner Nix client come from locked nixpkgs.

The shell is intentionally much smaller than Symthaea's normal development shell.

## Cargo.lock semantics

The capsule preserves the checked-in lockfile, asks Cargo itself to reconcile workspace metadata, and compares the result structurally.

The reconciliation may contain only new workspace/path package nodes. Any removed package, changed existing package, or newly sourced registry/Git package is a qualification failure.

If Cargo produces only an otherwise acceptable additive workspace-node repair, the capsule continues to run source and protocol qualification against Cargo's exact candidate but exits:

```text
42 / FAIL_LOCK_STALE
```

A stale lock therefore never produces a qualified PASS, while the run still yields useful compiler/protocol evidence and the exact Cargo-generated candidate.

## Source qualification

The current capsule performs:

- rustfmt as diagnostic evidence;
- `cargo test --locked -p symthaea-platform-attestation`;
- `cargo clippy --locked -p symthaea-platform-attestation --all-targets -- -D warnings`;
- exact qualification-probe build.

The probe hash is retained in the evidence bundle.

## Hermetic verifier qualification

The capsule builds `nix/agency-tpm2-verifier-tools.nix` from the exact locked nixpkgs input and records the resulting Nix-store paths and digests.

It requires both reviewed launchers to have no ELF `INTERP` segment and verifies that callers cannot override:

- the quote TCTI;
- the quote PCR serialization format;
- the checkquote TCTI.

The quote launcher's protocol fixes the PCR representation to `serialized`.

## swtpm adversarial lane

The capsule starts an isolated TPM 2.0 software TPM and creates a qualification-only EK/AK fixture.

The baseline PCR-16 profile is produced through the hermetic quote launcher while the ambient process environment deliberately contains bogus:

- `TPM2TOOLS_TCTI`;
- `LD_PRELOAD`;
- `BASH_ENV`.

The hermetic launcher must still reach its compiled-in swtpm endpoint without dynamic-loader contamination.

The Rust production adapter then verifies a fresh challenge through the exact Nix-store quote/checkquote launchers with `require_nix_store_tools = true`.

Finally PCR 16 is extended. A second legitimate fresh TPM quote must fail the original reviewed PCR-profile policy.

This proves the distinction:

```text
fresh authentic quote
        !=
approved platform state
```

## Evidence output

The default evidence directory is under:

```text
target/agency-qualification/
```

The capsule records, where available:

- exact Git HEAD and tree;
- detached-worktree cleanliness;
- rustc/Cargo/Nix/kernel identity;
- checked-in and Cargo-candidate lock hashes;
- Cargo lock reconciliation and exact diff;
- flake.lock and rust-toolchain hashes;
- locked nixpkgs metadata;
- source test/Clippy/rustfmt results;
- exact probe hash;
- verifier Nix-store paths and references;
- quote/checkquote launcher hashes and ELF program headers;
- override-denial evidence;
- swtpm/tpm2-tools versions;
- AK public-key commitment;
- approved PCR-profile commitment;
- successful fresh-attestation output;
- mutated-PCR denial output;
- final result and last completed phase.

Raw temporary TPM contexts, raw AK private material, quote messages, quote signatures, and PCR blobs remain in a temporary runtime directory and are removed before the evidence archive is finalized.

Every retained evidence file is covered by `MANIFEST.sha256`. The directory is then archived with normalized tar metadata and `gzip -n`, and the archive itself receives a SHA-256 commitment.

## Result classes

`PASS` means the exact committed HEAD completed the authored local source + hermetic swtpm qualification and the checked-in Cargo.lock was already fresh.

`FAIL_LOCK_STALE` means source/protocol qualification reached completion but the checked-in lockfile differed from Cargo's acceptable additive workspace-node candidate. It is not a qualified release result.

`FAIL` means a qualification phase failed. `LAST_PHASE.txt` identifies the boundary active when execution stopped.

## Non-claims

A local PASS does not establish:

- independent hosted-runner agreement;
- physical-TPM security;
- firmware event-log correctness;
- Secure Boot correctness;
- IMA measurement-log correctness;
- a compromised-kernel defense;
- remote attestation;
- production Xenia/effect-admission integration.

Those remain separate qualification and architecture gates.

## Intended evidence model

The preferred final state is evidence diversity, not CI replacement:

```text
exact source head
     |
     +--> local/Nix capsule ----+
     |                          |
     +--> hosted GitHub lane ---+--> compare exact commitments
     |                          |
     +--> physical TPM lane ----+
                                |
                                v
                       release qualification
```

A disagreement between independent lineages is itself a failure requiring explanation; evidence from one environment should never be silently substituted for another.
