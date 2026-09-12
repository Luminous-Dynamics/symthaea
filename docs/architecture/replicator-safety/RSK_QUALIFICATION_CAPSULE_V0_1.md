# Replicator Safety Kernel — Qualification Capsule v0.1

Status: **normative evidence-generation contract; not a production-admission record**

This document defines how the RSK reference Rust crates are qualified consistently across local machines, GitHub Actions, or other controlled executors.

It contains no physical replication mechanism.

---

## 1. Objective

The qualification capsule answers a narrow question:

> Did the declared RSK Rust qualification gates pass for this exact Git subject, under the exact repository-pinned Rust channel, without mutating the qualified repository state?

It does **not** answer whether RSK is production-admitted or physically safe.

CI and local qualification execute one shared harness:

```text
scripts/rsk_qualification.py
```

rather than maintaining independent command lists.

---

## 2. Qualification scope

Receipt scope:

```text
rsk-reference-rust-v0.1
```

Receipt schema:

```text
symthaea.rsk.qualification-receipt.v1
```

Production status carried by every v0.1 receipt:

```text
DENIED / NOT YET ELIGIBLE
```

A passing receipt must never be interpreted as changing that status.

---

## 3. Required gates

### Format phase

```text
cargo fmt --check \
  -p symthaea-replicator-safety \
  -p symthaea-replicator-ledger
```

### Core phase

```text
cargo check -p symthaea-replicator-safety --all-targets --locked
cargo check -p symthaea-replicator-ledger --all-targets --locked
cargo test -p symthaea-replicator-safety --all-targets --locked
cargo test -p symthaea-replicator-ledger --all-targets --locked
cargo clippy -p symthaea-replicator-safety --all-targets --locked -- -D warnings
cargo clippy -p symthaea-replicator-ledger --all-targets --locked -- -D warnings
```

### All phase

The union of format and core.

Commands are run sequentially and all selected commands are attempted so one run can reveal more than one independent failure.

`--locked` is part of the normative semantic gate. Dependency resolution may not rewrite the qualified lockfile.

---

## 4. Exact subject binding

Each receipt records:

- `git rev-parse HEAD`;
- `git rev-parse HEAD^{tree}`;
- branch name;
- whether the worktree was dirty before execution;
- whether it was dirty after execution;
- pre/post dirty path lists when applicable.

A clean tree is required both before and after qualification for `admissible_evidence = true`.

The Git tree digest is the primary content identity for tracked repository state. Additional selected file hashes make important evidence inputs human-visible without replacing the tree binding.

---

## 5. Selected input hashes

The receipt records SHA-256 **before and after execution** for at least:

```text
Cargo.lock
rust-toolchain.toml
scripts/rsk_qualification.py
scripts/test_rsk_qualification.py
.github/workflows/rsk-safety.yml
scripts/check-class-a-changes.sh
docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md
```

The receipt also records whether these selected inputs remained unchanged.

Missing expected inputs are represented explicitly as `null`; they are not silently ignored.

A change to one of these inputs during an initially clean qualification makes the receipt non-admissible even if all Rust commands exit successfully.

---

## 6. Toolchain and executor identity

The repository currently pins Rust through `rust-toolchain.toml`.

The harness reads the declared channel and requires the observed `rustc --version --verbose` release version to match it exactly.

A mismatch yields:

```text
qualification_status = fail-toolchain-mismatch
admissible_evidence  = false
```

The receipt also records tool output for:

- `rustc --version --verbose`;
- `cargo --version`;
- `rustfmt --version`;
- `cargo clippy --version`.

It records:

- expected Rust channel;
- toolchain pin-match result;
- platform identity as observed by Python;
- Python version;
- available GitHub Actions run/job/ref/SHA context when executed in Actions.

These are evidence facts, not proof that the executor itself is trusted for production promotion. Exact admitted executor/runtime identity remains part of the separate production-admission process.

---

## 7. Command evidence

For every selected command the receipt records:

- stable command name;
- argument vector;
- UTC start and finish time;
- duration in milliseconds;
- process exit code;
- relative log path;
- SHA-256 of combined stdout/stderr.

Each command's full combined output is persisted under the capsule's `logs/` directory.

Missing executables are represented as exit code `127` with a hashed error log.

---

## 8. Receipt digest

The receipt is first serialized as canonical compact JSON:

```text
sort_keys = true
separators = (",", ":")
```

SHA-256 is computed over that representation before `receipt_sha256` is inserted.

The output directory contains:

```text
receipt.json
receipt.sha256
logs/*.log
```

The digest provides integrity binding for the receipt contents. It is not a digital signature and does not prove who executed the qualification.

---

## 9. Worktree-integrity states

### Clean pass

```text
qualification_status = pass-clean
admissible_evidence  = true
```

This requires:

- clean initial worktree;
- clean post-run worktree;
- unchanged selected input hashes;
- exact toolchain pin match;
- every selected command passing.

### Dirty default

The harness refuses qualification before Rust commands execute:

```text
qualification_status = blocked-dirty-worktree
admissible_evidence  = false
```

Exit status is nonzero.

### Tool-induced mutation

If an initially clean worktree becomes dirty or selected evidence inputs change during qualification:

```text
qualification_status = fail-worktree-mutated
admissible_evidence  = false
```

This remains a failure even if every selected Rust command returned zero.

### Dirty diagnostic override

With `--allow-dirty`, commands may be run for developer diagnosis, but even if all pass:

```text
qualification_status = pass-dirty-diagnostic
admissible_evidence  = false
```

Dirty results must never be promoted to qualification or release evidence.

---

## 10. Local usage

From a clean Symthaea worktree whose selected Rust toolchain is available:

```text
python3 scripts/rsk_qualification.py --phase all
```

Format only:

```text
python3 scripts/rsk_qualification.py --phase format
```

Core semantic gates only:

```text
python3 scripts/rsk_qualification.py --phase core
```

The default output directory is:

```text
target/rsk-qualification
```

---

## 11. Harness self-test

The qualification machinery has a Rust-independent Python/Git self-test:

```text
python3 scripts/test_rsk_qualification.py -v
```

The candidate test set covers:

1. clean matching toolchain with successful fake commands;
2. wrong Rust version despite successful fake commands;
3. dirty-start refusal;
4. fake Cargo mutation of `Cargo.lock` during qualification;
5. independent receipt-digest recomputation.

Before this contract update, the candidate self-test passed 5/5 locally.

The self-test itself is a Class A-protected file because weakening the evidence-generator tests could weaken qualification assurance.

---

## 12. CI usage

The focused RSK workflow runs format and core independently so style failure does not mask semantic evidence.

Its governance job runs the qualification-harness self-test before relying on the harness for Rust evidence.

Each Rust job invokes the same harness and uploads its output directory using `actions/upload-artifact@v4` with `if: always()`.

This gives useful distinct outcomes:

1. gate passes, receipt retained;
2. Rust gate fails, failure receipt/log retained;
3. wrong toolchain is detected and retained;
4. dirty/mutated subject is denied and retained;
5. missing tool is denied and retained;
6. harness crashes before producing a receipt, artifact upload itself exposes the missing-evidence path.

Queued or never-started jobs remain **no evidence**.

---

## 13. Evidence hierarchy

A clean passing Rust receipt is one evidence class only.

It does not replace:

- Class A ADR review;
- TLA+ execution and formal bounds;
- property/generated histories;
- fuzzing;
- verified grant/quorum/monitor/time evidence;
- durable journal/CAS/crash/fork tests;
- recovery tests;
- build/runtime admission evidence;
- protected-branch enforcement evidence;
- deployment-specific physical safety cases.

The production-admission matrix remains authoritative.

---

## 14. Anti-downgrade requirements

The qualification harness, its self-test, focused workflow, and Class A classifier are themselves Class A safety-governance surfaces.

A future change must not silently:

- remove a required command;
- remove `--locked` from semantic Cargo gates;
- stop using `--all-targets` where specified;
- remove `-D warnings` from Clippy;
- accept the wrong Rust channel;
- treat dirty evidence as admissible;
- omit the post-run worktree check;
- omit before/after input hashes;
- omit commit/tree binding;
- omit tool/input identities;
- stop hashing logs;
- change failure into success;
- remove the production-admission denial marker;
- stop retaining failure receipts in CI;
- stop self-testing the evidence generator.

Such changes require a Class A ADR.

---

## 15. Future strengthening

Potential future work, separately reviewed:

- signed qualification receipts;
- transparency inclusion receipts;
- Nix derivation/store-path identity;
- hermetic/offline dependency proof;
- SBOM/provenance binding;
- TPM/TEE executor attestation where justified;
- reproducible-binary comparison;
- TLA+/property/fuzz evidence manifests folded into a higher-level qualification bundle.

Those additions should compose with this receipt rather than changing a Rust test pass into an implicit production authority decision.
