# Replicator Safety Kernel — Qualification Capsule v0.1

Status: **normative evidence-generation contract; not a production-admission record**

This document defines how the RSK reference Rust crates are qualified consistently across local machines, GitHub Actions, or other controlled executors.

It contains no physical replication mechanism.

---

## 1. Objective

The qualification capsule answers a narrow question:

> Did the declared RSK Rust qualification gates pass for this exact Git subject, with this recorded toolchain and input set?

It does **not** answer whether RSK is production-admitted or physically safe.

The capsule is designed so CI and local qualification execute one shared harness:

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
cargo check -p symthaea-replicator-safety --all-targets
cargo check -p symthaea-replicator-ledger --all-targets
cargo test -p symthaea-replicator-safety --all-targets
cargo test -p symthaea-replicator-ledger --all-targets
cargo clippy -p symthaea-replicator-safety --all-targets -- -D warnings
cargo clippy -p symthaea-replicator-ledger --all-targets -- -D warnings
```

### All phase

The union of format and core.

Commands are run sequentially and all selected commands are attempted so one run can reveal more than one independent failure.

---

## 4. Exact subject binding

Each receipt records:

- `git rev-parse HEAD`;
- `git rev-parse HEAD^{tree}`;
- branch name;
- whether the worktree was dirty;
- dirty path list when applicable.

A clean tree is required for `admissible_evidence = true`.

The Git tree digest is the primary content identity for tracked repository state. Additional selected file hashes make important evidence inputs human-visible without replacing the tree binding.

---

## 5. Selected input hashes

The receipt records SHA-256 for at least:

```text
Cargo.lock
rust-toolchain.toml
scripts/rsk_qualification.py
.github/workflows/rsk-safety.yml
scripts/check-class-a-changes.sh
docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md
```

Missing expected inputs are represented explicitly as `null`; they are not silently ignored.

---

## 6. Tool and executor identity

The receipt records tool output for:

- `rustc --version --verbose`;
- `cargo --version`;
- `rustfmt --version`;
- `cargo clippy --version`.

It also records:

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

## 9. Clean versus dirty evidence

### Clean pass

```text
qualification_status = pass-clean
admissible_evidence  = true
```

### Dirty default

The harness refuses qualification before Rust commands execute:

```text
qualification_status = blocked-dirty-worktree
admissible_evidence  = false
```

Exit status is nonzero.

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

## 11. CI usage

The focused RSK workflow runs format and core independently so style failure does not mask semantic evidence.

Each job invokes the same harness and uploads its output directory using `actions/upload-artifact@v4` with `if: always()`.

This gives four useful cases:

1. gate passes, receipt retained;
2. gate fails normally, failure receipt/log retained;
3. harness detects missing tool or dirty subject, denial receipt retained;
4. harness crashes before producing a receipt, artifact upload itself exposes the missing evidence path.

Queued or never-started jobs remain **no evidence**.

---

## 12. Evidence hierarchy

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

## 13. Anti-downgrade requirements

The qualification harness, focused workflow, and Class A classifier are themselves Class A safety-governance surfaces.

A future change must not silently:

- remove a required command;
- stop using `--all-targets` where specified;
- remove `-D warnings` from Clippy;
- treat dirty evidence as admissible;
- omit commit/tree binding;
- omit tool/input identities;
- stop hashing logs;
- change failure into success;
- remove the production-admission denial marker;
- stop retaining failure receipts in CI.

Such changes require a Class A ADR.

---

## 14. Future strengthening

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
