# Replicator Safety Kernel — Qualification Receipt Verification v0.1

Status: **normative evidence-verification contract; not a production-admission record**

This document defines independent verification of RSK qualification capsules produced under `symthaea.rsk.qualification-receipt.v1`.

It contains no physical replication mechanism.

---

## 1. Purpose

A qualification receipt is untrusted input until independently verified.

The verifier answers:

> Is this capsule internally consistent with the frozen RSK v0.1 qualification profile, are its retained logs intact, and—when requested—does it bind to this exact checkout?

It does not answer whether the overall RSK is production-admitted.

---

## 2. Verifier interface

Reference verifier:

```text
python3 scripts/verify_rsk_qualification.py \
  <capsule>/receipt.json
```

Checkout-bound verification:

```text
python3 scripts/verify_rsk_qualification.py \
  <capsule>/receipt.json \
  --repo . \
  --require-current-subject
```

The second form is the focused CI mode.

---

## 3. Independent command profile

The verifier owns an independent copy of the required v0.1 command profile.

For `format` it requires exactly one command:

```text
cargo fmt --check \
  -p symthaea-replicator-safety \
  -p symthaea-replicator-ledger
```

For `core` it requires exactly:

```text
cargo check -p symthaea-replicator-safety --all-targets --locked
cargo check -p symthaea-replicator-ledger --all-targets --locked
cargo test -p symthaea-replicator-safety --all-targets --locked
cargo test -p symthaea-replicator-ledger --all-targets --locked
cargo clippy -p symthaea-replicator-safety --all-targets --locked -- -D warnings
cargo clippy -p symthaea-replicator-ledger --all-targets --locked -- -D warnings
```

For `all`, the verifier requires the exact format command followed by the exact core sequence.

Command omission, addition, reordering, renaming, or argv substitution invalidates the receipt even if the attacker recomputes the receipt's outer SHA-256.

---

## 4. Receipt digest verification

The verifier:

1. parses JSON with duplicate-key rejection;
2. removes `receipt_sha256`;
3. serializes the remaining receipt as sorted compact JSON;
4. recomputes SHA-256;
5. requires equality with the claimed digest;
6. if `receipt.sha256` is present, requires the sidecar to match too.

The digest establishes content integrity only. It does not authenticate the executor.

---

## 5. Log verification

Each command record must include a safe relative log path and SHA-256.

The verifier rejects:

- absolute paths;
- `..` traversal;
- paths that resolve outside the capsule root;
- missing logs;
- logs above the v0.1 size bound;
- log digest mismatch.

The verifier hashes retained log bytes independently.

---

## 6. Input-set verification

The v0.1 receipt must contain the exact selected input-path set defined by the qualification profile.

For each path, both `before` and `after` contain either:

- lowercase SHA-256; or
- explicit `null` when the expected input was absent.

The verifier independently derives:

```text
inputs_unchanged = before == after
```

and rejects a receipt whose summary boolean disagrees.

When `--repo` is supplied, the verifier hashes those current checkout paths and requires them to match the receipt's `after` map.

---

## 7. Toolchain verification

The receipt carries:

```text
expected_rust_channel
pin_match
rustc --version --verbose output
```

The verifier independently derives whether the first `rustc` version line matches the declared exact channel and requires the recorded `pin_match` to agree.

When a checkout is supplied, the channel in its `rust-toolchain.toml` must also match the receipt.

A receipt cannot become acceptable by changing `pin_match` and rehashing itself.

---

## 8. Repository-state verification

The verifier requires well-formed Git commit/tree IDs and checks consistency of:

- `dirty` with `dirty_paths`;
- `post_run_dirty` with `post_run_dirty_paths` where post-run state exists.

With `--require-current-subject`, it additionally requires:

```text
receipt.repository.commit == git rev-parse HEAD
receipt.repository.tree   == git rev-parse HEAD^{tree}
```

This is the mode used by exact-head focused CI.

---

## 9. Status derivation

The verifier does not trust `qualification_status` or `admissible_evidence` directly.

For non-blocked receipts it derives:

```text
commands_pass      = every required command exit_code == 0
pin_match          = independently derived from rustc evidence
clean_start        = dirty == false
clean_finish       = post_run_dirty == false
inputs_unchanged   = before == after
clean_evidence     = clean_start && clean_finish && inputs_unchanged
```

Then the v0.1 status function is independently reconstructed:

```text
if commands_pass && pin_match && clean_evidence:
    pass-clean
else if clean_start && (post_run_dirty || !inputs_unchanged):
    fail-worktree-mutated
else if commands_pass && pin_match:
    pass-dirty-diagnostic
else if commands_pass:
    fail-toolchain-mismatch
else:
    fail
```

For a pre-run dirty refusal:

```text
blocked-dirty-worktree
```

with no qualification commands.

The verifier requires the claimed status and admissibility boolean to equal its derived result.

---

## 10. Exact-head focused CI

For pull requests, the RSK workflow explicitly checks out:

```text
github.event.pull_request.head.sha
```

before governance, format, and core qualification.

This produces exact candidate-head RSK evidence rather than silently conflating it with GitHub's synthetic merge ref.

Broader repository CI can still test merge/integration behavior independently.

---

## 11. Failure-receipt verification

A failed qualification can still produce valid evidence.

Example:

```text
qualification_status = fail
admissible_evidence  = false
```

The verifier may return success for the *integrity of that failure receipt* while the original qualification command remains failed.

This distinction is intentional:

```text
receipt verification success != qualification success
```

CI preserves both facts because the qualification step and verifier step are separate.

---

## 12. Adversarial test obligations

At minimum the independent verifier test suite covers:

- valid exact-checkout receipt;
- receipt tamper;
- log tamper;
- required-command omission with attacker rehash;
- false toolchain pin result with attacker rehash;
- false input-unchanged result with attacker rehash;
- path traversal with attacker rehash;
- selected-current-input substitution.

Future additions should cover duplicate JSON keys, unknown fields, oversized inputs, malformed scalar types, and controlled fuzzing of bounded receipt structures.

---

## 13. Trust boundary

The verifier strengthens integrity checking, but the v0.1 capsule still does not prove:

- who executed the qualification;
- that the executor OS/hardware was uncompromised;
- hermetic dependency acquisition;
- binary reproducibility;
- runtime artifact identity;
- formal correctness;
- protected-branch policy;
- production admission.

Those remain separate evidence/admission gates.

---

## 14. Class A protection

All of these are Class A surfaces:

```text
scripts/rsk_qualification.py
scripts/test_rsk_qualification.py
scripts/verify_rsk_qualification.py
scripts/test_verify_rsk_qualification.py
.github/workflows/rsk-safety.yml
scripts/check-class-a-changes.sh
```

The focused workflow checks that the generic classifier retains these paths.

A future change cannot silently weaken generation or verification while appearing to be ordinary tooling maintenance.
