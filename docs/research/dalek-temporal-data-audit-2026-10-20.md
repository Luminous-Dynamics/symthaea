# Dalek Lean Bench temporal-data audit

**Audit date:** 2026-10-20  
**Upstream pin:** `9fdbfd066420aebcf61a70b06eba57a108c41340`  
**Scope:** read-only inspection of the upstream `README.md`, `proof_timeline.csv`, `dataset.jsonl`, `scripts/eval_timeline.py`, and one statement/proof pair. No upstream files were changed; no LLM proof run was executed.

## Why this audit matters

A timeline-ordered task list is not automatically the same as a historically faithful repository replay. PITB's new temporal track must validate the repository state that existed *before* the target proof, and it must reject a target when the theorem proposition changed before the later proof was recorded.

## Findings at the exact upstream pin

### 1. The current evaluator replays the ordering over a HEAD checkout

The upstream evaluator resolves `HEAD`, passes that SHA to `create_worktree`, and constructs the target worktree from that commit. It then uses the timeline position to inject `sorry` into later files and into the target file. The recorded per-row `commit_hash` is retained in the result metadata, but the evaluator's worktree setup does not select each row's historical `commit_before` snapshot.

Evidence: [`eval_timeline.py` worktree and target setup at the pinned commit](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/9fdbfd066420aebcf61a70b06eba57a108c41340/scripts/eval_timeline.py#L709-L758).

That remains a potentially useful *timeline-masked latest-snapshot* benchmark. It is not equivalent to replaying every theorem against the exact historical API, dependencies and definitions present at its original target time. PITB should report these as different regimes rather than presenting one as the other.

### 2. The top-level counts do not line up exactly

At this pin, parsing `proof_timeline.csv` yields **248 nonblank data rows** (243 with a nonempty `spec_theorem` field); `dataset.jsonl` has **146 records**. The README describes the CSV as a 249-theorem timeline.

Evidence: [pinned README](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/9fdbfd066420aebcf61a70b06eba57a108c41340/README.md), [pinned timeline CSV](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/9fdbfd066420aebcf61a70b06eba57a108c41340/proof_timeline.csv), and [pinned dataset JSONL](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/9fdbfd066420aebcf61a70b06eba57a108c41340/dataset.jsonl).

This may be a documentation/counting convention, but it should be reconciled before anyone reports exact coverage. The manifest generator must publish raw row counts and exclusions; it must not copy the README's headline count as if it were measured.

### 3. Joining dataset rows to timeline rows by file path alone is unsafe

The `dataset.jsonl` has 146 records; at this revision, 50 records occur beyond the first occurrence of their file path (96 unique paths). A straightforward path join against `proof_timeline.csv` found 39 dataset records with exactly one path match, 58 whose path matches multiple timeline records, and 49 with no nonempty timeline path match. The CSV has repeated paths and some rows without a `spec_theorem` path.

This is why a generator cannot simply join by `file_path`. It needs an exact theorem identity, target-state commit, completion commit, statement/content checks, and an explicit ambiguity outcome. If identity cannot be proved, mark the row `unknown` or `excluded`; do not select the first path match.

### 4. A concrete target shows theorem-statement drift

For dataset record `to_bytes_spec_074536e`:

- `dataset.jsonl` records `commit_before = 2263e8b2bbefed2f8e68879b3f89d051309fa54d`.
- The matching timeline row points to proof/verification commit `074536e0de997e115294a0efd09e9b6758fa942c`.
- The theorem at the target-state commit uses the signature `to_bytes_spec (u : Scalar52)` with the original proposition and a `sorry` body.
- At the later proof commit, `to_bytes_spec` has changed to take `self`, a limb-bounds hypothesis, and a canonical-range hypothesis; its proposition also uses the updated result type/conditions.

Evidence: [the pre-proof theorem source](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/2263e8b2bbefed2f8e68879b3f89d051309fa54d/Curve25519Dalek/Specs/Backend/Serial/U64/Scalar/Scalar52/ToBytes.lean), [the later proof source](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/074536e0de997e115294a0efd09e9b6758fa942c/Curve25519Dalek/Specs/Backend/Serial/U64/Scalar/Scalar52/ToBytes.lean), and [the pinned dataset record](https://github.com/Beneficial-AI-Foundation/dalek-lean-bench/blob/9fdbfd066420aebcf61a70b06eba57a108c41340/dataset.jsonl).

This pair must not be scored as proof success for the earlier proposition unless an independent statement-correspondence adjudication establishes equivalence. The safest initial rule is fail closed: compare a target statement digest from `commit_before` with a completion-time statement artifact from the proof commit. Mismatch means explicit exclusion/unknown with a reason, not success.

## PITB response already implemented

The temporal replay manifest now distinguishes three independent pieces of evidence:

1. **Target statement** — exact theorem statement in the target-time state, solver-visible at that target position.
2. **Completion-time statement** — evaluator-only statement bytes at the later proof-completion commit. Its content digest must equal the target statement digest, its path/file digest must agree with the proof artifact, and its source lines must precede the proof body.
3. **Proof body** — evaluator-only source span at the proof-completion commit, associated with the target position and hidden from the solver at that position.

The Rust verifier also checks the pinned repository HEAD/origin, timeline digest, actual `lean-toolchain` bytes, the pinned Mathlib revision extracted from the committed Lake manifest, Git ancestry, full source-file digests, exact line-span digests, future-proof visibility, and exact-content aliases. A local Git integration test exercises these checks over a synthetic three-commit history.

This is a provenance and statement-stability gate. It does **not** establish semantic equivalence when the statement changes; it rejects that case so a separate expert review can decide whether the task is worth remapping.

## Next external-track acceptance gates

- Generate a dry-run audit table from the exact pinned `dataset.jsonl` and `proof_timeline.csv`; report every row as accepted, excluded, or unknown, with the reason and exact hashes.
- Resolve timeline identities by theorem name/signature and source commit, not file path alone.
- Require target statement bytes to be present at the claimed `commit_before`, and the same proposition to be present at the claimed completion commit. Compare normalized elaborated expressions only as a separately audited fallback; never assume textual similarity proves semantic equivalence.
- Verify every historical worktree build and hidden-proof mutation before any model calls. A non-buildable historical snapshot is an explicit exclusion unless a deterministic, documented dependency-recovery procedure repairs it without importing future proof context.
- Run `T0` through `T6` only after source mapping, future-proof leakage controls, positive/negative transfer controls, and budget matching have passed. Keep the static PITB v0 family split and the temporal track's results separately reported.

**No temporal manifest was generated from the full Dalek dataset in this audit, no theorem was re-proved, and no transfer-success rate is claimed.** The point of the audit is to prevent an attractive-looking score from mixing different propositions or future repository context into its denominator.
