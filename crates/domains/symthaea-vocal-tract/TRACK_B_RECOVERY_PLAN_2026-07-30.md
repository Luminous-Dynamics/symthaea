# Track B (checkpoint-durability subsystem) — detailed recovery plan (2026-07-30)

Follow-up to `CHECKPOINT_RECOVERY_SCOPE_2026-07-30.md`, per request to investigate and plan
Track B specifically. Everything below was tested directly (dependencies added, modules wired
behind a feature flag, full compile log captured and categorized), then **reverted** — no
committed code changes yet. This is the plan; execution needs a separate go-ahead.

## What was tested

1. Added the 6 dependencies to `Cargo.toml` (5 via `{ workspace = true }` reusing already-vetted
   workspace versions: `blake3`, `getrandom`, `libc`, `zeroize`; 2 new direct deps: `postcard
   = "1.1"`, `fips204 = "0.4"`), all gated behind a new opt-in feature, `checkpoint-durability`.
2. Wired all 13 `checkpoint_*.rs` modules into `lib.rs` behind that same feature (NOT
   `articulatory_quality.rs` — that's Track A, blocked separately).
3. Ran `cargo check -p symthaea-vocal-tract --lib --features checkpoint-durability`.
4. Reverted both files (`git checkout --`) once the error set was fully captured and analyzed.

**Result: dependency resolution is clean (0 errors from the 6 new deps themselves).** The
remaining 47 errors are exactly the two categories the scoping doc predicted — plus one
genuinely new, more consequential finding.

## Design decision proposed (not yet applied): gate Track B behind an opt-in feature

Matches this crate's existing pattern (`hound`, `mel-conversion` are both optional). Keeps the
6 new dependencies — 2 of them (`postcard`, `fips204`) genuinely new to this crate — off the
default build for consumers who only want voice synthesis. **Flagging this as a decision, not
assuming it**: the alternative is making checkpoint-durability always-on. Recommend the opt-in
feature given the subsystem is thematically unrelated to the crate's stated purpose.

## Error breakdown: 48 errors, 3 real categories (not 2)

| Category | Count | Nature |
|---|---|---|
| Missing cross-module re-exports | 55 symbols → 38 resolvable, 17 genuinely absent (see below) | Mostly mechanical |
| Dependency-version API mismatches | 2 | Concrete, fast fixes |
| Missing crate-root utility functions | 2 | Small new code, not wiring |
| Confirmed field mismatch | ≥1 | Needs a design call per-site |

### 1. Re-exports: 38 of 55 are trivial, but 17 point to a genuinely missing "Series 20" file

Mapped every missing symbol to the file that actually defines it (`grep` for `pub struct/enum/
fn/const NAME` across all 13 files). **38 symbols map cleanly** to one of: `checkpoint_gossip_
archive.rs`, `checkpoint_gossip_transport.rs`, `checkpoint_hardware_signing.rs`, `checkpoint_
hybrid_public_verifiability.rs`, `checkpoint_power_loss_operations.rs`, `checkpoint_storage_
evidence.rs`, `checkpoint_series21_public_verifiability.rs`, `checkpoint_transparency_gossip.
rs`, `checkpoint_trusted_time.rs` — these just need `pub use module::{...};` blocks added to
`lib.rs`, matching the crate's existing re-export pattern. Purely mechanical.

**17 symbols do not exist anywhere in this crate, or anywhere in this monorepo** (re-confirmed
via a fresh `grep -rl` sweep and a `git log --all -S` search across the crate's full history —
the only commit ever mentioning e.g. `CheckpointPublicSigningKey` is the same `12ff3e5c88`
patch-series commit that added everything else, as a *reference*, never as a *definition*):

```
CheckpointPublicKeyId, CheckpointPublicSignature, CheckpointPublicSigningKey,
CheckpointPublicVerificationBundle, CheckpointPublicVerificationError,
CheckpointPublicVerifyingKey, MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES, MAX_CHECKPOINT_PUBLIC_SIGNERS,
CheckpointTransparencyLogId, CheckpointSignedTransparencyHead,
CheckpointTransparencyConsistencyProof, CheckpointAuditError, CheckpointAuditExportDurability,
CheckpointKeyAuditExportReceipt, CheckpointHybridDowngradeNegativeSummary,
merge_checkpoint_power_loss_operations_evidence
```

The first group (7 `CheckpointPublic*` symbols + 3 `MAX_CHECKPOINT_PUBLIC_*` consts) is the
most telling: `checkpoint_hybrid_public_verifiability.rs`'s own doc comment says **"Series 20
established a secret-free Ed25519 public-verification bundle. This module adds an ML-DSA-65
overlay..."** — but there is no `checkpoint_series20_*.rs` file anywhere among the 13
delivered. **Series 20 itself was never delivered as part of the patch series** — this is a
real content gap, the same class of finding as Track A's missing gesture layer, just smaller
in scope (one prerequisite file, not an entire subsystem). Similarly, the 3 `CheckpointTransparency
{LogId,SignedHead,ConsistencyProof}` symbols look like they belong to a missing "transparency
log" primitive file that `checkpoint_transparency_gossip.rs` builds on top of. The remaining 4
(audit-error/export-durability/receipt types + the power-loss merge function) are smaller,
plausibly-recoverable-by-writing-directly gaps in already-present files rather than a whole
missing file.

**This changes Track B's effort class**: it's not pure integration labor throughout. ~70% of
the re-export work is mechanical; the remaining ~30% needs either finding Series 20 elsewhere
(searched, not found — would need to be written) or scoping down which of the 13 files can be
wired without it.

**Per-file dependency check DONE (2026-07-30, "please proceed"), correcting the guess above.**
Checked each of the 13 files for a direct reference to any of the 17 missing symbols (first
attempt was a false negative — a zsh word-splitting bug in the check script silently searched
for the entire symbol list as one literal string on every file; caught by spot-checking one
known-true-positive file and re-running with a proper array). **Real result: only 3 of 13
files are genuinely Series-20-independent** — `checkpoint_power_loss_operations.rs`,
`checkpoint_replay.rs`, `checkpoint_storage_evidence.rs` (the latter two need only the small
`lock_exclusive`/no-Series-20 utility fixes already scoped above). The other **10 of 13 files
(77%) directly reference at least one missing Series 20 symbol** —
`checkpoint_audit_archive.rs`, `checkpoint_gossip_archive.rs`, `checkpoint_gossip_transport.rs`,
`checkpoint_hardware_signing.rs`, `checkpoint_hybrid_public_verifiability.rs`,
`checkpoint_power_loss_federation.rs`, `checkpoint_series21_public_verifiability.rs`,
`checkpoint_series22_public_verifiability.rs`, `checkpoint_transparency_gossip.rs`,
`checkpoint_trusted_time.rs`. This is the opposite of the earlier guess — `checkpoint_gossip_
archive.rs` (named above as a plausible Series-20-free candidate) in fact directly imports
`CheckpointPublicSignature`/`CheckpointPublicSigningKey`/`CheckpointPublicVerificationError`/
`CheckpointPublicVerifyingKey`/`MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES`. **Option (b) below (wire
only what doesn't need Series 20) only unlocks 3/13 files — a small win, not the substantial
scope reduction originally hoped.** Real progress on the other 10 files requires option (a):
writing Series 20 itself. That's real cryptographic-adjacent design work (a secret-free
Ed25519 public-verification bundle, per `checkpoint_hybrid_public_verifiability.rs`'s own doc
comment), not a quick fix — matches this crate's existing pattern of checking in with the user
before committing to new crypto-shaped design (see `mycelix-hearth`'s `break_glass.rs` in
`MASTER_ROADMAP.md` for the precedent), not decided unilaterally here.

### 2. Dependency-version API mismatches (2, both concrete)

- `getrandom::fill(...)` — doesn't exist in `getrandom` 0.2 (the workspace-pinned version this
  crate would inherit via `{ workspace = true }`). `fill()` was introduced in getrandom's 0.3
  API redesign; current stable is 0.4.3. **Fix**: pin `getrandom = "0.3"` or `"0.4"` directly
  for this crate rather than using the workspace default (multiple getrandom majors already
  coexist in this workspace's `Cargo.lock`, so this doesn't conflict with anything).
- `postcard::to_stdvec(...)` — not found in postcard 1.1.3 under the `alloc` feature as tested;
  the compiler's own suggested fix is `postcard::to_vec(...)` instead. Needs one more check
  during implementation: whether `to_vec`'s const-generic buffer-size API is a suitable
  replacement for every call site, or whether a different postcard feature flag (`use-std`?)
  actually provides `to_stdvec` and this was just a feature-name mistake in this plan's own
  first attempt (`features = ["alloc"]`).

### 3. Missing crate-root utility functions (2, small new code)

- `crate::effective_uid()` — expected to return the calling process's effective UID. Not
  defined anywhere in this crate. Straightforward to write: `libc::geteuid()` wrapped in a
  small typed function (a few lines).
- `crate::lock_exclusive(&lock)` — expected to take an exclusive advisory lock on a file/path
  argument. Not defined anywhere. Straightforward to write via `libc::flock()` (a few lines),
  matching the pattern already used by similar durability code elsewhere in this monorepo if
  one exists (worth a quick check before writing from scratch).

### 4. Confirmed field mismatch (≥1, expect more once re-exports are fixed)

`CheckpointHardwareSigningPolicy` is missing a `minimum_signing_counter` field some other file's
code expects. This surfaced past the dependency/re-export noise; **more mismatches like this
should be expected** once the compiler can see further into the 13 files with re-exports fixed.
Not yet possible to get a complete list without fixing category 1 first (errors that come after
an unresolved import in the same expression are often suppressed until the import resolves).

## Recommended execution sequence

1. **Decide the Series 20 scoping question first** (this determines how much of Track B is
   "wire it in" vs. "wire in what doesn't need Series 20, defer/write the rest"). Two options:
   - (a) Write a minimal `checkpoint_series20_public_verifiability.rs` from scratch, inferring
     the needed API surface from how the 7+3 missing symbols are *used* across the other 13
     files (a real but bounded design task — the usage sites describe the required shape).
   - (b) Wire in only the checkpoint modules that don't transitively need Series 20 (needs a
     per-file dependency check, not yet done) and leave the rest gated off / documented as
     blocked on a missing prerequisite.
2. Add the 6 dependencies (fixing the 2 version issues found: `getrandom` 0.3+, verify
   `postcard`'s correct feature for `to_stdvec` or switch call sites to `to_vec`).
3. Add the `pub use` re-export blocks for the 38 mechanically-resolvable symbols.
4. Write `effective_uid()` and `lock_exclusive()` (small, self-contained utility functions).
5. Re-run `cargo check --features checkpoint-durability`, fix the next layer of field/type
   mismatches (expect more beyond the 1 confirmed).
6. Once `--lib` compiles clean, run the 29 orphaned tests; fix or honestly disclose failures.
7. Re-check the 13 checkpoint-themed examples individually (not yet diagnosed beyond the shared
   lib-level errors).
8. **Deferred, not blocking**: whether this subsystem should eventually live in its own crate
   rather than inside a voice-synthesis crate — easier to decide once it's a working, tested
   unit than before.

## Effort estimate

Steps 2-4 and 6-7: genuinely mechanical, low-risk, a few focused hours. Step 1 (Series 20) is
the one place this plan can't give a tight estimate without either (a) doing the per-file
dependency check to see how much can be wired without it, or (b) starting the design work for
a minimal Series 20 file — recommend (a) first since it's cheap and might shrink the real scope
of (b) substantially.

## What this document does not do

Makes no code changes (the investigative changes described above were reverted). Does not
decide the Series 20 scoping question. Does not begin implementation. This is the plan
requested before that starts.
