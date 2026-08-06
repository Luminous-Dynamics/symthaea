# Architectural recommendation: the checkpoint-durability subsystem in `symthaea-vocal-tract`

Requested as the closing deliverable of the bounded power-loss cluster work unit (see
`POWER_LOSS_CLUSTER_SEMANTICS_FREEZE_2026-07-30.md`,
`CHECKPOINT_DURABILITY_INTEGRATION_STATUS_2026-07-30.md`). Track B (further wiring) is
deliberately **stopped** as of this document, per explicit instruction. This is a
recommendation only — no extraction or deletion is performed here.

## What the subsystem actually is

`symthaea-vocal-tract`'s `src/` directory contains 15 `checkpoint_*.rs` files (~10,833 lines,
**51% of the crate's 21,027-line total**) plus 12 `CHECKPOINT_*_CAMPAIGN.md` design docs and a
dozen `checkpoint_*`-prefixed examples, all added in a single commit
(`12ff3e5c88`, "apply symthaea vocal tract patch series", 2026-07-20).

Reading the campaign docs (`CHECKPOINT_POWER_LOSS_OPERATIONS_CAMPAIGN.md` et al.) makes the
actual subject matter clear: this is a **generic cryptographic attestation protocol for proving
that a physical durability/power-loss test campaign was honestly conducted** — independent
signing authorities for storage profiles vs. operations vs. federation vs. hardware time, lab
manifests with facility bindings, trial leases with monotonic journals, multi-lab evidence
merging, gossip-based federation of results, and (in the still-unwired tier) a Merkle
transparency log and post-quantum hardware-backed signing. It is a **hardware/lab
test-attestation system**, structurally similar in spirit to the DKG/threshold-signing work in
`mycelix-governance` or the multi-authority credential work in `mycelix-identity`, not to
anything in speech synthesis.

**It has zero connection to articulatory voice synthesis** — no shared types, no shared
dependencies until this feature was added (`ed25519-dalek`, `blake3`, `libc`, `getrandom`,
`zeroize`, `postcard`, and — if the remaining 7 files are ever finished — `fips204`, none of
which the rest of the crate touches), and the crate's own `lib.rs` comment already says so
explicitly ("Thematically unrelated to voice synthesis; opt-in only").

## Correcting an earlier claim in this audit's own memory

An earlier note in this audit (`memory/symthaea_vocal_tract_claim_audit_jul29.md`, "Follow-up
investigation") argued the vocal-tract instance was "an incomplete application of an otherwise-
standard pattern, not foreign/misplaced work," citing `symthaea-broca`, `symthaea-vision-
manifold`, and `symthaea-subterranean` as siblings with "their OWN checkpoint modules properly
wired into their `lib.rs`." **Checked directly for this document and found this claim does not
reproduce**: `find . -name 'checkpoint_*.rs'` across the entire `symthaea/` monorepo returns
matches only inside `symthaea-vocal-tract/src/` and `symthaea-vocal-tract/examples/` (plus one
unrelated `symthaea-music-theory/examples/support/` hit). No other crate in this ~1.68M-line
monorepo has a `checkpoint_*.rs` module. The pattern is **not standard anywhere else** — this
is genuinely one-off content, dropped into this specific crate by a "patch series" application
process, not a partially-completed instance of an established convention. The memory file has
not yet been corrected to reflect this — flagging it here since it materially changes the
"is this foreign work" judgment this recommendation depends on.

## Current state (as of this pass)

- 8/15 files wired, compiling, and tested (154/154 lib tests with `--features
  checkpoint-durability`, clippy/rustfmt clean, zero effect on the 106/106 default-feature
  tests).
- Real, working Ed25519 public-verification crypto (`checkpoint_series20_public_verifiability.rs`,
  16 tests including tamper/forgery rejection).
- Real, working power-loss operations/federation evidence validation (this pass's work — 8 new
  adversarial tests, 1 end-to-end integration test).
- 7/15 files remain unwired, blocked on either a not-yet-built RFC-6962 transparency log or the
  not-yet-added `fips204` post-quantum crate.
- **Zero consumers anywhere else in the monorepo** — nothing outside
  `symthaea-vocal-tract/src/` references any `checkpoint_*` type (checked via grep across the
  whole tree). It is entirely self-contained and entirely inert unless a consumer opts into the
  feature and calls it directly, which nothing currently does.
- It is correctly feature-gated (`checkpoint-durability`, off by default) — it costs nothing at
  compile time or runtime for any consumer of the crate's actual speech-synthesis API today.

## Options considered

**A. Keep growing it in place (finish all 15 files, add `fips204`, build the transparency
log).** Rejected. This would mean a crate whose stated purpose is "LTC-driven articulatory
synthesis" eventually carrying more lines of hardware-test-attestation cryptography than voice
code, with a dependency footprint (post-quantum ML-DSA, Merkle logs) that has nothing to do
with speech. Every future contributor opening this crate to work on formants or HDC encoding
has to first understand what a "checkpoint power-loss lab manifest" is and why it's here. This
is the option the user's own bounding instruction ("do not add fips204, design the transparency
log...") already declines for this pass, and nothing about finishing it would change the
underlying category error.

**B. Retire it (delete the 15 files, the campaign docs, and the feature).** Rejected as the
default, but not unreasonable if no real consumer ever materializes. The work is real, tested,
and non-trivial (Series 20's cryptography, the power-loss evidence-merging logic, ~10.8K lines
overall) — deleting it destroys that investment for no gain, when the actual problem is
*location*, not *existence*. Retirement is the right call only if nobody can name a plausible
consumer (see below) — worth revisiting if that stays true.

**C. Extract into its own crate.** Recommended. Move all 15 `checkpoint_*.rs` files, the 12
campaign docs, and the `checkpoint-durability` feature's dependencies out of
`symthaea-vocal-tract` into a new standalone crate (e.g. `symthaea-checkpoint-durability` under
`crates/domains/` or `crates/core/`, matching this monorepo's existing crate-per-concern
pattern). This:
- Removes ~51% of unrelated code from a crate that should be about speech synthesis, with zero
  loss of the real engineering work already done (Series 20 crypto, power-loss evidence
  validation, 154 tests all move intact).
- Makes the subsystem honestly named and discoverable for what it is, rather than hidden inside
  a voice crate where nobody would think to look for hardware-attestation crypto.
- Removes the compile-time/dependency-graph coupling: `symthaea-vocal-tract` currently pulls
  `ed25519-dalek`/`blake3`/`libc`/`postcard`/`zeroize` into its own `Cargo.lock` footprint the
  moment the feature is touched, for a capability its own consumers never asked for.
- Leaves the door open for a real consumer to use it (see below) without forcing anyone
  building or auditing voice synthesis to reason about lab test-attestation protocols first.

## Is there a plausible consumer? (bears on whether extraction is worth doing now)

Not identified with confidence in this pass — this is exactly the kind of judgment call that
should be surfaced, not assumed. Candidates that exist in this monorepo and share a *conceptual*
need (proving a durability/checkpoint claim was honestly produced) without necessarily sharing
this *specific* protocol:
- GPU training checkpoint survival (`CLAUDE.md`'s `with-heartbeat.sh` note — real incidents of
  training jobs losing their target dir mid-run) is about infrastructure robustness, not
  cryptographic attestation of a lab test; a weak match.
- Robotics/hardware qualification tracks (`symthaea-manipulator`, the nuclear-energy detector
  work, `SYMTHAEA_ROBOTICS_IMPROVEMENT_PLAN`'s "Part II hardware/qualification tracks") are the
  closest conceptual fit — "did this physical unit actually survive a power-loss test, with
  independently verifiable evidence from multiple labs" is exactly what this subsystem's
  campaign docs describe, and those tracks are explicitly still open/unstarted per
  `MASTER_ROADMAP.md`.
- `nixward`'s `ConfigWriter` (git-backed, atomic, real durability guarantees for NixOS config
  writes) is a software-only durability concern with no lab/hardware framing — not a fit.

No commit, doc, or issue anywhere in this monorepo currently names this subsystem as intended
for any of these. It reads as generated/templated content ("apply X patch series" is a
recurring commit-message pattern in this monorepo for what appear to be batch-applied spec+code
bundles) that landed in this crate without an identified downstream need — which is itself
useful information: **the extraction shouldn't be scheduled as urgent work**, since there is no
known blocked consumer waiting on it. It should be treated as "real, tested code sitting in the
wrong place, worth relocating whenever someone next touches this area" rather than "a blocking
architectural debt."

## Recommendation

**Extract when convenient, not urgently.** Concretely:
1. Do not invest further effort finishing the remaining 7/15 files (`fips204`, transparency
   log) inside `symthaea-vocal-tract` — this pass already stops here per instruction, and
   nothing above changes that.
2. The next time someone works in this area (whether that's finishing Track B's remaining
   files, or doing general crate hygiene), move the `checkpoint_*` module family and its
   feature into a new crate rather than continuing to build it out in place. This is a
   mechanical `git mv` + `Cargo.toml` split, not a redesign — the code itself doesn't need to
   change, only its address.
3. If, when that time comes, no consumer has appeared and none is on the near-term roadmap,
   reconsider option B (retire) at that point rather than extracting speculatively — a crate
   with zero consumers is only marginally better than a misplaced module with zero consumers.
4. Either way, this is not blocking — `symthaea-vocal-tract`'s actual speech-synthesis surface
   (encoder/controller/fep/pipeline/speech/metrics/phonetics) is fully independent of this
   subsystem today (feature-gated, zero shared types), so its presence costs nothing for anyone
   working on voice synthesis specifically.
