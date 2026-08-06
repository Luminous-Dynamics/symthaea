# symthaea-vocal-tract — orphaned-code recovery scope (2026-07-30)

Follow-up to `VERIFICATION_LEDGER_2026-07-29.md`, per the user's request to "scope out the
recovery work properly" before any implementation. Every claim below was directly verified
(grep, git history, `cargo check`), not assumed. **No production code changes are proposed to
land from this document by itself** — it's the scoping pass the ledger's summary flagged as
needed before authorizing further fixes.

## Headline finding: this is not one recovery task, it's three independent tracks

The 14 undeclared files + 15 broken examples split cleanly into three buckets with very
different risk profiles, effort, and even different **kinds** of work (integration vs. new
design vs. a false alarm). Treating this as a single "wire it all in" task would conflate a
tractable afternoon of integration work with a genuine feature-design decision.

| Track | What | Files | Effort class |
|---|---|---|---|
| **A** | Physical gesture-synthesis layer | `articulatory_quality.rs` + `cognitive_physical_voice.rs` example + 5 placeholder docs + 2 empty bundle dirs | **New design/implementation, not recovery — the code to recover does not exist** |
| **B** | Generic checkpoint-durability/attestation subsystem | 13 `checkpoint_*.rs` files + 13 checkpoint-themed examples | **Real, bounded integration work** |
| **C** | `f1_probe.rs` | 1 example | **Likely not actually broken — see below** |

## Track A: the "gesture" layer is genuinely missing, not just unwired

`articulatory_quality.rs` itself (the one orphaned file that's actually on-topic for a
voice-synthesis crate) imports `crate::GestureFrame` and
`crate::{ArticulatoryGestureScheduler, ArticulatoryScore, TimedPhoneme, UnitInterval}`.
`cognitive_physical_voice.rs` additionally imports `ArticulatoryQualityRequirements`,
`ArticulatoryTimingConfig`, `BranchedWaveguideConfig`, `BranchedWaveguideV2`, `GesturePlanner`,
`IdentityAnatomy`, `IdentityPhysiology`, and calls `pipeline.bootstrap_gesture_projection(...)`.

**Searched the entire monorepo** (`grep -rl`, all `.rs` files, plus the crate's own sibling
`.tar.gz` snapshot) for every one of these type names. Result: **none of them exist anywhere**,
except `TimedPhoneme`, which exists in a *different* location
(`symthaea/src/voice/articulatory_synthesizer.rs`, the main crate, not this sub-crate) as an
apparently unrelated, independently-defined struct — not a match to import, a naming collision
at best. This directly confirms the external critique's most specific claim ("the example
imports APIs that do not exist in the supplied crate") and extends it: it's not just the
example — the crate's OWN `articulatory_quality.rs` file depends on the same missing layer.

This also explains the crate's other placeholder artifacts, all of which reference a "Series
23" campaign: `SERIES23_EVIDENCE.md`, `SNAPSHOT_STATUS.md`, `SNAPSHOT_INTEGRITY.md`,
`tools/verify_snapshot.py`, `tools/generate_series23_matrix.py` (all literal `// placeholder`),
plus `combined/`, `symthaea-vocal-tract-series-42-43-bundles/`,
`symthaea-vocal-tract-series-48-49-bundles/` (README-only, claiming patch archives/snapshots
"included beside them" that don't exist). The gesture/physical-renderer layer these all
describe was, as far as this repository shows, **never delivered** — not lost, not misplaced,
never written into this codebase.

**Recommendation**: this is not something to "recover." It requires a product decision:
1. **If physical gesture-based synthesis is still wanted**: this is new feature design and
   implementation (a gesture scheduler, articulatory trajectory scoring, branched waveguide
   synthesis, identity anatomy/physiology modeling) — scope unknown without its own design pass,
   comparable to standing up a new subsystem, not fixing one.
2. **If it's not wanted / superseded by the Kokoro+WORLD+Vocos singing-voice research
   direction** (see `symthaea_diffsinger_singing_voice_gates_jul26.md`) **or by this crate's own
   working HDC/LTC formant-synthesis path**: retire cleanly — delete
   `articulatory_quality.rs`, `cognitive_physical_voice.rs`, the 5 placeholder files, and the 2
   empty bundle directories (with their READMEs), and correct
   `COGNITIVE_PHYSICAL_VOICE_CAMPAIGN.md`'s false "frozen implementation" claims (11 of them,
   per the critique) to state plainly that this line was never completed.

**Not this session's call to make** — flagging for explicit direction.

## Track B: the checkpoint-durability subsystem is real, bounded integration work

The 13 `checkpoint_*.rs` files (audit archive/retention receipts, gossip transport/archive,
hardware-custody attestation, hybrid classical+post-quantum public verification, federated
power-loss campaign authorization, authenticated operator workflows, restart-durable replay
protection, storage-profile evidence, transparency-head gossip/split-view detection, multi-
authority trusted time, and two "portable bundle" summarizers for series 21/22) are a
coherent, self-contained **generic checkpoint-durability and public-verifiability framework** —
thematically unrelated to voice/articulatory synthesis, but real, substantial, and (per module
doc comments) carefully reasoned about threat models (e.g. explicit "this does not claim X"
scoping statements throughout).

**Why this is recoverable, not speculative**: all 14 files were added in a single commit,
`12ff3e5c88` ("apply symthaea vocal tract patch series", 2026-07-20) — the same commit-message
convention (`git log --oneline --all | grep "patch series"`) used for Pulse and lawful-identity
hardening campaigns elsewhere in this monorepo. Three sibling crates (`symthaea-broca`,
`symthaea-vision-manifold`, `symthaea-subterranean`) received the identical kind of campaign and
**do** have their checkpoint modules properly wired into their own `lib.rs`. This is the one
place the wiring step was never finished, not foreign or abandoned-by-design work.

### Confirmed concrete requirements (tested directly, then reverted)

1. **6 external crates need adding to `Cargo.toml`**: `zeroize`, `libc`, `blake3`, `postcard`,
   `getrandom`, `fips204`.
   - **5 of these are zero-risk**: `blake3` (1.8.5), `getrandom` (multiple versions already
     present: 0.1.16/0.2.17/0.3.4/0.4.2 — pick the one this code actually needs), `libc`
     (0.2.186), `postcard` (1.1.3), `zeroize` (1.9.0) are **already in the workspace
     `Cargo.lock`**, used elsewhere in this monorepo — adding them to this crate's `Cargo.toml`
     is just referencing already-vetted dependencies, no new supply-chain surface.
   - **`fips204` is genuinely new to this workspace** (not found in `Cargo.lock` under that name
     or `ml-dsa`) — a post-quantum ML-DSA signature crate. This one needs its own quick vetting
     pass (license, maintenance status, crates.io page) before adding, per this project's
     standing practice for new dependencies. Given `checkpoint_hybrid_public_verifiability.rs`'s
     own doc comment ("Series 20 established a secret-free Ed25519 public-verification bundle.
     This module adds an ML-DSA-65 overlay"), this is very likely the same ML-DSA family already
     used by mycelix-identity/governance elsewhere in the broader monorepo (see
     `mycelix-zkp-core`, `ml_dsa` references in `MASTER_ROADMAP.md`'s P0-#1 row) — worth checking
     whether an existing, already-integrated ML-DSA crate can be reused instead of adding a
     second, different one.

2. **Cross-module re-exports are missing.** The 13 files reference each other via `crate::
   TypeName` paths (e.g. `checkpoint_transparency_gossip.rs` expects `crate::
   CheckpointPublicVerifyingKey` from `checkpoint_hybrid_public_verifiability.rs`), matching the
   existing pattern already used for the crate's real modules (`pub use controller::{...};` at
   the top of `lib.rs`). Wiring needs matching `pub use checkpoint_X::{...};` blocks added
   alongside the 13 `pub mod` declarations, not just the `pub mod` lines alone.

3. **At least one field mismatch confirmed**: `CheckpointHardwareSigningPolicy` is missing a
   `minimum_signing_counter` field some other file's code expects. Given the compile only got
   partway before the missing-dependency errors dominated the output, **more mismatches like
   this should be expected** once dependencies and re-exports are fixed and the compiler can
   see further into the files.

4. **13 checkpoint-themed examples** will need their own re-check once the library itself
   compiles — not yet individually diagnosed beyond the shared lib-level errors above.

5. **29 orphaned `#[test]`s** exist in these files — once compiling, run them to see how many
   actually pass. Structural compile success is necessary but not sufficient; these have never
   been executed since being committed.

### Recommended sequence if this track is authorized

1. Vet `fips204` (or find/reuse the monorepo's existing ML-DSA dependency instead).
2. Add the 6 dependencies to `Cargo.toml`.
3. Add the 13 `pub mod` declarations + matching `pub use` re-exports to `lib.rs`.
4. Iteratively fix compile errors (`cargo check --all-targets --keep-going` in a loop) — expect
   several more rounds beyond what's confirmed above; the missing-dependency errors were
   dominating the log and likely masked further type/field mismatches.
5. Once `--lib` compiles clean, run the 29 orphaned tests; fix or honestly disclose failures.
6. Re-check the 13 checkpoint-themed examples individually.
7. **Deferred, not blocking**: whether this ~9,400-line subsystem should eventually live in its
   own crate rather than inside a voice-synthesis crate is a real question, but easier to answer
   *after* it's a working, self-contained unit than before — recommend deciding that later, not
   as a precondition to wiring it in.

**Effort class**: real but bounded integration labor — no new design needed, the actual logic
already exists and reads as carefully thought through. Expect several iterative fix-check cycles,
not one pass.

## Track C: `f1_probe.rs` is very likely not actually broken

Unlike every other failing example, `f1_probe.rs` imports only two already-wired, real modules
(`symthaea_vocal_tract::formant_extraction`, `symthaea_vocal_tract::types`) and uses the
`hound` crate for WAV I/O — which is **already an existing, correctly-defined optional feature**
in this crate's `Cargo.toml` (`hound = ["dep:hound"]`), not a missing dependency. The example's
own doc comment even documents its intended invocation: `cargo run -p symthaea-vocal-tract
--example f1_probe --features hound -- corpus_dir/*.wav`. The original ledger's "15/15 examples
fail" check ran `cargo check --all-targets` **without** `--features hound`, which would
correctly fail to compile any `hound`-gated code — this looks like a false alarm from the
audit's own feature-flag scope, not a real crate defect.

**Verified**: `cargo check -p symthaea-vocal-tract --example f1_probe --features hound` —
`Finished` cleanly, 0 errors. This example was never actually broken; the Phase-1 ledger's
"15/15 examples fail" count should be read as **14 real failures + 1 feature-flag false alarm**.
No fix needed here at all.

## What this document does not do

Makes no code changes. Adds no dependencies. Does not decide Track A's product question. Does
not begin Track B's integration work. This is the scoping artifact requested before any of that
starts.
