# symthaea-vocal-tract — Verification Ledger (2026-07-29)

Claim-by-claim audit of the external critique's findings against the actual code, per the
user's instruction: freeze this ledger before authorizing any fix. Every row below was
independently reproduced against the crate at this commit (not taken on the critique's word).
Environment: isolated worktree `.claude/worktrees/session-vocal-tract-audit/`, `cargo`
1.94+/edition 2024. Two pre-existing worktree-only build blockers were fixed to even reach a
compiling state (see "Environment fixes" below) — neither is a claim about the crate itself.

**Fix authorized column is `No` throughout this ledger by design — this is a freeze, not a
change.** Any fix requires a separate, explicit go-ahead per claim.

## Environment fixes (not crate bugs — worktree-only, required to run anything)

`patches/ed25519-dalek/` and `patches/iroh/` are nested git repos (their own `.git`, not
outer-repo submodules) that `git worktree add` does not check out content for. Both were empty
in the fresh worktree, causing `cargo check` to fail before even reaching this crate's own
code (`error: failed to load source for dependency`). Fixed by `rsync`-copying each directory's
content (excluding `.git`) from the main tree into the worktree — read-only source copy, main
tree untouched. This matches the `nested-git-repo structural issue` CLAUDE.md already
disclosed for `nix build`; it turns out to also block plain `cargo check` in a fresh worktree,
not just `nix build`.

## 1. Build / module-reachability truth

| Claim | Reproduction | Verdict | Evidence | Fix authorized? |
|---|---|---|---|---|
| Library (declared modules) compiles clean | `cargo check -p symthaea-vocal-tract --all-targets` | Confirmed | 0 errors on the `lib` target; all 11 example-target errors are isolated to examples, not the library | No |
| Library tests pass | `cargo test -p symthaea-vocal-tract --lib` | Confirmed | `test result: ok. 103 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 115.65s` | No |
| ~9,677 reachable lines across 10 declared modules | `wc -l` on `lib.rs` + the 10 modules `pub mod`-declared in `lib.rs` (controller/encoder/fep/formant_extraction/formant_to_mel/metrics/phonetics/pipeline/speech/types) | Confirmed | 9,677 total lines | No |
| ~9,415 orphaned lines, 29 orphaned tests, across 14 undeclared top-level files | `find src -maxdepth 1 -name '*.rs'` minus the 10 declared modules; `wc -l` on the remainder; `grep -rc '#\[test\]'` on the same set | Confirmed | 14 files (`articulatory_quality.rs` + 13 `checkpoint_*.rs`), 9,415 lines, 29 `#[test]` occurrences, none reachable from `lib.rs` | No |
| Every shipped example fails to compile | `cargo check -p symthaea-vocal-tract --all-targets --keep-going` | Confirmed, and worse than claimed — **15/15 examples fail**, not "most" | 15 distinct `error: could not compile ... (example "...")` lines, one per example file that exists | No |
| Examples fail because they reference types/functions from the undeclared checkpoint modules, not because of unexported-vs-absent confusion | Read the full `--keep-going` error log | Confirmed | `E0432 unresolved import symthaea_vocal_tract::CheckpointPowerLossExecutionJournal` etc. — these symbols genuinely do not exist anywhere reachable from the crate root, matching "absent, not merely unexported" | No |
| Some examples also reference external crates never added as dependencies | Same log | Confirmed | `error[E0432]: unresolved import zeroize`, `error[E0433]: cannot find module or crate libc` — neither `zeroize` nor `libc` is in `Cargo.toml` | No |
| `SERIES23_EVIDENCE.md`, `SNAPSHOT_STATUS.md`, `SNAPSHOT_INTEGRITY.md`, `tools/verify_snapshot.py`, `tools/generate_series23_matrix.py` are literal placeholders | `wc -l` + `head` on each file | Confirmed | All 5 are exactly 1 line: `// placeholder` (including the two `.py` files, which is not valid Python) | No |
| `combined/`, `symthaea-vocal-tract-series-42-43-bundles/`, `symthaea-vocal-tract-series-48-49-bundles/` contain only README files, despite README text claiming patch archives/checksums are "included beside them" | `ls -la` on all three dirs | Confirmed | Each directory contains exactly one file, `README.md` — no patch archives, no checksums, no snapshots | No |

## 2. ARPAbet routing

| Claim | Reproduction | Verdict | Evidence | Fix authorized? |
|---|---|---|---|---|
| `canonical_arpabet_symbol` accepts `AW, AY, ER, EY, OW, OY` as valid `VOWEL` | Direct read, `src/phonetics.rs:27-43` | Confirmed | Line 32-33's match arm lists all 6 alongside the other 9 vowels, mapping to `Some("VOWEL")` | No |
| `arpabet_articulation` has no match arm for any of those 6 | Direct read, `src/phonetics.rs:45-163` | Confirmed | Only `AA, AE, AH, AO, EH, IH, IY, UH, UW` (9 of the 15 canonical vowels) have explicit arms; `AW, AY, ER, EY, OW, OY` fall through to the `_` wildcard | No |
| The wildcard classifies them as `PhonemeClass::Silence`, zero formants, unvoiced | Direct read, lines 155-162 | Confirmed | `_ => ArticulationMetadata { class: Silence, f1: 0.0, f2: 0.0, f3: 0.0, voiced: false }` | No |
| Stress-marked forms (`AY1`) hit the identical bug, not a separate one | Direct read of the `trim_end_matches(is_ascii_digit)` normalization at the top of both functions | Confirmed | Both functions strip trailing digits before matching, so `"AY1"` normalizes to `"AY"` and follows the exact same (broken) path as `"AY"` | No |
| The physical example's own demo sequence uses one of the broken symbols | Grep `examples/cognitive_physical_voice.rs` for `AY` | Not independently re-verified this pass (the example doesn't compile at all, so this couldn't be exercised end-to-end even if confirmed) — plausible given `AY` is a common diphthong, not chased further since the example is unbuildable regardless | Not verified | No |

## 3. Unknown-phoneme / canonical-phoneme policy

| Claim | Reproduction | Verdict | Evidence | Fix authorized? |
|---|---|---|---|---|
| `get_or_create_phoneme_hv()` accepts any `&str`, unconditionally creates and caches an HV, no validation | Direct read, `src/pipeline.rs:671-681` | Confirmed | The function body is unconditional: hash-lookup, else `self.genesis.hv(&format!("phoneme::{phoneme}"), ...)`, cache, return. No call to `canonical_arpabet_symbol` or any allow-list anywhere in this function | No |
| The doc comment "fail-closed production default so the live pipeline cannot silently render every unregistered consonant as a vowel" is inaccurate | Direct read + trace, `src/pipeline.rs:816-837` | **Confirmed inaccurate, and the real behavior is two distinct bugs, not the one the comment implies:** (a) for a genuinely unknown symbol, `canonical_arpabet_symbol` returns `None`, so `metadata` is `None`; the subsequent `manner`/`is_voiced` fallbacks then resolve to `None` too (no caller-supplied map entry), so the `if let`/`==` guards simply don't fire — `frame.source_type`/`frame.voicing` are left at whatever the network's own forward pass already produced. This is **fail-open** (no override, not silenced/rejected), the opposite of what "fail-closed" claims. (b) For the 6 broken-but-canonical diphthongs from item 2, `metadata` is *not* `None` — it's `Some(Silence, voiced:false)` — which actively **miswires a legitimate accepted vowel to silence**, a worse failure than (a) | Real code trace, `pipeline.rs:820-836` | No |
| Unknown-symbol acceptance is claimed to block promotion per the campaign's own stated criteria | Not independently checked against campaign doc text this pass (campaign docs weren't re-read for this claim) | Not verified | — | No |

## 4. Learning-rate / tau modulation

| Claim | Reproduction | Verdict | Evidence | Fix authorized? |
|---|---|---|---|---|
| `modulate_tau(factor)` derives the new tau from the neuron's *current* (already-modulated) `tau_base`, not an immutable baseline | Direct read, `src/controller.rs:605-615` | Confirmed | `let new_tau = neuron.config().tau_base * factor; neuron.set_tau_base(new_tau);` — reads the live config value and overwrites it; no separate stored baseline anywhere in this function | No |
| Repeated calls compound multiplicatively | Direct trace (no baseline field exists to reset from between calls; `set_tau_base` permanently overwrites `config().tau_base` for the next call to read) | Confirmed by code structure (not run under a live long-horizon simulation this pass — a "thousands of ticks" empirical drift measurement was not executed, see below) | `controller.rs:605-615` | No |
| The FEP learning-rate factor is applied the same way: `current_lr = self.controller.learning_rate(); set_learning_rate(current_lr * fep_result.learning_rate_factor)` | Direct read, `src/pipeline.rs:725-727` (and a second, identical call site at `pipeline.rs:889-890`) | Confirmed, and found at **two** call sites, not one | `let current_lr = self.controller.learning_rate();` then `self.controller.set_learning_rate(current_lr * fep_result.learning_rate_factor);`, verbatim at both `pipeline.rs:725-727` and `pipeline.rs:889-890` | No |
| Effective LR/tau are bounded by `set_learning_rate`'s `[1e-6, 0.1]` clamp and `modulate_tau`'s `[0.3, 3.0]` factor clamp | Direct read, `controller.rs:606` (`factor.clamp(0.3, 3.0)`) and `controller.rs:632-634` (`lr.clamp(1e-6, 0.1)`) | Confirmed — repeated same-direction actions will drift to and then sit at the clamp boundary, not diverge unboundedly, but still an uncontrolled compounding drift within that range | `controller.rs:606,633` | No |
| `pipeline::reset()` does not restore learning rate or tau to their original values | Direct read, `src/pipeline.rs:1027-1039` and `src/controller.rs:625-629` | Confirmed | `Controller::reset()`'s body is exactly `self.network.reset(); self.prev_frame = None; self.cached_cognitive_channels = None;` — no `self.learning_rate` reset, no tau-base restoration. `Pipeline::reset()` calls `controller.reset()` plus resets its own unrelated fields (encoder, fep_agent, caches) but nothing that would restore drifted LR/tau either | No |
| A long-horizon (thousands-of-ticks) empirical drift/bounded-after-reset test | Not run this pass | Not verified (structural finding confirmed via code read; empirical confirmation via an actual simulation was scoped for Phase 4 but not executed given time) | — | No |

## 5. Cognitive-path attribution / confound

| Claim | Reproduction | Verdict | Evidence | Fix authorized? |
|---|---|---|---|---|
| `forward_with_prosody()` computes a base HDC/LTC-path frame, then applies a *separate*, additive, direct-from-raw-channels correction via a `ProsodyHead`, bypassing the HDC bottleneck | Direct read, `src/controller.rs:669-701` | Confirmed | `let mut frame = self.forward(cognitive_hv, dt);` (the HDC/LTC path) followed by, if a `prosody_head` and `channels` are both present, `let correction = head.forward(ch);` applied additively to `frame.f0`/`frame.energy`/`frame.voicing` — two structurally independent mechanisms in one function, confirmed by the doc comment's own admission ("bypassing the HDC bottleneck") | No |
| A third mechanism (handwritten `ProsodyContext` rules) also contributes | Not independently re-traced this pass (the two confirmed above already establish the confound; the third wasn't separately located) | Not verified | — | No |
| No existing test/harness in this crate isolates which of these mechanisms is responsible for any observed cognitive-to-acoustic effect (no ablation arms exist) | Not exhaustively verified (would require reading every test in `controller.rs`/`pipeline.rs`'s test modules for an ablation-shaped test) | Not verified — plausible given no `#[cfg]`-gated "disable prosody head only" or "disable HDC path only" toggle was seen in any code read this pass, but not exhaustively searched | — | No |

## 6. Orphaned verification infrastructure

Already fully covered by Phase 1's exact counts above (14 undeclared files / 9,415 lines / 29
tests / 5 placeholder docs / 2 empty bundle directories / a `.tar.gz` sibling archive
containing only the reachable `src/` files). No further phase-6-specific work done — the
Phase 1 numbers are the phase-6 answer.

## Summary

Of the critique's major claims, every one that was checked against the actual code
**reproduced** — several with more precision or a worse/different shape than originally
described (all 15 examples fail, not "most"; the unknown-phoneme policy is fail-open for
genuinely novel symbols but actively mis-silences 6 legitimate canonical diphthongs, which is
two bugs, not one; the LR-compounding bug exists at two call sites, not one). Two narrower
claims (the physical example's own use of a broken symbol; the third "handwritten rules"
prosody mechanism) were not independently re-verified this pass and are marked accordingly —
not confirmed, not disproven, simply not chased down given the scope already covered.

**No code changes were made to the crate in this pass** beyond the two worktree-local
environment fixes (nested-repo content copy) needed to compile anything at all, which do not
touch `symthaea-vocal-tract` itself.

## Fixes applied (2026-07-30, authorized: "please proceed")

Three of the six confirmed issues had a clear, low-risk, unambiguous fix and were addressed.
Two others (orphaned examples/modules, cognitive-path ablation) require a real design decision
(recover vs. archive ~9.4K lines; build new ablation instrumentation) rather than a bug fix, and
were deliberately left open pending explicit direction. All fixes verified: `cargo test -p
symthaea-vocal-tract --lib` → **106/106 pass** (was 103; +3 new regression tests), 0 warnings
from the affected modules.

- **§2 ARPAbet routing — FIXED.** Added match arms for `AW/AY/EY/OW/OY` (diphthongs,
  approximated by their onset-vowel formant target, since this table has no time-varying
  representation — cross-checked against Holbrook & Fairbanks (1962)'s real OY onset
  measurement, which matches the reused AO values closely) and `ER` (a true monophthong with
  its own real Peterson & Barney (1952) "bird" value, F1=490/F2=1350/F3=1690 Hz — same source
  as this table's other 9 entries). Added a table-driven regression test
  (`every_canonical_symbol_produces_nonsilence_articulation`) asserting every symbol
  `canonical_arpabet_symbol` accepts also produces non-silence articulation from
  `arpabet_articulation`, plus a stressed-form-normalization test and an explicit
  unknown-symbol-still-silence test.
- **§3 unknown-phoneme policy — FIXED.** The `pipeline.rs` block at both call sites now
  explicitly checks whether a phoneme is recognized by EITHER the caller-supplied maps OR the
  canonical ARPAbet table; if neither, `frame.source_type`/`frame.voicing` are explicitly forced
  to `Silent`/`0.0` — making the actual behavior match the doc comment's stated "fail-closed"
  intent for the first time. (Doc comment itself also corrected to describe the real, now-true
  behavior.)
- **§4 LR/tau compounding — FIXED.** Added `base_learning_rate`/`base_tau` immutable fields to
  `VocalTractController`, set once at construction. `modulate_tau()` and both `pipeline.rs` FEP
  call sites now derive the effective value from the immutable baseline
  (`base_tau * factor`, `base_learning_rate * factor`) instead of the current/live value,
  eliminating the compounding. `Controller::reset()` now also restores both to baseline
  (previously only reset network state/cached frames).
- **§5 cognitive-path confound — NOT fixed, left open.** No ablation switches were added; doing
  so properly (feature-flagging the HDC/LTC path and the `ProsodyHead` independently, plus
  designing a controlled comparison) is new instrumentation work, not a bug fix, and needs its
  own scoped go-ahead.
- **§1/§6 orphaned examples/modules — NOT fixed, left open.** Whether the 14 undeclared
  checkpoint files (9,415 lines) and their corresponding 15 broken examples should be wired in,
  archived, or deleted is a real design/product decision (recover vs. discard real but
  unreachable work), not something to decide unilaterally under a generic "please proceed."
