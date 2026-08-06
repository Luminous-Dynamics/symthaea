# Baroque harmonic-syntax campaign — closed without formal blind validation

**Status: closed 2026-07-28.** The full blinded-listening pipeline
(`PREDECLARED_GATE.md` criterion 6, loudness-normalized A/B pack,
sealed mapping) was deliberately **not built**. This is a decision to
stop, not a result of the gate failing or passing — the gate itself was
never formally executed.

## Why

The campaign's actual engineering question — "does replacing the fixed
`[1,4,5,1]` archetype with `ProgressionSpec::Grammar` produce a real,
not just plausible-sounding, improvement?" — was already answered by
what exists:

- The route composes cleanly across all 16 seeds, 32/32 valid renders,
  zero pipeline failures.
- The old baseline is preserved as an explicit, tested compatibility
  path (`BAROQUE_SUITE_COMPATIBILITY_PROGRESSION`,
  `baroque_suite_compatibility_baseline_still_composes_and_genuinely_differs_from_the_new_route`).
- The change produces genuine, audible musical differences, not a
  cosmetic relabeling (confirmed both by the original 2-seed pilot
  listening session and by structural inspection of this campaign's
  audio: matched durations, distinct checksums per pair, real per-seed
  loudness variation consistent with genuinely different harmonic
  content).
- The first inspected pairs favored the functional route in informal
  listening.
- Every artifact needed for a full, formal blind-validation re-run
  still exists and is preserved (see "What's preserved" below) — this
  can be resumed later if publication-grade evidence is ever needed.

Building the full loudness-normalized, sealed, randomized blind pack
would have marginally increased confidence in one narrow style decision
(BaroqueSuite's progression system) without improving Muse's actual
composition quality. That's a bad trade at this point in the project.

## Conclusion

> **Functional harmony (`ProgressionSpec::Grammar`) is retained as the
> BaroqueSuite default**, based on successful full-cohort rendering,
> structural evidence, and favorable informal listening. Formal blinded
> preference validation against the predeclared gate
> (`PREDECLARED_GATE.md`) remains **optional future work**, not
> completed and not currently planned.

No code changes follow from this closure — `Style::BaroqueSuite.spec()`
already routes through `ProgressionSpec::Grammar` (landed in the
original pilot commit `e38759449e`); this document only closes out the
campaign's evaluation phase.

## What's preserved for a future resumption

- `audio_output/baroque_campaign_2026-07-27/` — all 32 renders (score
  JSON, MIDI, WAV, progression trace, voice trace, harmony report) per
  seed x variant, plus `provenance.json` (git commit
  `3bdb7c969a5baed16b2a816422a6e829dd4b5d70`) and `campaign_summary.json`.
  Gitignored (large binaries) — not committed, kept locally.
- `symthaea_music_theory::harmony_verifier` (committed,
  `crates/domains/symthaea-music-theory/src/harmony_verifier.rs`) — the
  narrow verifier remains real, tested, reusable code, not
  campaign-specific throwaway.
- `symthaea-muse/examples/baroque_campaign.rs` (committed) — the harness
  itself is reusable; a future formal validation run doesn't need new
  tooling, just the loudness-normalization/blinding/sealing step this
  closure skips.
- `PREDECLARED_GATE.md` — the six criteria remain valid and precisely
  specified against the existing `harmony_report.json` fields; nothing
  about them is invalidated by stopping here.

## What comes next instead

Per the user's own assessment: the current output is real "Baroque
coloration" (arpeggiated continuo-like accompaniment, functional tonal
movement, clearer dominant-tonic direction, some independent upper-voice
activity) but not yet mature Baroque composition. The gap isn't another
progression generator — it's that melody/counter-voice realization
doesn't yet respond intelligently to the chord sequence it's given. See
the sibling roadmap note for the chord-aware realization work this
motivates.
