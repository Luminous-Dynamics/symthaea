# Synthesis-event record: fixes the two disclosed instrumentation gaps, confirms the core-coverage finding

Builds the `SynthesisEvent` record schema the reviewer specified
(`phone`, `native_span`, `ctc_span`, `ctc_confidence`,
`acoustic_event_start/end`, class-specific sub-events, `preservation_
start/end`) and fixes the two limitations disclosed in
`acoustic-event-semantics-audit/`: overlapping search windows for
closely-spaced phones, and the flux/adjacent-consonant confound.

## Fix: neighbor-clamped search windows

Every phone's acoustic-event search window is now clamped to never
cross into an adjacent phone's own CTC span (`back_limit =
max(prev_phone_ctc_end, ctc_start - 150ms)`, `fwd_limit =
min(next_phone_ctc_start, ctc_end + 150ms)`), instead of a flat
window applied independently per phone.

**Confirmed fixed**: the two adjacent stops in `phrase_final_stops`
(`t`,`d`) that previously shared identical closure/burst timestamps
(1098.8/1100.0ms both) now resolve to genuinely distinct values
(`t`: closure=1082.5, burst=1083.75; `d`: closure=1183.75,
burst=1185.0) -- **0 duplicate burst timestamps** among this phrase's
5 stops, down from 1 duplicate pair.

## Stops: bursts are findable, but ~22% remain genuinely far even with a correct search direction

All 23/23 stops now find a burst within their clamped window (up from
an ambiguous picture before). But **5/23 (~22%) are flagged "burst far
from CTC span even after clamping"** -- these are not a search-window
artifact (the window was legitimately bounded by a real neighbor or a
generous 150ms cap and the burst still landed far outside the CTC
span), so this is a real, standing limitation: for roughly 1 in 5 stops
here, the CTC span is simply not a useful anchor for burst location even
with a correctly-directed, neighbor-respecting search.

## Fricatives: the ~20% core-coverage finding is confirmed real, and gets a real, disclosed refinement

Investigating one specific record (`fricative_heavy`'s "she" /S/) found
a genuine inconsistency worth chasing down before trusting any
aggregate: `noise_onset_ms` (277.5) was LATER than `stable_core_start_ms`
(172.5) -- a logically backwards "event." Root cause, confirmed by
directly inspecting the raw high-band-energy trace: **this fricative is
phrase-initial** (no preceding phone exists to clamp the backward search
against), so its window fell back to the flat 150ms cap -- but the true
frication for "she" actually starts at essentially t=0 (the very
beginning of the utterance) and runs continuously (with one brief dip
around 291-296ms) to ~380ms, a ~380ms span far exceeding the 150ms cap.
Both landmarks were reporting artifacts of the truncated window, not the
real event. A second phrase-initial fricative (`consonant_clusters`'s
"strong" /s/) shows the identical pattern.

**The other 10/12 fricatives (all word/syllable-medial, with a real
preceding phone to clamp against) show none of this** -- their
`noise_onset`/`stable_core_start` agree closely (often identical to the
millisecond) and sit comfortably inside their windows, not at any
boundary value.

Recomputing core-coverage excluding the 2 flagged phrase-initial cases:

| | n | mean core coverage | std |
|---|---|---|---|
| All 12 fricatives | 12 | 0.209 | 0.098 |
| Excluding 2 phrase-boundary-capped tokens | 10 | **0.250** | **0.035** |

Removing the 2 known-flawed edge cases makes the statistic both
slightly higher AND much tighter (std drops from 0.098 to 0.035) --
consistent with those 2 tokens being noisy outliers from a real,
identified limitation, not evidence against the underlying finding.
**The ~20-25% core-coverage result is confirmed real** and holds up
independently of both the window-overlap bug (now fixed) and the
phrase-boundary-capping issue (now disclosed and isolated, not silently
included in a "clean" headline number).

## What this changes, concretely

- Phrase-initial/final phones need a DIFFERENT backward/forward search
  policy than medial phones -- not a flat 150ms cap, but something that
  extends toward the true utterance/silence boundary instead of an
  arbitrary neighbor-shaped limit. Not fixed this pass; every such
  phone is now explicitly flagged rather than silently mismeasured.
- For medial fricatives specifically (10/12 here), the class-specific
  detector's `stable_core_start/end` is now a trustworthy, internally
  consistent extraction span -- confirmed by two independent landmarks
  (noise_onset, stable_core boundary) agreeing closely.
- For stops, ~78% (18/23) have a usable burst location once the search
  is correctly directed forward and neighbor-bounded; the remaining
  ~22% need a different strategy (possibly: treat as evidence the CTC
  span itself is unreliable for that specific token and fall back to a
  wider, class-agnostic search, or flag for exclusion from raw-transient
  preservation).

## Not yet done

- A real backward/forward policy for phrase-initial/final phones
  (currently just flagged, not fixed).
- Understanding the ~22% of stops whose burst remains far even after
  correct-direction, neighbor-bounded search.
- Wiring this record into an actual renderer (the 4-arm synthesis
  matrix still hasn't been attempted).
- Affricates (n=1) and voiced obstruents remain out of scope for this
  pass's sub-event detectors.
- The human listening check -- still the standing, most important item.

## Files

- `14_synthesis_event_record.py` -- the `SynthesisEvent` schema +
  neighbor-clamped class-specific detectors.
- `synthesis_event_records.json` -- one record per non-marker phone,
  all 6 phrases (111 total records).

## Update (2026-07-28): both "not yet done" items closed

Per the plan agreed after this doc's first version: implement the
utterance-boundary-aware search policy for phrase-initial/final phones,
then investigate the ~22% of stops whose burst stayed far from the CTC
span even after neighbor-clamping.

### Phrase-initial/final phones: fixed

Replaced the flat `MAX_BACK_S`/`MAX_FWD_S` cap with the true utterance
boundary (`0.0` / `len(audio)/fs`) when a phone has no real neighbor to
clamp against. Confirmed: "she" /S/'s stable core now extends
`[5.0, 272.5]` ms (267.5ms duration, matching the true continuous
frication found by direct inspection) instead of the previous
window-truncated `[172.5, 272.5]` (100ms).

**This makes the phrase-initial-fricative finding MORE decisive, not
less**: with the correct (long) core now measured, its overlap with the
CTC span is **exactly 0.0** for both affected tokens (the core
`[5, 272.5]`/`[5, 271.25]` ms ends well before the CTC span
`[322, 342.2]`/`[302.4, 322.6]` ms even begins) -- CTC's span isn't
just under-covering these two, it's placed in a completely disjoint
region. `noise_onset` (the crossing-based landmark) still can't
represent an onset that occurs at/before the very start of a search
window that's now itself mid-frication -- a known, now-precisely-
understood limitation of that specific field; `stable_core_start/end`
(what actually feeds `preservation_start/end`) is correct and should be
treated as authoritative over `noise_onset` wherever they disagree.

### The ~22% "unreachable" stops: root-caused as a detector bug, not a real placement issue

Traced `phrase_final_stops`' /k/ directly against its raw RMS/flux
trace. The true burst is unambiguous: RMS jumps from ~0.001 (genuine
silence, the closure) to real energy at ~1434-1440ms, cleanly INSIDE
the CTC span `[1430.3, 1450.5]` ms. But the detector had reported
burst=1291ms -- 139ms away, flagged "far." Root cause: the old detector
picked the single highest-spectral-flux frame **anywhere** in the whole
search window, and a louder, more modulated vowel region earlier in the
window (flux up to 2.45) simply had more frame-to-frame spectral change
than the real but acoustically weaker stop burst (flux only ~0.7-0.8) --
a global-max search has no way to prefer "the change right after a
closure" over "the loudest change anywhere."

**Fix**: `detect_stop_closure_and_burst` -- find sustained near-silence
runs (adaptive threshold, >=15ms) in the window, take the LATEST such
run (closest to where a stop's closure should be), and place the burst
at the first real RMS rise immediately after it ends, instead of an
unconstrained flux-argmax. Falls back to the old flux-max method only
when no qualifying closure exists in-window, explicitly flagged
`"lower trust"` when it does.

**Result**: `/k/` now resolves to closure=1391.25ms, burst=1433.75ms
(matching the manually-traced ~1434-1440ms almost exactly), voicing
onset=1462.5ms -- no warnings. Across all 23 stops, "still far from CTC
span" dropped from **5/23 (~22%) to 2/23 (~9%)** -- back in line with
the original (unclamped) 2/23 burst-inside-span estimate from the first
acoustic-event-semantics pass, now understood to be the genuinely hard
cases rather than an artifact of a naive detector. The 2 remaining: one
(`ɡ` in `long_sustained_vowels`) is only ~12ms off, plausibly within
normal CTC/acoustic disagreement rather than a real failure; the other
(`t` in `semantically_unusual`) genuinely has no detectable closure in
its window and falls back to the lower-trust flux-max path -- disclosed
as a real remaining edge case, not investigated further this pass.

### What this changes

Both fixes point the same direction: **when a detector disagrees
sharply with the CTC-span prior, check whether the detector itself is
wrong before concluding the CTC span or the underlying acoustic
placement is unreliable.** In both cases here, a naive signal-processing
heuristic (flat window cap; unconstrained flux-argmax) was the actual
source of the apparent "22% unreachable" and "0% core overlap" results
for their respective edge cases -- fixing the detector, not the prior,
resolved most of it. The `synthesis_event_records.json` fields
(`preservation_start_ms`/`preservation_end_ms`, `closure_start_ms`/
`burst_peak_ms`/`voicing_onset_ms` for stops, `stable_core_start_ms`/
`stable_core_end_ms` for fricatives) are now the most trustworthy
extraction-boundary source built in this arc.

## Not yet done (updated)

- The `ɡ`/`t` remaining 2 stop edge cases -- not investigated further.
- Wiring this record into an actual renderer (the 4-arm synthesis
  matrix still hasn't been attempted).
- Affricates (n=1) and voiced obstruents remain out of scope for this
  pass's sub-event detectors.
- The human listening check -- still the standing, most important item.
