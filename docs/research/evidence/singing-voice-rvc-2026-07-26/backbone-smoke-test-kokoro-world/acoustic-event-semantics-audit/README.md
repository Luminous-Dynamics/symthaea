# Acoustic-event semantics audit (2026-07-28)

Per the reviewer's bounded, two-part gate: resolve whether the CTC
aligner's "mixed result" (from `ctc-forced-aligner/`) was really a
placement error, or a mismatch between what CTC boundaries mean and
what the renderer's synthesis-relevant sub-events (burst, frication
core, voicing onset) actually are. Uses the existing 23 stops
(the reviewer's Part-1 list treats voiced b/d/g as stops by manner, not
just p/t/k -- widened from the earlier 12), 12 fricatives, and the
1 available affricate from the same 6-phrase corpus.

## Part 1: stops -- the burst is NOT inside the CTC span, almost ever

| Metric | Result |
|---|---|
| Burst inside CTC span | **2/23 (8.7%)** |
| closure_onset - ctc_start | mean +7.6ms, **std 64.1ms** |

This is a stronger, more specific, and more sobering finding than the
"CTC's convention might just define the span differently (e.g. closure-
to-voicing) and the burst is still inside it somewhere" hypothesis this
audit set out to test. In nearly every case, the detected burst occurs
**after** the CTC span's own end -- often 30-75ms later (e.g.
`consonant_clusters` /t/: CTC span ends at 403ms, burst detected at
466ms; /p/: CTC ends 1169ms, burst at 1235ms). `closure_onset -
ctc_start`'s huge std (64ms on a 7.6ms mean) also rules out "closure
onset ≈ CTC start" as a clean relationship.

**Practical conclusion**: CTC's stop span should NOT be trusted to
bound burst location at all -- not "inside," not even "near" in a tight
sense. A burst-search window anchored on the CTC span needs to extend
substantially FORWARD past the span's end (this data suggests at least
~80ms), not the "inside or near" framing originally proposed.

**Real limitation found in this pass, disclosed rather than hidden**:
two adjacent stops in `phrase_final_stops` (`t` and `d`) returned
IDENTICAL closure/burst timestamps (1098.8/1100.0ms) -- their search
windows overlapped and both picked up the same single nearby transient.
The current per-token independent-window search isn't safe when two
target phones sit close together; a joint/exclusive-window search would
be needed to fix this properly. Not fixed this pass -- flagged as a
known instrumentation limitation, not silently accepted.

**Also disclosed, not independently verified**: the burst detector
(spectral-flux maximum in the search window) could plausibly fire on
the FOLLOWING VOWEL's onset transition instead of the stop's own release
transient, especially since a vowel's broadband energy onset can produce
a larger flux spike than a brief stop burst. Not ruled out this pass.

## Part 2: fricatives -- the early-realization skew is real (3/4 landmarks agree); but the CTC span barely covers the actual frication core

| Landmark | n | mean offset (ms) | std (ms) |
|---|---|---|---|
| High-band energy (>=3kHz) | 11 | -40.7 | 20.1 |
| Spectral flatness (Wiener entropy) | 9 | -48.2 | 20.7 |
| Zero-crossing rate | 11 | -42.1 | 23.0 |
| Spectral flux peak | 12 | -3.9 | **68.3** |

**Three independent, genuinely different acoustic measures (energy,
flatness, ZCR) converge tightly on the same ~40-48ms early-realization
skew found earlier with high-band energy alone.** This resolves the
reviewer's Part-2 question cleanly: **the early acoustic frication is
real**, not an artifact of any single detector, and not specific to the
high-band-energy measure used in the original native-duration audit.

The 4th landmark (spectral flux peak) does NOT agree -- near-zero mean
but very high variance (std 68.3ms), with individual values scattered
both strongly positive (+50 to +71ms) and strongly negative (-85 to
-98ms) rather than clustered. Plausible explanation (not confirmed):
`consonant_clusters` is full of consonant clusters (`str-`, `spl-`)
placing fricatives directly adjacent to stops -- spectral flux (which
measures the SIZE of frame-to-frame change generically, not fricative-
specific content) may be picking up a neighboring stop's much sharper
transient instead of the fricative's own onset. Flux is not a reliable
fricative-onset landmark in cluster-heavy contexts on this evidence.

**The more decision-relevant number**: stable-frication-core coverage
inside the CTC span -- **mean 0.203, std 0.049, range [0.113, 0.278]**.
Only about a **fifth** of the stable frication core (a run of high-band
energy staying above threshold for >=20ms) falls inside the CTC-proposed
span, and this ~20% figure is itself remarkably CONSISTENT (low std)
across tokens. This is more precise and somewhat more sobering than the
reviewer's optimistic framing ("CTC consistently captures the stable
core but not earliest noise") -- the data shows CTC captures only a
small, if highly consistent, SLICE of the core, not most of it.

## Reframing the extraction strategy (what this changes)

Neither `pred_dur` nor the CTC phone span should be used directly as a
raw-extraction boundary for either stops or fricatives. Both remain
useful as coarse ANCHORS for a search region, but:

- **Stops**: the burst reliably sits AFTER the CTC span, sometimes by
  70+ms -- search forward, not "inside."
- **Fricatives**: the stable core reliably sits mostly BEFORE/overlapping
  the CTC span's start, consistent with the ~40-48ms early-realization
  skew -- search backward from CTC's start, and use the core-detector's
  own (energy/flatness/ZCR-agreeing) boundaries as the actual
  extraction span, not CTC's own start/end.

This directly informs the "synthesis-event record" the reviewer
proposed: `preservation_start`/`preservation_end` should come from the
class-specific acoustic-event detector (already built here), with
`ctc_span`/`native_span` retained only as anchors/confidence context,
never as the extraction boundary itself.

## Not yet done

- Fixing the overlapping-search-window bug for closely-spaced stops.
- Ruling out the flux-detector/adjacent-stop confound for cluster-heavy
  fricatives.
- Extending Part 1/2 to affricates properly (n=1 here, descriptive only)
  and to a genuinely independent voiced-obstruent landmark.
- Building the full synthesis-event record schema and wiring it into a
  renderer.
- The 4-arm synthesis matrix (A/B/C/D) -- correctly still not attempted,
  now for a clearer reason: we know what to extract (core-detector
  spans, not CTC/native spans directly) but haven't built that extractor
  yet.
- The human listening check -- still the standing, most important item.

## Files

- `13_acoustic_event_semantics_audit.py` -- the audit script (closure/
  burst/voicing-onset detection for stops; high-band/flatness/ZCR/flux
  landmarks + stable-core detection for fricatives).
- `acoustic_event_semantics_audit.json` -- raw per-token records (23
  stops, 12 fricatives, 1 affricate).
