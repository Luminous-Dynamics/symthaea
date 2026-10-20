# Melothaea Performance Dialects v2: Mazurka and Toccata

**Status: implementation hypotheses awaiting audio calibration.** This change gives the two new symbolic routes an explicit interpretation in the performance layer. It is not a claim of historical authenticity, listener preference, or state-of-the-art performance.

## Why this is a separate gate

The symbolic composition layer now declares distinct mechanisms for the two styles:

- **Mazurka** selects `MazurkaGait`, a 3/4 short-short-long-long accompaniment cell with a shifting secondary accent.
- **Toccata** selects `ToccataMotion`, continuous sixteenth-note passagework with quarter-note pulse accents.

Before this change, both patterns fell through to `ClassicalRubato`: the instrument palette and symbolic note events differed, but the performance-dialect layer did not express the distinction. The new mapping closes that cross-layer gap.

## Explicit timing hypotheses

All values are deterministic scale factors over existing expressive timing, note-timing jitter, and the learned expressive blend. They are priors to test, not fitted parameters.

| Scale | ClassicalRubato | MazurkaLilt | ToccataMotor |
|---|---:|---:|---:|
| Phrase-start breath | 1.00 | 0.90 | 0.15 |
| Climax dwell | 1.00 | 0.90 | 0.15 |
| Cadential ritardando | 1.00 | 0.80 | 0.20 |
| Timing-jitter SD | 1.00 | 1.15 | 0.45 |
| Expressive-model blend | 1.00 | 0.90 | 0.35 |

The intended direction is modestly looser timing and mostly retained phrase expression for the Mazurka prototype, versus a steadier motor and reduced phrase stretching for the Toccata prototype. The audio must be heard and measured before either hypothesis is described as musically successful. A knowledgeable practitioner may reject the priors.

## Mechanic-level evidence

The `dialect_derivation_maps_every_style_preset` test is extended from the previous 29-entry hand table to all 37 executable style presets. Six style-completion routes remain on the identity dialect until a style-specific performance hypothesis is independently justified. Two new tests protect the Mazurka/Toccata spec routing and the exact scale table.

## Next promotion gates

1. Run formatting and focused tests against the exact PR head.
2. Render a fixed-seed, matched-intent cohort for Mazurka versus Waltz/Minuet and Toccata versus Classical/Baroque Suite.
3. Publish unmodified renders and the level-matched blinded pack separately; freeze the candidate budget and do not cherry-pick without reporting best-of-N.
4. Measure onset timing, duration, note density, register, dynamic range, peak/loudness and render integrity, using the same renderer/mastering policy in every arm.
5. Have listeners judge style recognition, rhythmic fit, coherence, expressiveness and overall preference; report by case and listener, not only a pooled score.
6. Promote or recalibrate the scale values only on held-out prompts/seeds with uncertainty reported.

No rendering, benchmark execution, listener result or CI pass is claimed in this document.
