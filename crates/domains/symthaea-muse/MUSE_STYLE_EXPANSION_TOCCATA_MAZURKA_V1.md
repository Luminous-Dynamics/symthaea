# Melothaea Style Expansion: Mazurka and Toccata v1

**Status: developing symbolic recipes, not perceptually qualified.** This tranche adds two executable style profiles and two accompaniment mechanisms. It does not claim historical authenticity, listener preference, or state-of-the-art performance.

## Why these two

The existing catalog already includes **Toccata** (ID 15) as a research-only entry with no composer route. The first improvement is to close this gap: route Toccata to an actual symbolic profile and performance palette instead of growing the catalog without capability.

**Mazurka** is added as catalog ID 153 with `ExpertReviewRequired`. The implementation is deliberately a developing Mazurka-inspired 3/4 recipe, not an authenticity claim. The profile has its own motif tiers, tonal progression, accompaniment gait, tempo range and piano-led renderer palette.

## Implemented profiles

### Mazurka (catalog ID 153)

- Triple meter.
- A developing short-short-long-long motif vocabulary across arousal tiers.
- A distinct accompaniment cell of two eighth-note attacks followed by two quarter-note attacks.
- The secondary accent alternates between beats two and three using the absolute bar index, so it does not reset at a phrase boundary.
- A piano-led palette with cello grounding.
- Canonical status: `ExpertReviewRequired`; source is not promoted to Foundation.

Research context: the USC Polish Music Center describes the mazur/mazurka family and its dance context: https://polishmusic.usc.edu/research/dances/mazur/. The Berliner Philharmoniker's music lexicon also describes the dance's triple time and characteristic weight on the second or third beat: https://www.berliner-philharmoniker.de/en/stories/a-short-piano-lexicon/what-is-a-mazurka/. These are design references, not a substitute for specialist review or repertoire study.

### Toccata (existing catalog ID 15)

- The existing catalog entry now routes to a callable `Style::Toccata` preset.
- Fast, continuously moving sixteenth-note chord-tone passagework with quarter-note pulse accents.
- Separate calm, medium and busy motif banks.
- A distinct progression, ternary starting form, and sequence-enabled development settings.
- Harpsichord/organ roles with cello bass in the current renderer palette.

Research context: standard reference descriptions characterize the toccata as a keyboard form that often foregrounds touch, rapid passagework and virtuosity. See https://www.encyclopedia.com/literature-and-arts/performing-arts/music-theory-forms-and-instruments/toccata. This profile is a developing toccata-inspired route, not a claim to reproduce a complete historical toccata-fugue.

## Engineering changes

- Extend `Style` from 35 to 37 entries, maintaining a single canonical `Style::ALL` roster.
- Add `MazurkaGait` and `ToccataMotion` to the reusable accompaniment vocabulary.
- Make the style profiles select their actual rhythmic mechanisms instead of relying on instrument name or tempo alone.
- Add accompaniment tests for the Mazurka onset/duration cell, alternating secondary accent, and Toccata's 16 sixteenth-note attacks per 4/4 bar.
- Add exact catalog route/count tests; keep Mazurka expert-review-gated.
- Add explicit renderer palette cases for both styles.

## Proposed gates before any stronger claim

1. Run formatting and all focused tests on the exact PR head.
2. Run style-profile validation over every arousal tier and fixed seed.
3. Run a broader seed census for within-style variety, pairwise structural distance and control responsiveness.
4. Render fixed, hash-pinned examples and inspect audio integrity separately.
5. Build blinded, level-matched listening comparisons against the nearest existing styles (especially Waltz / Minuet for Mazurka, and Classical / Baroque Suite for Toccata).
6. Require knowledgeable practitioner review before describing Mazurka as tradition-authentic or promoting it beyond the current review status.

A callable style, passing unit tests, or a renderer label is not proof that listeners recognize the intended style. No rendering or listening result is claimed in this document.
