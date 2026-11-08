# Evidence-bearing pronunciation plan (research prototype)

This is a standalone, dependency-light Rust contract for the next Muse G2P architecture. It is intentionally not in the production workspace and is not called by current renderers.

## Paradigm

Do not equate "G2P returned a string" with "the word was understood". Preserve multiple pronunciation candidates, their exact source provenance, syllable boundaries and stress, plus an explicit resolution state:

- \`needs_review\`: multiple or single candidates exist, but none has been selected;
- \`selected\`: an explicit candidate ID and a review/user/policy receipt exist;
- \`out_of_vocabulary\`: no candidate was invented;
- \`unsupported\`: the selected language/profile cannot handle the token.

The plan uses UTF-8 byte spans into the exact original text. Every non-whitespace source character must be covered by one and only one lexical-token span; one span cannot merge whitespace-separated words. A selected pronunciation is bound to a candidate with a non-empty evidence reference. Raw provider scores are retained as *raw scores*, never described as calibrated probabilities.

A renderer should receive a frozen, validated pronunciation plan and syllable-to-score alignment. It must not rerun G2P, reselect a heteronym, or silently approximate an unsupported phone during rendering.

## Contract and test coverage

The library checks schema version, language presence, exact source-text spans, full non-whitespace coverage, duplicate candidate IDs, source provenance, finite raw scores, stress ranges, syllabic nuclei, normalized IPA tokens, explicit selection receipts, and OOV/unsupported state constraints.

Synthetic tests cover contextual heteronyms, user/human selection receipts, OOV without guessed phones, missing source coverage, UTF-8 span errors, unknown candidate selection, stress/length markers mixed into phone tokens, consonant-only syllables, non-finite scores, missing provenance, explicit-rest continuity, beat-to-seconds conversion, tempo changes inside notes, and renderer-precision overflow rejection.

## Score alignment contract

The same crate also defines a research-only `VocalScoreAlignment` contract. A syllable binds to one or more ordered time intervals inside score events. It supports both **many syllables subdividing one long note** and **one syllable spanning consecutive notes as melisma**. Multi-note melisma requires a designated vowel phone; rests remain explicit and cannot be bound. Every sung note's complete interval must be partitioned exactly once by syllable intervals—no overlap or uncovered time—and phrase spans partition the syllable sequence with finite breath durations. Score events are monophonic and chronological; unrepresented time gaps must be represented as explicit rests.

The crate now also includes an additive `BeatDomainScore` source representation: score events are expressed in quarter-note beat coordinates, an explicit piecewise-constant tempo map converts beat positions to absolute seconds, and tempo changes inside a note split that note's elapsed-time integration correctly. A materializer converts the beat-domain events into the existing seconds-domain `ScoreEvent` representation at the renderer boundary while preserving the original beat score for receipts. The current contract intentionally does not yet encode time signatures, tuplets as typed events, ties, grace notes or continuous tempo ramps.

This removes the current constructor's "surplus notes become final-syllable melisma" assumption from the target architecture. The current production constructor remains a compatibility adapter until a separately reviewed integration is made.

## Integration path

1. Adapt the exact Piper Plus and current Muse outputs into candidate records without discarding raw outputs.
2. Add CMUdict/curated variant candidates with context metadata and clear licensing receipts.
3. Annotate a speaker/dialect-balanced English pilot with human-reviewed phones, stress, syllable boundaries and pronunciation variants.
4. Evaluate multilingual G2P systems against mPhonBench-style phonological tasks where language coverage overlaps, then calibrate provider scores against held-out word-level correctness and OOV detection. mPhonBench covers 58 languages and 19 scripts, but its abstract alone does not establish coverage of isiZulu or isiXhosa; verify its released language manifest before selecting those tasks.
5. Use the score-alignment object to bind syllables to sub-intervals within note events. Validate both rapid syllabic subdivisions of one note and a single nucleus sustained across consecutive notes.
6. Only then migrate the production VocalPerformance constructor and DiffSinger worker to consume a frozen pronunciation artifact.

## Claim boundary

This crate encodes and validates an evidence structure. It does not perform G2P, infer context, determine which heteronym is correct, estimate confidence, or establish pronunciation quality.
