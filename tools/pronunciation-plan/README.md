# Evidence-bearing pronunciation plan (research prototype)

This is a standalone, dependency-light Rust contract for the next Muse G2P architecture. It is intentionally not in the production workspace and is not called by current renderers.

## Paradigm

Do not equate "G2P returned a string" with "the word was understood". Preserve multiple pronunciation candidates, their exact source provenance, syllable boundaries and stress, plus an explicit resolution state:

- \`needs_review\`: multiple or single candidates exist, but none has been selected;
- \`selected\`: an explicit candidate ID and a review/user/policy receipt exist;
- \`out_of_vocabulary\`: no candidate was invented;
- \`unsupported\`: the selected language/profile cannot handle the token.

The plan uses UTF-8 byte spans into the exact original text. Every non-whitespace source character must be covered by one and only one lexical-token span; one span cannot merge whitespace-separated words. A selected pronunciation is bound to a candidate with a non-empty evidence reference. In schema v2, the receipt also stores a BLAKE3 digest of the exact candidate record and a second BLAKE3 digest binding that candidate to the schema version, language, full source text, lexical byte span, surface token, and selected candidate ID. A change to the phones, stress, syllable segmentation, provider provenance, source text, language, or token span invalidates the unchanged receipt. These are content-integrity bindings, not digital signatures: the external review/user/community evidence record must still authenticate the authority. Raw provider scores are retained as *raw scores*, never described as calibrated probabilities.

A renderer should receive a frozen, validated pronunciation plan and syllable-to-score alignment. It must not rerun G2P, reselect a heteronym, or silently approximate an unsupported phone during rendering.

## Contract and test coverage

The library checks schema version 2, language presence, exact source-text spans, full non-whitespace coverage, duplicate candidate IDs, source provenance, finite raw scores, stress ranges, syllabic nuclei, normalized IPA tokens, explicit selection receipts, candidate/context digest integrity, and OOV/unsupported state constraints.

Synthetic tests cover contextual heteronyms, user/human selection receipts, stale-receipt rejection after candidate-phone mutation, stale-receipt rejection after language-context mutation, OOV without guessed phones, missing source coverage, UTF-8 span errors, unknown candidate selection, stress/length markers mixed into phone tokens, consonant-only syllables, non-finite scores, and missing provenance.

## Score alignment contract

The same crate also defines a research-only `VocalScoreAlignment` contract. A syllable binds to one or more ordered time intervals inside score events. It supports both **many syllables subdividing one long note** and **one syllable spanning consecutive notes as melisma**. Multi-note melisma requires a designated vowel phone; rests remain explicit and cannot be bound. Every sung note's complete interval must be partitioned exactly once by syllable intervals—no overlap or uncovered time—and phrase spans partition the syllable sequence with finite breath durations. Score events are monophonic and chronological; unrepresented time gaps must be represented as explicit rests.

This removes the current constructor's "surplus notes become final-syllable melisma" assumption from the target architecture. The current production constructor remains a compatibility adapter until a separately reviewed integration is made.

## Integration path

1. Adapt the exact Piper Plus and current Muse outputs into candidate records without discarding raw outputs.
2. Add CMUdict/curated variant candidates with context metadata and clear licensing receipts.
3. Annotate a speaker/dialect-balanced English pilot with human-reviewed phones and stress.
4. Calibrate provider scores against held-out word-level correctness and OOV detection.
5. Use the score-alignment object to bind syllables to sub-intervals within note events. Validate both rapid syllabic subdivisions of one note and a single nucleus sustained across consecutive notes.
6. Only then migrate the production VocalPerformance constructor and DiffSinger worker to consume a frozen pronunciation artifact.

## Claim boundary

This crate encodes and validates an evidence structure. It does not perform G2P, infer context, determine which heteronym is correct, estimate confidence, or establish pronunciation quality.
