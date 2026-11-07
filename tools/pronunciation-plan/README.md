# Evidence-bearing pronunciation plan (research prototype)

This is a standalone, dependency-light Rust contract for the next Muse G2P architecture. It is intentionally not in the production workspace and is not called by current renderers.

## Paradigm

Do not equate "G2P returned a string" with "the word was understood". Preserve multiple pronunciation candidates, their exact source provenance, syllable boundaries and stress, plus an explicit resolution state:

- \`needs_review\`: multiple or single candidates exist, but none has been selected;
- \`selected\`: an explicit candidate ID and a review/user/policy receipt exist;
- \`out_of_vocabulary\`: no candidate was invented;
- \`unsupported\`: the selected language/profile cannot handle the token.

The plan uses UTF-8 byte spans into the exact original text. Every non-whitespace source character must be covered by one and only one word span. A selected pronunciation is bound to a candidate with a non-empty evidence reference. Raw provider scores are retained as *raw scores*, never described as calibrated probabilities.

A renderer should receive a frozen, validated pronunciation plan and syllable-to-score alignment. It must not rerun G2P, reselect a heteronym, or silently approximate an unsupported phone during rendering.

## Contract and test coverage

The library checks schema version, language presence, exact source-text spans, full non-whitespace coverage, duplicate candidate IDs, source provenance, finite raw scores, stress ranges, syllabic nuclei, normalized IPA tokens, explicit selection receipts, and OOV/unsupported state constraints.

Synthetic tests cover contextual heteronyms, user/human selection receipts, OOV without guessed phones, missing source coverage, UTF-8 span errors, unknown candidate selection, stress/length markers mixed into phone tokens, consonant-only syllables, non-finite scores, and missing provenance.

## Integration path

1. Adapt the exact Piper Plus and current Muse outputs into candidate records without discarding raw outputs.
2. Add CMUdict/curated variant candidates with context metadata and clear licensing receipts.
3. Annotate a speaker/dialect-balanced English pilot with human-reviewed phones and stress.
4. Calibrate provider scores against held-out word-level correctness and OOV detection.
5. Add a score-alignment object that binds selected syllables to note indices; represent melisma as one selected vowel spanning several note events.
6. Only then migrate the production VocalPerformance constructor and DiffSinger worker to consume a frozen pronunciation artifact.

## Claim boundary

This crate encodes and validates an evidence structure. It does not perform G2P, infer context, determine which heteronym is correct, estimate confidence, or establish pronunciation quality.
