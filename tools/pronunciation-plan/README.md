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


## Deterministic score-alignment planner

The contract now includes `VocalScoreAlignment::from_duration_weights` and `from_beat_domain_score`. These construct a validated mapping from:
- a fully selected pronunciation plan;
- ordered note/rest events (or a beat-domain score with a tempo map);
- explicit rest-to-syllable boundary receipts;
- one positive duration weight per selected syllable;
- phrase ranges and breath metadata.

The planner allocates each uninterrupted sung-note run among its explicitly assigned syllables in proportion to those caller-supplied weights. A syllable may split a long note with other syllables, or cross consecutive note events and become an explicitly designated final-vowel melisma. Rest events terminate runs and require a boundary receipt. The returned alignment is passed through the existing full-coverage validator before it can be consumed.

The planner intentionally does not infer rest boundaries from audio, invent phrase boundaries, or claim that simple weights are learned natural-duration estimates. The caller owns those priors and boundaries; they should eventually be emitted with source/provenance in the run receipt. This is a deterministic proposal mechanism, not a replacement for musical interpretation or human review.

Synthetic tests cover shared-note subdivisions, multi-note melisma, explicit rests, unassigned note runs, missing rest receipts, invalid weights and beat-domain conversion across a tempo change.

## Versioned language-profile contract

The library now defines a separate `LanguageProfileManifest` and a `ProfiledPronunciationPlan` wrapper. This binds an exact pronunciation plan to a specific profile ID, profile digest, BCP-47 tag, dialect, orthography, phone-inventory version, source revision, per-resource license/attribution receipts, and explicit task qualifications.

The phone inventory attaches typed feature labels to IPA units (click class/place, aspiration, ejective/implosive, voicing, syllabicity, vowel length/quality). This is deliberately separate from spelling: a multi-letter click grapheme may normalize to one click phone; the phone inventory—not the count of letters—defines the linguistic unit.

The profile lists capability status independently for G2P, syllabification, morphology, tone prediction, speech synthesis and singing synthesis. A task marked `research_only`, `not_evaluated` or `unsupported` cannot be used as a qualified renderer capability. `validate_for_task` permits rendering qualification only when the exact profile marks the requested task `qualified` with both an evidence reference and digest. A language's G2P status does not imply singing support.

Resource receipts include the exact source repository/revision/path, source blob digest, license-document path/digest, license expression, attribution and rights-review status. Metadata-audited is not equivalent to legal approval. Excluded resources cannot appear in an active profile. The profile digest is externally computed over a canonical payload excluding the digest field; the profile loader/registry must verify the stored manifest bytes before supplying it to this contract.

Synthetic tests exercise an isiXhosa-like click inventory, syllabic consonant feature consistency, unknown phone rejection, mismatched profile binding, and refusal to qualify singing from a research-only profile. These are contract fixtures, not evidence that isiXhosa singing is supported.

## Integration path

1. Adapt the exact Piper Plus and current Muse outputs into candidate records without discarding raw outputs.
2. Add CMUdict/curated variant candidates with context metadata and clear licensing receipts.
3. Annotate a speaker/dialect-balanced English pilot with human-reviewed phones, stress, syllable boundaries and pronunciation variants.
4. Evaluate multilingual G2P systems against mPhonBench-style phonological tasks where language coverage overlaps, then calibrate provider scores against held-out word-level correctness and OOV detection. mPhonBench covers 58 languages and 19 scripts, but its abstract alone does not establish coverage of isiZulu or isiXhosa; verify its released language manifest before selecting those tasks.
5. Use the score-alignment object to bind syllables to sub-intervals within note events. Validate both rapid syllabic subdivisions of one note and a single nucleus sustained across consecutive notes.
6. Only then migrate the production VocalPerformance constructor and DiffSinger worker to consume a frozen pronunciation artifact.

## Claim boundary

This crate encodes and validates an evidence structure. It does not perform G2P, infer context, determine which heteronym is correct, estimate confidence, or establish pronunciation quality.
