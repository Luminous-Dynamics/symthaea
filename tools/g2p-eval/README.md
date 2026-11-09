# Independent G2P reference evaluator

A standalone Rust scorer for evaluating a pronunciation frontend against independently reviewed reference variants.

## Input model

Each JSONL reference record represents one lexical item or fixed phrase:

- `case_id`, canonical `language`, explicit versioned `profile_id`, `text`, and disambiguating `context`;
- `expected_oov`;
- zero or more accepted pronunciation variants. Each variant contains canonical IPA phone tokens, exclusive syllable-ending phone indices, and one stress value per syllable (0 none, 1 primary, 2 secondary).

Each candidate JSONL record carries the same case ID, candidate/system identity, source identity, explicit language tag and profile ID, canonical phone tokens, optional syllable boundaries and stress values, OOV decision, and optional probability that the complete pronunciation is exactly correct. Candidate and reference language and profile tags must match case-by-case. A profile ID distinguishes varieties that share a broad language tag—for example, an English variety, a dialect, or a normalized spelling/phonology standard. A backend may omit syllable boundaries or stress; those metrics remain null while phone error rate is still calculated. An OOV candidate must return empty phones and null syllable metadata.

Accepted variants are explicit so contextual heteronyms and dialect variants can be represented without treating one dictionary entry as universal truth. The evaluator selects the candidate's closest accepted variant using deterministic phone-edit, boundary-error, and stress-error tie-breaking.

## Metrics

- In-vocabulary phone error rate: (substitutions + deletions + insertions) / in-vocabulary reference phone count. OOV cases are excluded from this denominator and numerator; guessed phones on an OOV case remain visible in the case-level candidate phone count and are evaluated separately by OOV detection.
- Exact pronunciation rate by case.
- Syllable-boundary precision, recall, and F1 only when phone sequences are exactly aligned; indices are not compared across phone insertions/deletions. Final utterance terminators are excluded from internal-boundary scoring.
- Lexical stress accuracy on cases where reference and candidate syllable counts agree.
- OOV precision, recall, F1, and accuracy.
- Optional confidence calibration for a system-provided exact-pronunciation probability: Brier score, ten-bin expected calibration error (ECE), and reliability-bin mean confidence/observed accuracy. These are omitted when the system does not expose confidence. Calibration is computed only over in-vocabulary cases with a defined exact-pronunciation outcome; OOV classification remains a separate metric.

The scorer reports aggregate, by-language and by-profile metrics; the case receipt retains the selected profile ID. All metrics with no valid denominator or no candidate estimate are null/undefined, not zero. Calibration needs enough held-out independent cases to be meaningful; a handful of synthetic observations exercises the contract but is not a calibration result. The exact-pronunciation and phone-error-rate denominators contain only in-vocabulary cases; OOV detection is scored separately. Unknown phones remain tokens; no fuzzy symbol mapping is applied. Malformed reference or candidate structure, duplicate case/variant IDs, missing cases, mixed system identities, standalone stress/length-marker tokens, stress values without boundaries, and contradictory OOV declarations fail closed.

## Run

```bash
cargo run --manifest-path tools/g2p-eval/Cargo.toml -- \
  --references reference.jsonl \
  --predictions piper-plus.jsonl \
  --output report.json
```

The scorer is dependency-isolated from the production workspace. The committed CI fixtures are synthetic contract fixtures, **not human-reviewed linguistic ground truth**.

## Dataset policy

Build a small, licensed, human-reviewed gold set with at least two independent annotators and an adjudication record. Keep training, tuning, and evaluation items disjoint by lemma, speaker, dialect, and song. Store pronunciation source and variant rationale. Context must be supplied for heteronyms. For African languages, use language/community reviewers and separately consented singing data. Do not use African Next Voices/Swivuriso for TTS, cloning, or singing synthesis under its published restrictions.

## Research use

Compare the current Muse frontend, Piper Plus G2P, and any legally approved alternative against the same gold references. Do not use one candidate's output to create its own test labels. Report metrics by language, profile ID, dialect, phoneme, word-frequency band, morphology, OOV class, and context. Use immutable profile IDs that bind the orthography, phoneme inventory, stress/tone conventions, syllabification policy and version.
