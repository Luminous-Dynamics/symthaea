# Independent G2P reference evaluator

A standalone Rust scorer for evaluating a pronunciation frontend against independently reviewed reference variants.

## Input model

Each JSONL reference record represents one lexical item or fixed phrase:

- `case_id`, canonical `language`, `text`, and disambiguating `context`;
- `expected_oov`;
- zero or more accepted pronunciation variants. Each variant contains canonical IPA phone tokens, exclusive syllable-ending phone indices, and one stress value per syllable (0 none, 1 primary, 2 secondary).

Each candidate JSONL record carries the same case ID, candidate/system identity, source identity, canonical phone tokens, syllable boundaries, stress values, OOV decision, and optional probability that the complete pronunciation is exactly correct.

Accepted variants are explicit so contextual heteronyms and dialect variants can be represented without treating one dictionary entry as universal truth. The evaluator selects the candidate's closest accepted variant using deterministic phone-edit, boundary-error, and stress-error tie-breaking.

## Metrics

- Phone error rate: (substitutions + deletions + insertions) / reference phone count.
- Exact pronunciation rate by case.
- Syllable-boundary precision, recall, and F1; final utterance terminators are excluded from internal-boundary scoring.
- Lexical stress accuracy on cases where reference and candidate syllable counts agree.
- OOV precision, recall, F1, and accuracy.
- Optional Brier score for a system-provided exact-pronunciation confidence. It is omitted if no such probability is supplied.

All metrics with no valid denominator are null/undefined, not zero. Unknown phones remain tokens; no fuzzy symbol mapping is applied. Malformed reference or candidate structure, duplicate IDs, missing cases, mixed system identities, and contradictory OOV declarations fail closed.

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

Compare the current Muse frontend, Piper Plus G2P, and any legally approved alternative against the same gold references. Do not use one candidate's output to create its own test labels. Report metrics by language, dialect, phoneme, word-frequency band, morphology, OOV class, and context.
