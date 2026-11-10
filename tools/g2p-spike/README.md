# G2P comparison spike

This standalone Rust crate probes `piper-plus-g2p = 0.5.3` against a fixed English challenge set. It enables the crate's `bundled-dicts` feature and uses `EnglishPhonemizer::new_bundled()`, so a clean CI runner does not depend on an ambient `cmudict_data.json`. It is deliberately outside the production workspace and has no effect on the active Muse/DiffSinger frontend.

## Run

```bash
cargo run --manifest-path tools/g2p-spike/Cargo.toml
cargo test --manifest-path tools/g2p-spike/Cargo.toml
```

The spike enables the crate's `bundled-dicts` feature and uses `EnglishPhonemizer::new_bundled()`, so CI does not depend on a machine-local `cmudict_data.json` path. This embeds the crate's bundled dictionary data in the research binary; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for CMUdict attribution.

The output is a tab-separated diagnostic table containing case ID, input text, Piper Plus raw IPA tokens, current Muse IPA tokens, Piper Plus prosody debug output, and a structured `normalized_word_hypotheses_json` review view. The JSON preserves the full raw token and A1/A2/A3 arrays, per-character prosody, stress marker offsets, punctuation tokens with word/character offsets, and explicit empty word slots. Punctuation and stress markers are not folded into the IPA string. Both frontends receive the same fixture text. Muse is given one synthetic note per syllable so the existing singing bridge exposes its current per-syllable phone sequence. This makes the baseline comparison executable, while the human-reviewed reference remains a separate source of truth.

## Reference scorer

The research-only scorer compares raw candidate outputs with an independently completed word-level reference CSV:

```bash
python3 -m unittest discover -s tools/g2p-spike -p 'test_*.py' -v
python3 tools/g2p-spike/score_reference.py \
  --candidates piper-plus-g2p-spike.tsv \
  --references tools/g2p-spike/references.csv \
  --output g2p-reference-report.json
```

Copy `references.template.csv` to a separate working copy and have qualified reviewers fill in `accepted_ipa_variants`, `reference_status`, and `reviewer_evidence`. The template contains one positional row for every whitespace-delimited word in each challenge phrase, including unreviewed words in the heteronym contexts. Keep those rows as `pending` until they are independently adjudicated. Pending rows are excluded from metrics but retained as positional placeholders so word indices do not shift. The scorer requires the candidate output to contain exactly the same number of word slots as the full template; an omitted phone sequence must remain an explicit empty slot. Do not fill reference phones from Piper Plus or Muse output. Mark a word `reviewed` only after independent adjudication; mark an OOV only when reviewers have confirmed the reference status and recorded an evidence reference.

The scorer reads the structured `normalized_word_hypotheses_json` column, verifies that its raw token stream and punctuation offsets reconstruct the TSV source, and then reports a Unicode-codepoint edit rate after token-boundary flattening, exact normalized IPA-sequence accuracy, heteronym exact sequence accuracy, missing/empty candidate words, and the rate of non-empty IPA guesses on reviewer-confirmed OOV words. Length marks remain individual Unicode code points; no phone-like merging is performed. The latter is not an OOV false-positive rate from an OOV classifier: neither current frontend exposes an explicit calibrated OOV decision. This is a deliberately simple, transparent first metric. It is **not** a full phonological edit distance: stress accuracy, syllable-boundary F1, dialect variance, phone-feature distances, and contextual variant selection need separate annotation and scoring. The report records the normalization method so results are not mistaken for a standard benchmark PER. This is a smoke-test tool until the reference CSV has been completed by qualified reviewers; the committed template is intentionally pending and cannot produce a linguistic-quality result.

## User-generated text track

The fixture now also includes a separate UGT challenge set inspired by UGTPHON (Findings of EMNLP 2026): abbreviations (`ppl`, `imo`), initialisms (`brb`, `idk`), noncanonical spellings (`u r`, `wanna`), elongation (`soooo`, `reallly`), slang (`tho`, `cuz`, `hafta`), and digit/letter mixtures (`gr8 2 c u`).

This track measures two distinct decisions:
1. **Surface normalization:** infer a plausible canonical lexical form from context, while retaining the exact user-supplied surface string and the alternatives considered.
2. **Pronunciation:** produce one or more language/dialect-specific pronunciation candidates for that canonical form.

Do not apply a global rewrite such as `u -> you` or `2 -> to` without context. The intended reading may be ambiguous, dialectal, or artistically deliberate. The target artifact should retain a lattice of candidate normalizations, provenance and uncertainty, with an explicit user/reviewer selection before rendering.

The committed UGT reference rows are all `pending`. They are positional placeholders, not labels. They must be adjudicated independently before any UGT score is reported.

UGTPHON reports that canonical/non-canonical input can cause large G2P degradation (up to 66.8 PER points in its studied settings) and explores retrieval-augmented canonicalization with staged decoding. This is an external research result, not a claim about this Piper spike. Source: https://aclanthology.org/2026.findings-emnlp.1020/

## Why this candidate

Piper Plus G2P 0.5.3 is a Rust crate released 26 October 2026. Its project documentation states MIT licensing, an IPA-first output API, and eight language backends; English is described as a rule-based CMUdict-derived backend. It avoids linking the GPL eSpeak NG engine into the proposed pipeline.

That is a useful implementation candidate, not evidence of pronunciation quality. The English backend is still a finite rule/dictionary system and may mishandle accents, heteronyms, OOV names, contractions, and contextual pronunciations. Its output must be measured, never assumed correct.


## Output normalization boundary

Piper Plus 0.5.3's English implementation emits each IPA Unicode character as an individual token. It emits primary/secondary stress markers as separate tokens and a parallel `ProsodyInfo` vector. For English, A2 encodes stress (0 none, 1 secondary, 2 primary); A3 carries a word-level count. The isolated adapter in `src/normalize.rs` reconstructs word-level raw IPA and preserves the exact raw token/prosody streams, per-character A1/A2/A3 evidence, stress-marker offsets, and punctuation offsets. Punctuation inside a contraction is recorded as a character-offset event without splitting the lexical word. Explicit spaces create word slots, including empty intermediate slots; trailing separators do not invent a ghost final slot.

It deliberately does **not**:
- claim that each character token is one phoneme;
- merge affricates, diphthongs or combining diacritics into canonical phone units;
- infer syllable boundaries from stress alone;
- resolve contextual heteronyms such as *read* or *lead*;
- infer OOV status from whether a G2P hypothesis was returned.

Every normalized word is marked `needs_review`, with `syllable_boundaries: null` and `oov_status: not_assessed`. A later language-specific canonicalizer must use a reviewed phone inventory and independently supported syllabification before constructing a renderable `PronunciationPlan`.

This boundary is important: it is better to preserve a raw, inspectable pronunciation hypothesis than to convert a string of characters into apparently authoritative phone and syllable structure.

## Evaluation protocol

For each text case, compare:
1. the current Muse output produced by `symthaea_muse::singing_bridge`;
2. Piper Plus IPA and prosody output;
3. independently reviewed reference phones and syllable boundaries.

Use this helper only for a coarse raw-IPA character-level diagnostic. For canonical phone error rate, use `tools/g2p-eval` with independently annotated canonical phone tokens, syllable boundaries and stress. The human-reviewed corpus should additionally support word-level pronunciation accuracy, syllable-boundary F1, stress accuracy, variant/heteronym selection, OOV detection and coverage, stratified by word frequency, dialect, proper names, contractions and heteronym context. Do not score a library against labels generated by that same library.

## Licensing boundary

The crate itself is documented as MIT. Its English backend is CMUdict-derived; upstream CMUdict explicitly says research and commercial use are unrestricted, while requesting attribution and warning that errors/omissions remain. Still retain upstream attribution and review every dictionary/data asset actually bundled by the selected crate release before any product integration.

The pure-Rust eSpeak NG port is a separate GPL-3.0-or-later project and is not included in this spike. Do not substitute it without a deliberate license review.

## Non-goals

- This spike does not change the production G2P path.
- It does not claim multilingual South African coverage.
- It does not use African Next Voices/Swivuriso for TTS or singing synthesis. Its published terms prohibit that use.
- It does not claim that any phoneme sequence is correct merely because a library returned it.
