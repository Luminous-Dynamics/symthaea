# MUSE-MIR-020 — Rust-native Humdrum **kern parser (V1)

**Issue:** #12063  
**Parent:** MUSE-MIR-019 source-native score-event evaluation  
**Implementation:** `crates/domains/symthaea-muse/src/humdrum_kern.rs`  
**Workflow:** `.github/workflows/muse-humdrum-kern.yml`  
**Date:** 14 November 2026  
**Status:** conservative parser and tests authored; not compiled or executed in this authoring environment

## Why **kern matters

The MIREX 2026 Audio-to-Score task requires valid Humdrum `**kern` output. Its principal ranking criteria are WER and MV2H, with CER and LER as diagnostic measures. The format carries the symbolic score syntax needed to evaluate pitches, note values, barlines and voices; an intermediate MIDI representation would discard written details that matter to score comparison.

Humdrum **kern is not an engraving or page-layout format. It encodes semantic/syntactic notation such as pitch, duration, accidentals, articulations, ornaments, ties, slurs, phrasing, barlines, stem direction and beaming. The Humdrum file structure organizes those records into tab-separated spines.

References:
- [Humdrum **kern representation](https://www.humdrum.org/rep/kern.html)
- [Humdrum representation specification](https://github.com/humdrum-tools/humdrum-documentation/blob/master/rep/kern.md)
- [Humdrum syntax and spine organization](https://www.humdrum.org/Humdrum/commands/humdrum.html)
- [MIREX 2026 Audio-to-Score task and metrics](https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription)

## V1 scope

Adds a source-native Rust parser for UTF-8 Humdrum files containing one or more `**kern` spines with stable topology.

### Preserved source structure

- Exact original source SHA-256 and caller-supplied score-clock identity.
- Ordered source records: blank lines, global/local comments, exclusive/tandem interpretations, data and termination.
- Source line number, spine index, source token order and raw token strings.
- Raw tandem interpretations, categorized conservatively as clef, key signature, meter, tempo, instrument, transposition or other.
- Per-spine notes, rests, multiple-stop members, explicit pitch spelling, octave, explicit accidental string/value and opaque suffix signifiers.
- Per-spine barline records, source measure labels, exact score positions and measure-map status.
- Independent rational quarter-beat timelines for each spine.

### Exact rhythmic normalization

The V1 reciprocal-duration subset maps to quarter-note beats without floating point:

| **kern reciprocal** | Duration |
|---|---:|
| `4` | 1 quarter beat |
| `8` | 1/2 quarter beat |
| `2` | 2 quarter beats |
| `1` | 4 quarter beats |
| `0` | 8 quarter beats (breve) |
| `00` | 16 quarter beats (longa) |
| `000` | 32 quarter beats (maxima) |

Augmentation dots are reduced as exact rational multipliers. For example, `8..` becomes (7/8) of a quarter beat. The parser does not substitute a default duration when the source reciprocal is missing or unsupported.

### Pitch normalization

Humdrum's octave convention is retained:
- lowercase `c` is middle C (C4);
- `cc` is C5;
- uppercase `C` is C3;
- `CC` is C2;
- the octave changes between B and C.

V1 records only explicit accidental marks (`#`, `-`, `n`) and the raw spelling. It does not apply hidden key-signature or accidental-carry state to infer sounding pitch. That semantic state machine needs its own specification and fixture corpus.

### Multiple stops and spine timelines

- Whitespace-delimited multiple stops within one spine token are parsed as concurrent note/rest members.
- All members in a multiple stop must have the same exact duration. Unequal durations are rejected in V1 rather than silently sequenced.
- A null token `.` produces no event and does not advance the spine cursor.
- Each spine has its own score cursor. One spine's rhythmic duration does not advance a sibling spine.
- Explicit barlines are retained as score-position events. A per-spine measure map is Supported only when explicit barlines bound complete positive-duration measures and the final content does not extend beyond the last barline.
- Missing or incomplete barline structure yields Unknown; the parser never infers bars from a meter interpretation or note density.

## Deliberate refusal boundary

V1 refuses:
- non-`**kern` spines;
- inconsistent tab-separated field counts;
- missing or premature exclusive interpretations;
- spine split `*^`, join `*v`, exchange `*x` or add `*+` operations;
- early or missing spine termination;
- data or interpretation records after termination;
- missing/invalid reciprocal durations and unsupported pitch/accidental spellings;
- unequal-duration multiple stops;
- file, line, token, spine and event-budget violations.

The parser does not attempt to flatten changing spine topology. Such operations alter the meaning and identity of voices; V1 returns a typed Unsupported error so a later topology-aware implementation can model them directly.

## Receipt and replay

The parsed score contains:
- source SHA-256;
- score-clock identity;
- canonicalization version;
- all preserved source records;
- per-spine event, barline, interpretation and measure-map results;
- a canonical report commitment.

Replay reparses the exact source bytes with the same score-clock identity and compares the complete report. Tampering with events, barlines, source records, score clock or digest invalidates the supplied report.

The digest binds content and enables deterministic replay. It does not authenticate the author, prove the correctness of an editorial reading, or establish dataset rights.

## Tests authored

Seventeen focused tests cover:
1. single-spine note/rhythm and explicit measure boundaries;
2. pitch octave mapping and explicit accidentals across B/C;
3. reciprocal and dotted durations;
4. rests and multiple-stop event ordering;
5. unequal multiple-stop duration refusal;
6. tie/slur signifier preservation;
7. clef/key/meter/tempo/instrument tandem interpretations;
8. multiple independent spine cursors and measure maps;
9. null tokens and global/local comment preservation;
10. missing barline Unknown state;
11. split/join/exchange/add topology refusal;
12. inconsistent rows, non-`**kern` spines and early termination refusal;
13. report commitment and tamper replay rejection;
14. invalid reciprocal and pitch spelling refusal;
15. incomplete trailing measure not inferred from meter;
16. missing termination and post-termination record refusal;
17. strict tab-separated row widths for interpretations and comments.

## Benchmark and SOTA boundaries

This parser is format infrastructure, not an audio-to-score system and not a transcription result.

The official MIREX metrics have separate roles:
- **CER:** character-level edit distance on the `**kern` serialization;
- **WER:** tab/newline-separated token-level edit distance;
- **LER:** complete-line error rate, sensitive to voice and temporal alignment;
- **MV2H:** an F-score combining multi-pitch detection, voice separation, metrical alignment and note-value detection, with harmonic-analysis details depending on the task edition.

The current task uses WER and MV2H as main ranking criteria and reports CER/LER for detail. V1 does not implement any of those metric engines and must not call its parser output “MIREX-valid” until independent Humdrum syntax validation and official evaluator fixtures are reproduced.

The sensible next qualification sequence is:
1. validate generated **kern files with the official Humdrum syntax tools;
2. reproduce the official MIREX WER/CER/LER/MV2H implementations against public golden fixtures;
3. implement spine split/join/exchange semantics with exact source-to-voice provenance;
4. build a rights-cleared composer-separated **kern/score/audio corpus;
5. compare Blind and Staves-Informed output separately, with held-out composers and correction-time evidence.

MIREX 2026 results illustrate why these tracks must remain separate: Blind systems score about 0.95 MV2H on the familiar Quartets split but about 0.60 on the held-out QuartetsMIREX set; the Staves-Informed track has a different input protocol and cannot be ranked against Blind systems. See the [official MIREX results](https://futuremirex.com/mirex/wiki/2026:Audio-to-Score_Transcription_Results).

## Qualification boundary

- Tests are authored, not executed.
- No local `cargo`, `rustc` or `rustfmt` toolchain is available in this authoring environment.
- The focused GitHub Actions workflow is authored and path-scoped; no exact-head workflow pass is claimed.
- No official Humdrum syntax-tool parity, MIREX evaluator parity, transcription score, parser accuracy or SOTA claim is made.
- V1 is a conservative source adapter; unsupported spine topology remains explicit rather than being flattened.
