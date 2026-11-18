# MUSE M1 Period/Sentence Semantic Audit — Qualification Protocol

**Date:** 2026-11-18  
**Status:** engineering/research specification; no new Rust test, compile, CI, or musical-quality result is claimed  
**Owner path:** #12810  
**Consumer:** #12769, Classical → Minimalism M1 trajectory  
**Existing semantic owners:** #9797 / #9835; source PRs #9832 / #9883

## 1. Executive decision

Do not add another Period/Sentence verifier.

The repository already has the right ownership split:
- #9832 defines the versioned Analyst obligation payload and an independent score-side `ReturnMotif` oracle.
- #9883 carries the Period/Sentence plan's typed return promise and exact theory-side source/target score references into the existing Analyst protocol.
- #12769 retains each segment's score, grammar plan, grammar trace and independently replayable trace report, and currently reports `period_sentence_semantics_verified = false`.

The next increment is a consumer integration: replay the existing ReturnMotif oracle against the exact Classical segment during M1 realization and during receipt verification. The M1 receipt must preserve the distinction between **verified semantic fulfillment**, **insufficient evidence**, and **contradictory evidence**.

A typed plan is not a fulfilled obligation. A trace with valid event references is not necessarily a correct musical return. An unresolved return is not a success.

## 2. Repository state at the time of this note

| Item | Observed public state |
|---|---|
| #12745 | Open draft, mergeable, head `0f401d3c3c706a1875a4466c3f0d129006a8043f`; ProcessAdditive score audit |
| #12769 | Open draft, mergeable, based on #12745; head `89d56008173a519e28470abebe5500c1556124b3`; M1 receipt replay and ProcessAdditive receipt |
| #9832 | Open draft, mergeable, head `74c573f0ab71ff270ebc3c099be15f503753efbb`; typed Analyst obligation verifier |
| #9883 | Open draft, mergeable, stacked on #9832; head `c36ce37461bcf7a3a60535aebb05a8468b6af289`; Period/Sentence producer binding |
| #12810 | Open issue for the M1 semantic consumer integration |

The dependency branches are not yet one integrated, tested head. The integration should follow the repository's merge/stack discipline rather than copying a second implementation of the same verifier.

### Validation boundary

The exact #12769 head queried for this note reported CI and PR Governance workflows as completed with conclusion `skipped`; the combined commit status contained no status entries. Its PR description records 31 authored trajectory tests, not compiled or executed, and 29 authored process-grammar tests from #12745, also not compiled or executed. Cargo/rustc/rustfmt were unavailable in the execution environment used for that report.

These are **not passes**. This note makes no claim that the Rust code compiles, that the authored tests pass, or that M1 is musically successful.

## 3. The semantic seam

### Existing contract path

The producer/verifier path to consume is:

```text
PeriodSentencePlan.typed_return_obligation
    + exact theory-side source_evidence / target evidence
        ↓
GrammarStructuralTrace::ObligationAssertion
        ↓
composer_trace_from_theory
        ↓
ObligationDefinitionV1 + score-scoped source/target event IDs
        ↓
verify_composer_trace
        ↓
ReturnMotif score-side semantics + explicit issue codes/status
```

The typed ReturnMotif definition and its source event IDs must remain stable across the Created → Fulfilled/Unresolved lifecycle. The target event references must be later than, and distinct from, the bound source events. Missing source, target or motif-definition evidence must not be filled from section labels, string obligation IDs, endpoint notes or approximate contour matches.

### M1 integration boundary

The M1 code already constructs a score-scoped Listen projection for each segment and replays `verify_composer_trace`. The dependency change from #9883 is what makes the Period/Sentence ReturnMotif visible to that verifier. Once the dependency is available on the M1 integration base:
1. preserve the typed plan, theory trace and protocol definition as produced;
2. replay the same Analyst verifier on the exact Classical segment score and digest;
3. retain its verifier version, issue codes, evidence status and relevant event references in the Classical segment receipt;
4. derive the outer receipt's Period/Sentence semantic disposition from that replayed report;
5. repeat the identical check when a serialized receipt is independently verified.

Do not promote `period_sentence_semantics_verified` by inspecting a boolean written by the producer. The stored report is evidence to compare; the receipt verifier must recompute it.

### Disposition semantics

Reuse the existing Analyst status and issue vocabulary where possible. The M1 summary may expose a compact derived disposition, but must not invent a fourth meaning or collapse the distinctions:

| Disposition | Meaning | M1 treatment |
|---|---|---|
| `Verified` | The exact typed ReturnMotif declaration has sufficient source, target and motif evidence and passes the existing score-side oracle | May set the semantic-verified summary true |
| `InsufficientEvidence` | Missing/legacy/unresolved/unsupported evidence; no contradiction is established | Never claim semantic verification; expose the limitation or reject promotion to a fully verified M1 receipt |
| `Discrepancy` | Contradictory source/target relation, malformed identity, invalid transformation, stale or mismatched artifact | Typed failure; receipt cannot be accepted |

The current M1 realization path requires a fully verified trace report. If a declared return remains unresolved, fail closed rather than weakening this gate. A future policy that permits a partially qualified M1 artifact must encode that explicitly as a non-verified disposition; it must not return an all-green receipt.

## 4. Paradigm shifts this increment enables

### A. Composition as a set of promises with due evidence

Move beyond a generator producing sections and then narrating what happened. A grammar plan declares a typed promise; a later score realization supplies exact event identities; an independent verifier determines whether the promise was fulfilled. The system remembers what it committed to and can be held accountable to that commitment.

### B. Evidence disposition, not optimistic pass/fail

A binary flag hides why a system did not qualify. Preserve three outcomes—verified, insufficient, discrepancy—plus coverage. A cautious system that abstains on unsupported material should not be scored like a false accept, nor should it be allowed to advertise 100% accuracy on a tiny easy subset.

### C. Identity before similarity

A motif return is a lineage claim, not simply two similar note sequences. The declared source identity and exact score-scoped events determine what is being returned; symbolic transformation checks determine whether the target satisfies the declared relation. The broader source-bound material graph in #11380 can later carry this identity through more complex transformations. Do not use pitch/register/role similarity to invent persistent identity.

### D. Counterfactual qualification

For every claimed control, compare the same pinned engine revision under a frozen intent/seed while changing one declared factor. Check both the intended semantic effect and non-target invariants. A copied parameter or a plausible label is not evidence that the musical result changed as requested.

## 5. Deterministic qualification protocol

### 5.1 Discovery cohort (not the reported test set)

Run the actual Period/Sentence producer over a preregistered discovery matrix. One proposed first matrix is:
- bars: 8 and 16;
- arousal: 0.25, 0.50, 0.75;
- seeds: 0 through 63 inclusive;
- all other intent fields and engine revision frozen.

This gives 384 producer runs. The exact machine-readable case grid is frozen in [`muse-m1-period-sentence-returnmotif-cohort-v1.json`](muse-m1-period-sentence-returnmotif-cohort-v1.json); it also declares a separate 120-case confirmation cohort on seeds 1000–1039. The manifest is marked `preregistered_not_executed`, pins no engine SHA yet, and makes no execution or accuracy claim. Record each result as `Verified`, `InsufficientEvidence`, `Discrepancy`, or generation failure. Retain all failures and the complete denominator. Do not discard seeds that fail to realize a return.

Use discovery outcomes only to select candidate deterministic fixtures and diagnose the generator. Do not optimize the verifier against the eventual confirmation cohort.

### 5.2 Confirmation cohort

After the generator and fixture selection are frozen, use a separate preregistered seed range and an unchanged engine revision. Publish the full manifest, exact intent payloads, score/trace digests, toolchain identity, per-case outcome and all issue codes.

A suggested confirmation size is 120 cases (40 seeds × 3 arousal levels, fixed bars = 8). If compute limits require a smaller cohort, publish the actual denominator and confidence intervals; do not silently substitute a pilot result for confirmation.

### 5.3 Positive fixtures

Freeze at least one positive fixture only after the existing Analyst ReturnMotif oracle accepts the exact target/source relation. A producer test that permits either Fulfilled or Unresolved is useful for lifecycle testing but is not, by itself, a positive semantic fixture.

If no stable positive fixture exists, diagnose and repair the planner/realizer contract. Do not loosen the oracle, broaden the target window until unrelated material matches, or change a typed transformation to match the output after generation.

### 5.4 Adversarial controls

The confirmation fixture set must include deterministic mutations for:
- unrelated but contour-similar target notes;
- wrong source IDs or wrong target IDs;
- duplicate IDs and repeated event references;
- target preceding or overlapping the source when the declared semantics require a later return;
- missing motif definition or empty source evidence;
- source/target events drawn from the wrong role or score segment;
- a target that fails the declared transformation;
- changed typed definition between Created and Fulfilled;
- stale score digest or altered score with unchanged pinned digest;
- mutated stored semantic report followed by recomputing untrusted segment/outer content hashes;
- legacy payload without a typed definition;
- explicitly unresolved producer result.

Expected outcomes must be specific: invalid references/transformations are discrepancies, legacy/unsupported/unresolved cases are insufficient evidence, and only valid known-answer fixtures are verified.

### 5.5 Reporting

Publish at least:
- accepted positive count / positive fixture count;
- false accepts / contradictory controls;
- false rejects / valid fixtures;
- InsufficientEvidence count and reasons;
- generator failure count;
- coverage (verified cases / all attempted cases);
- deterministic replay agreement;
- p50/p95/p99 verifier latency;
- breakdown by bars, arousal, seed cohort and issue code.

Do not report conditional accuracy without coverage. Do not combine this symbolic semantic panel with listener preference or audio quality.

## 6. Industry and research positioning

The field is moving along several separate axes. M1 should not claim to beat all music-generation systems because it verifies one symbolic grammar transition.

### Blind audio preference

As accessed 2026-11-18, Artificial Analysis AA-Music-Instrumental v1.1 lists Suno v6 at 1148±22 (2,208 samples). The vocal board lists ACE Music v1 at 1160±21 (2,119 samples) and Suno v6 at 1144±18 (3,063 samples); their 95% intervals overlap. These are blind same-prompt preference estimates, not measures of symbolic structure, edit preservation or proof of musical intent.

Sources:
- https://artificialanalysis.ai/music/leaderboard/instrumental
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

### Direct symbolic control

A 2026-11-17 symbolic-conditioning paper reports, over 150 generated examples, 82.0% genre, 88.7% tempo and 88.0% density adherence for its Transformer. Its controls are coarse classes; the study does not establish exact event-level constraint verification, negation/conflict resolution or protection of unrelated score events.

Muse's relevant challenge is not merely to exceed those categorical percentages. It is to test exact requested controls against the score, one-factor counterfactuals, protected-event invariance and explicit unsupported/abstention behavior.

Source: https://doi.org/10.1007/s44163-026-02725-8

### Long-range symbolic structure

CAST reports structural error 0.22±0.02 and chord accuracy 0.96±0.01 versus its MuseFormer baseline's 0.58±0.03 and 0.64±0.02, on a five-run GiantMIDI-Piano protocol. This is a useful reproduction target, not a multitrack or orchestration result. Reproduce the paper's data split, representation, run count and metric before making a direct claim.

Source: https://doi.org/10.1038/s41598-026-46750-0

### Editing and preservation

SMEBench provides paired synthetic music edits and a real-audio complement, with separate source-preservation, target-agreement and text-compatibility dimensions. Its CC BY-NC-SA 4.0 licence and upstream dataset/component licences must be respected; do not treat the benchmark corpus as unrestricted commercial training material.

Sources:
- https://arxiv.org/abs/2610.36715
- https://creativecommons.org/licenses/by-nc-sa/4.0/

MMAE for ICASSP 2027 exposes 2,000 public development examples and 17,741 atomic rubrics, with instruction-following, consistency and exact-match dimensions. Its organizers report existing systems below 5% exact match. Registration closed on 2026-10-08; the public material is an internal diagnostic opportunity, not a basis for claiming eligibility or an official leaderboard result.

Source: https://audio-editing-challenge.github.io/

### Commercial production UX and direct symbolic-input baselines

Basic editing controls are no longer differentiators:

- **Suno MIDI-in (2026-11-04)** accepts MIDI files as a starting point, letting users request exact following or motif development. Its Studio workflow also supports MIDI note/timing edits. This makes symbolic-input generation an explicit current product baseline, not a hypothetical future feature. Sources: https://help.suno.com/en/articles/14594177 and https://suno.com/blog/MIDI-in-create
- **Eleven Music v2.5** supports structured composition plans with up to 30 ordered chunks, per-chunk duration/style/text, and audio-reference chunks that can be inserted unchanged. Its inpainting API can retain selected source slices while regenerating other chunks. Sources: https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans and https://elevenlabs.io/docs/eleven-api/guides/how-to/music/inpainting
- **Suno Studio 2.0** also exposes MIDI editing, automation, synths, effects and chat-assisted production. Source: https://help.suno.com/en/articles/13670529

Muse must not claim novelty merely for accepting MIDI, rendering a score, chunking a composition, or editing a selected region. Its stronger testable hypothesis is a source-authored and replayable edit contract: persistent identity where supported, exact allowed/protected scope, independent target and preservation checks, explicit abstention when unsupported, and deterministic undo/redo receipts.

### Comparator protocol caveat

YuE2's public WildSongBench report includes a best-of-eight result, while its benchmark notes explain that candidate counts and selection protocols differ among systems. Report candidate budget, reranking metric, number of ASR repeats, and whether the displayed result is single-sample or selected-from-N. Do not treat the highest selected-candidate score as a matched one-sample head-to-head against every commercial baseline.

Sources: https://map-yue2.github.io/ and https://github.com/dsmpd/YuE2/blob/main/docs/benchmarks.md

### Primary-source benchmark update — 18 November 2026

The current research comparison is stronger when the evaluation objects are kept distinct:

- **YuE2 (27 September 2026)** is a direct symbolic-planning and editing comparator. Its paper reports 49.3% expert preference for the same checkpoint with symbolic planning versus 34.6% without planning, 6.73 SongBench Global Average (6.96 best-of-eight), and expert preference for best-of-eight over Suno v4.5 with near-balanced preferences against Suno v5. The model exposes a readable melody/harmony plan, uses it for audio realization, and describes agentic score revisions that largely preserve unedited content. This means Muse must not claim that readable score plans or conversational score editing are unique. Our testable hypotheses are stronger source-authored PartId/MaterialId lineage, typed localized operation contracts, zero unauthorized score mutations, reversible candidate receipts and explicit abstention.
  Source: https://arxiv.org/abs/2609.33757

- **SMEBench** contains 11,160 paired source/reference-edited examples in total, with a balanced 3,600-example test split (300 per operation across twelve edit types) and a separate 416-pair real-audio complement. Its 17-person study yielded 236 ratings: mean target quality 4.23/5, source alignment 4.65/5 and instruction alignment 3.99/5. Three operations (Legato, Remove_Instrument, Swap_Drums) did not reliably exceed the neutral midpoint for instruction following. This is a useful paired-target benchmark, not an infallible reference oracle. Its CC BY-NC-SA 4.0 licence and all upstream component licences remain applicable; repository issues #12003/#12029 own rights-bound admission. Do not use downloadable assets for commercial work without purpose-specific reviewed permission.
  Source: https://arxiv.org/abs/2610.36715

- **MMAE / ICASSP 2027** released the challenge test sets and SDK on 10 November. The official final submission deadline is **25 November 2026**. Registration closed on 8 October; the Agent Track also requires model versions and weights to have been public before 1 November. Existing registered teams may still submit; new entrants must not assume eligibility. The organizer reports existing systems below 5% Exact Match Rate. The public development materials are useful for understanding task rubrics, but the dataset card's lack of a visible licence field is not a rights grant; #12003/#12029 continue to treat the assets as blocked until authoritative component and purpose-specific terms are recorded.
  Source: https://audio-editing-challenge.github.io/

- **Artificial Analysis** uses separate instrumental and vocal boards, same-prompt/same-modality blind pairwise comparisons, Bradley–Terry estimation, a minimum ten-second listen per track, and loudness/format normalization. This is a credible preference baseline, but it does not assess symbolic edit correctness or protected-event preservation. Keep it as a separate human-preference scorecard.
  Source: https://artificialanalysis.ai/methodology/music

This refresh changes our competitive framing: Muse must be compared against symbolic planning plus audio realization systems, not only text-to-audio systems. A score-side proof is a complementary capability, not a substitute for listening studies; equally, preference Elo cannot authorize a symbolic invariant violation.

## 7. Benchmark targets: proposed, not achieved

| Dimension | Qualification target |
|---|---|
| Frozen semantic fixtures | 100% valid positive acceptance and 100% contradictory control non-pass |
| False semantic fulfillment | Zero on the frozen adversarial panel |
| Unknown / legacy / unresolved cases | 100% remain explicitly InsufficientEvidence or equivalent non-pass |
| Receipt replay | 100% recomputed semantic report agreement on valid frozen receipts |
| Mutation detection | 100% rejection of frozen score/trace/report mutations |
| Held-out coverage | Publish full attempted denominator and coverage; do not hide unsupported cases |
| Causal controls | Demonstrate intended score-level change while protected dimensions remain within preregistered tolerances |
| Latency | Report p50/p95/p99 independently of correctness |
| Human recognition | Separate blinded musician/listener experiment; do not infer recognition from symbolic motif match |
| Audio production | Separate renderer/audio qualification and blind preference; no promotion from score-side proof |

These are release targets. They are not measured achievements and should be revised only through a documented protocol amendment, not after inspecting failed examples.

## 8. Implementation order

1. Land/rebase the existing dependency path #9832 → #9883 and #12745 → #12769 without merging overlapping incompatible bases.
2. Integrate the existing typed Period/Sentence semantic report into Classical M1 realization and receipt replay.
3. Discover and freeze positive and negative fixtures using the cohort protocol above.
4. Add a precise receipt disposition; derive it from recomputed Analyst evidence rather than a stored assertion.
5. Run exact-head Rust formatting, compilation, focused tests, broader workspace tests and CI. Keep each result tied to the exact head SHA.
6. Only after symbolic qualification, connect material lineage (#11380), PartId propagation (#11351) and reversible edits (#11332) to a multi-part source-preservation benchmark.
7. Keep blind preference (#9535), score-first capability (#9581), style-response causality (#7422), and human/cultural review as distinct evaluation tracks.

## 9. Stop conditions

Stop and report non-pass if:
- the Period/Sentence plan has no typed return definition but the receipt claims semantic verification;
- the producer emits a fulfilled transition without exact source and target identities;
- the target does not satisfy the declared transformation;
- receipt replay differs from the stored report;
- the fixture only passes after relaxing the verifier or excluding unresolved seeds;
- CI is skipped, missing or attached to a different commit SHA.

The objective is not to maximize the number of green receipts. It is to make each green receipt mean exactly what the independent evidence establishes.
