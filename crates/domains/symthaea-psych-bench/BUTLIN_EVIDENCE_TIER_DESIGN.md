# Butlin Indicator Evidence-Tier Design

**Status**: implemented (`src/benchmarks/butlin/{indicators,report,ablation,mod}.rs`,
`tests/butlin_regression.rs`, `src/wm/full.rs`). This document records the model as built. Arkh-
node's public review on PR #30 raised the original questions — aggregation, the architectural
ratchet, merge semantics, and probe-quality/refutation gating — and prompted a multi-round
adversarial follow-up audit conducted separately from that public review; the deeper problems
below (circular null-result gating, unstable duplicate-tick identity, untrusted derived flags,
unknown fallback metadata, unvalidated benchmark measurements) were found during that follow-up,
not by arkh-node directly. See "Corrections after review" below.

**Scope**: `src/benchmarks/butlin/` and the Butlin test files.

## Problem this replaces

Two related defects in the original suite, found while responding to issue #7:

1. **`IndicatorStatus::Present` was hardcoded** on every one of the 14 indicators, regardless of
   the computed score. The enum existed but carried no information.
2. **The `0.6 × static + 0.4 × runtime` blend let a fully-dead live signal still read as
   "present".** A live signal reading `0.0` still floored out at `0.6 × 0.85 ≈ 0.51` for most
   indicators — high enough to pass under nearly any plausible threshold.

Underneath both: the suite conflated genuinely different claims under one score —
*architectural plausibility* (a hand-assigned constant), *causal-implementation validity* (an
ablation shows disabling the mechanism moves the signal), and *functional validity* (disabling the
mechanism also degrades a real downstream competency). A single blended score can't represent
"strong evidence for the first claim, none for the others" — it just averages them together.

## The model

### `SupportTier` and `EvidenceOutcome` — two concepts, not one ladder

```rust
pub enum SupportTier {
    ArchitecturalOnly, Observed, CausallySupported, FunctionallySupported,
}
pub enum EvidenceOutcome {
    Supported(SupportTier),
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}
```

`NotDemonstrated` (a qualified probe that didn't move), `Contradicted` (a qualified probe that
moved the wrong way), and `Inconclusive` (the probe wasn't qualified at all — see "Corrections"
below) are **not** "below `ArchitecturalOnly`" on the same ladder as the positive tiers. They're
different findings with different implications, and treating them as ordinal invites comparisons
like `tier >= Observed` that don't mean anything for a negative result.

| Outcome | Meaning | How it's earned |
|---|---|---|
| `Supported(ArchitecturalOnly)` | The mechanism exists and is wired in; no live signal measured, or a single snapshot can't rule out a frozen one. | Default from `evaluate()`. |
| `Supported(Observed)` | A live signal exists whose *own* measurement already embeds a real responsiveness check. | Currently only HOT-4 (its smoothness probe tests dissimilarity growth across several perturbation magnitudes within one call). |
| `Supported(CausallySupported)` | A targeted ablation, from a **qualified probe**, dropped the indicator's own signal. | `annotate_with_ablation_results()`. |
| `Supported(FunctionallySupported)` | As above, and the paired downstream benchmark also degraded. | `annotate_with_ablation_results()`. |
| `NotDemonstrated` | A qualified probe was ablation-tested and didn't move. | `annotate_with_ablation_results()`. |
| `Contradicted` | A qualified probe moved significantly the *wrong* direction. | `annotate_with_ablation_results()`. |
| `Inconclusive` | The probe itself wasn't interpretable (frozen, non-finite, near-zero baseline). | `annotate_with_ablation_results()`, gates all four outcomes above it. |

### Scores reported separately, never blended

- `architectural_score: f64` — the hand-assigned constant. An expert heuristic, not an empirical
  result — always present.
- `live_score: Option<f64>` — the raw, unblended probe value when one exists. `None` at
  `ArchitecturalOnly` with no runtime data.

A single `evaluate()` snapshot can only ever produce `ArchitecturalOnly` (with the HOT-4/GWT-1
exceptions noted above) — one number can't rule out a frozen or fallback signal.
`CausallySupported`/`FunctionallySupported`/`NotDemonstrated`/`Contradicted`/`Inconclusive` only
come from `annotate_with_ablation_results()`, a pure, strict, provenance-checking merge (rejects
unknown/duplicate indicator IDs, never silently upgrades a tier, deterministic and idempotent) —
the only thing with the baseline-vs-ablated comparison `evaluate()` alone lacks.

`GWT-1` (a derived aggregate of the other 13 signals, not an independent probe) carries an
explicit `EvidenceAnnotation::DerivedAggregate` rather than folding "this is derived" into the
tier itself — the tier answers "how strongly supported," annotations answer "what kind of
evidence is this," and conflating the two would lose information.

## Design invariant: no first-party scalar aggregates across outcomes

**`ButlinIndicatorReport` never exposes, and must never be extended to expose, a single number
computed across indicators carrying different `EvidenceOutcome`s.** The old `mean_quality_score`
did exactly that, and it's how a suite with real taxonomy and scoring gaps still reported "14/14,
mean 0.85". The report's only aggregate view is a **vector**: per-tier counts
(`architectural_only_count` .. `inconclusive_count`) and `tier_summary()`'s distribution — never a
weighted or averaged reduction to one scalar.

This is enforced at the level Rust can actually enforce: no first-party method, report, or
serialized field does this. It is **not** claimed to be type-level-impossible — a caller can
always pull `architectural_score`/`live_score` out of `indicators` and compute their own average;
Rust has no way to forbid arbitrary downstream arithmetic on public `f64` fields. The guarantee is
narrower and honest about its scope: nothing in this crate's own API does it.
`test_no_first_party_scalar_aggregates_across_outcomes` enumerates the exact expected top-level
JSON keys of `ButlinIndicatorReport`, so a future scalar field sneaking in would have to change a
named, visible test rather than slip in silently.

## Corrections after review

The first implementation had real gaps, found by independent review rather than internal testing,
across **three rounds**. Arkh-node's public review on PR #30 supplied the original questions
(Round 1's probe-quality/refutation gap, plus the aggregation and architectural-ratchet points
addressed elsewhere in this document); the deeper problems in Rounds 2 and 3 — the first fix
attempt's circularity, unstable duplicate-tick identity, untrusted derived flags, unknown fallback
metadata, unvalidated benchmark measurements — came from a separate multi-round adversarial
follow-up audit, not from arkh-node directly. Full forensic chronology (exact commit-by-commit
history, exact reviewer wording) lives in the PR discussion, not here — this section keeps only
what's durably load-bearing for understanding the design: the invariant, the rejected approach and
why, the final rule, and any remaining limitation.

### Round 1 — the probe-quality gate

**Invariant**: `Contradicted`/`CausallySupported`/`FunctionallySupported` must never be produced
from a probe that wasn't itself interpretable (frozen, non-finite, or no usable dynamic range) —
otherwise a broken measurement crossing a threshold by accident reads as a scientific finding.

**Rejected approach**: treating `baseline == ablated` as proof the probe was frozen
(`DegeneracyReason::Frozen`). This is circular — it infers "the probe can't move" from the very
same null delta the test produced, with no independent evidence. RPT-2/HOT-1's real findings are
byte-identical baseline/ablated arms; mislabeling that `Inconclusive` would hide a genuine null
result behind a "measurement failure" excuse.

**Final rule**: `ProbeQuality` carries two gates of different strictness. `qualifies_as_observed()`
requires *demonstrated responsiveness* (the bar for a static/live `evaluate()` snapshot to earn
`SupportTier::Observed`). `qualifies_for_ablation_interpretation()` requires only that the
measurement itself is trustworthy — finite, real dynamic range, no confirmed fallback — with **no
requirement that the probe moved**; movement is the ablation's actual result, not a precondition
for trusting the measurement that produced it. `annotate_with_ablation_results()` checks the latter
before any outcome; a disqualified probe becomes `Inconclusive` regardless of what the raw data
says. An equal-baseline/ablated pair that passes this gate correctly becomes `NotDemonstrated`, not
`Inconclusive`. `REPORT_SCHEMA_VERSION` bumped 2 → 3 for the new `Inconclusive` variant.

**Remaining limitation**: `DegeneracyReason::Frozen` stays in the enum for its intended use — an
*independent* responsiveness control establishing a probe can't move — but nothing currently
produces it; a disclosed open gap, not a silent removal.

### Round 2 — hardening the evidence boundary against malformed, unknown, and ambiguous data

Three further gaps, found by a second review pass over the round-1 fix itself:

1. **Trusted derived booleans.** `annotate_with_ablation_results()` read
   `indicator_dropped`/`contradicted`/`benchmark_degraded` straight off `AblationResult`, so a
   malformed, stale, or externally constructed evidence bundle could claim `contradicted: true`
   alongside `baseline == ablated` and the merge would report a refutation that never happened.
   **Fixed** by `classify_ablation()` — the single canonical classifier both `ablation.rs`
   (measurement time) and `annotate_with_ablation_results()` (merge time) now call. This round's
   fix recomputed from raw scores and used the recomputed value, treating the stored booleans as
   cached diagnostics only — **superseded by Round 3 below**, which rejects the whole bundle
   outright on any disagreement rather than silently accepting a self-contradictory row.
2. **Unknown metadata encoded as known.** `fallback_used: false` and `sample_count: 1` were both
   defaults standing in for "not tracked at this layer," which then fed directly into
   `qualifies_for_ablation_interpretation()`'s `!fallback_used` check as if confirmed. **Fixed** by
   a tri-state `FallbackStatus { NotUsed, Used, Unknown }` and `sample_count: Option<usize>`.
   `Unknown` deliberately does **not** disqualify ablation interpretation (only `Used` does) —
   `AblationResult` never tracks this, so disqualifying `Unknown` would make every ablation row
   `Inconclusive` forever; instead the merge attaches an explicit `KnownConfound` annotation
   disclosing the gap. The stricter `qualifies_as_observed()` bar does require confirmed
   `NotUsed`. `sample_count` is `None` wherever the producing layer doesn't actually know how many
   underlying measurements a score aggregates (an ablation baseline/ablated pair aggregates 200
   cognitive-loop cycles per arm into one scalar — `Some(1)` would have understated that).
3. **A passing test that only proved production could misattribute boosts.** The demonstration
   that `ItemKey`'s `(arrival_tick, rank)` scheme isn't stable across duplicate-tick mutations
   (Round-1 finding, unchanged) was real but not a regression guard on its own. **Fixed** by making
   `reconcile_boosts_after_state_change` fail closed: whenever either tick snapshot contains a
   duplicate, it clears the whole `boost` map rather than attempting any merge/eviction
   reconciliation that could misattribute rehearsal evidence. See
   `test_duplicate_tick_identity_ambiguity_fails_closed`. The root limitation is unchanged
   (`ItemKey` still isn't a stable identity, and a real fix needs an immutable item ID from
   `ContinuousMind`, out of this crate's scope) — this is a fail-closed mitigation, not a fix at
   the identity layer.

Two related bugs from the same overall arc, already fixed before round 2 and unaffected by it:
`detect_consolidation_merge` had a false-positive risk (stopped at the first mismatch instead of
verifying the whole remainder — `before=[10,20,30]`/`after=[10,40]` isn't a single removal of `20`);
and consolidation-boost semantics were unspecified, now defined as `max(removed_boost,
survivor_boost)`.

### Round 3 — provenance consistency, functional-measurement validity, and doc accuracy

A third pass, over Round 2's own fixes, found:

1. **Round 2's `classify_ablation()` fix silently recomputed and accepted disagreement.** A
   malformed bundle claiming `contradicted: true` alongside `baseline == ablated` would still merge
   successfully (landing on the raw-score-correct `NotDemonstrated`) rather than being flagged as
   malformed. Since the real producer now derives its cached booleans from this exact classifier,
   disagreement in legitimate evidence should be impossible — silently tolerating it conceals
   producer-version drift, corruption, or tampering. **Fixed**: `annotate_with_ablation_results()`
   now compares the stored classification against the recomputed one during validation and returns
   `EvidenceMergeError::ClassificationMismatch { indicator_id, stored, recomputed }`, rejecting the
   whole bundle, before any indicator is mutated. `AblationClassification` promoted from
   `pub(crate)` to `pub` so the public error type can embed it. See
   `test_merge_rejects_inconsistent_{contradicted,indicator_dropped,benchmark_degraded}_flag` and
   `test_merge_accepts_consistent_classification`. (This also caught a real latent bug in this
   file's own `stub_ablation_result()` test helper: its hardcoded `contradicted` branch used `1.2`,
   which doesn't actually exceed `baseline * 1.5 = 1.35` — fixed to `1.4`.)
2. **`FunctionallySupported` didn't validate the downstream-benchmark measurement itself.**
   `ablation_probe_quality()` validates the indicator's own baseline/ablated scores, but
   `benchmark_degraded`'s inputs (`baseline_benchmark_accuracy`/`ablated_benchmark_accuracy`) had no
   equivalent check — an infinite baseline accuracy paired with any finite ablated value trivially
   satisfies `ablated_acc < baseline_acc * 0.7`, manufacturing false functional evidence from a
   broken measurement (and this case isn't caught by fix 1 above either, since `classify_ablation`
   computes `benchmark_degraded: true` for it consistently on both the stored and recomputed side —
   the bundle isn't self-contradictory, the underlying data is just invalid). **Fixed** by
   `benchmark_measurement_is_valid()` — requires both accuracies finite and within `[0.0, 1.0]` —
   gating the `CausallySupported` → `FunctionallySupported` transition specifically; an indicator
   whose own probe passed quality but whose benchmark measurement is invalid caps at
   `CausallySupported` with a `KnownConfound` annotation, rather than either losing the indicator's
   valid causal evidence or granting an unearned functional claim. See
   `test_merge_infinite_benchmark_baseline_caps_at_causally_supported`,
   `test_merge_out_of_unit_range_benchmark_accuracy_caps_at_causally_supported`. **Remaining
   limitation**: this is a narrower boolean gate, not a full parallel `ProbeQuality` for the
   benchmark side — a `downstream_benchmark_quality` field alongside `probe_quality` would be a more
   complete design, deferred to avoid a second schema break in the same round.
3. **`FallbackStatus::Unknown`'s doc comment contradicted its own gating method.** The variant's
   doc said `Unknown` was "treated as disqualifying for ablation interpretation," while
   `qualifies_for_ablation_interpretation()` explicitly lets it through (only `Used` disqualifies).
   **Fixed**: the doc comment now states the real policy — permitted for provisional ablation
   interpretation with a mandatory disclosure annotation, insufficient for the stricter `Observed`
   tier or any future publication-claim gate. Also fixed `ProbeQuality::none_collected()`, which
   claimed `NotUsed` despite collecting nothing to confirm that from (now `Unknown`) — its
   `degeneracy` already disqualified it either way, but the metadata itself should stay truthful.
   See `test_none_collected_fallback_status_is_unknown_not_confirmed`,
   `test_merge_always_discloses_fallback_status_unavailable`.

### The architectural-constant floors aren't a true cross-commit ratchet

`butlin_regression.rs`'s `ARCHITECTURAL_FLOORS` mirror `indicators.rs`'s hand-assigned constants
closely enough that a single commit can lower both together and the gate stays green — the drift
the gate exists to catch is exactly the drift it can't see this way. A real ratchet needs to
compare the proposed change against the base branch or a separately versioned claims manifest,
which a same-commit unit test structurally cannot do. **Not fixed here** — `butlin_regression.rs`
is now documented as a *structural-consistency check* (floors match the current constants,
nothing else), and the asymmetric ratchet rule (lowering an architectural claim is a visible but
allowed weakening; raising one requires an explicit reviewed claims revision) is scoped to the
follow-up claim-gate/CI-lane work in issue #7, alongside the committed evidence-baseline artifact
it depends on the same infrastructure to compare against.

## What's still follow-up, not in this crate yet

- Splitting CI into three lanes — this file's structural-consistency gate (`butlin_regression.rs`),
  an evidence-integrity lane (backend-enabled, compares live effect estimates against a committed
  baseline), and an explicitly opt-in claim/milestone gate carrying the architectural-constant
  ratchet asymmetry described above.
- The committed evidence-baseline artifact the integrity lane would compare against.
- Multi-seed thresholds and targeted-ablation negative controls (does an indicator's own targeted
  mechanism move it, or does *anything* breaking move it?) — `EffectEstimate`/`ProbeQuality` carry
  real magnitude/quality data specifically so this is addable without another schema break.
