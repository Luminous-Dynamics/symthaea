# Forecast interval overlap policy — read-only audit

Bounded audit, no code changed. Commissioned to settle whether arbitrary partially overlapping
continuous intervals are rejected or given defined semantics, which a review flagged as
unestablished.

**Headline: partial overlap IS rejected, so the specific dispute is closed. But the audit found a
larger inconsistency underneath it — the validation gate enforces one interval semantics and the
scoring layer implements two others.**

## Q1 — Can distinct interval branches partially overlap today?

**No.** `ForecastDistribution::try_new` rejects them:

```rust
(OutcomeRegion::Interval(x), OutcomeRegion::Interval(y)) => {
    if x.overlaps(y) { return Err(ForecastError::OverlappingIntervals { .. }); }
}
```

Covered by `rejects_overlapping_intervals` (`[0,10]` vs `[5,15]`). Serde routes through
`try_from = "..."`, so deserialization cannot bypass it — covered by
`deserialization_rejects_what_construction_rejects`.

**Undocumented consequence: *touching* intervals are also rejected.** `overlaps()` is

```rust
self.low <= other.high && other.low <= self.high
```

which is correct for **closed** intervals — `[0,5]` and `[5,10]` genuinely share the point `5.0`,
and a value of exactly 5.0 would belong to both. But it means the natural contiguous binning
`[0,5], [5,10], [10,15]` is **inexpressible**. The existing test works around this with a
deliberate epsilon gap:

```rust
(0.5, interval(0.0, 5.0)), (0.5, interval(5.000001, 10.0))   // note the 0.000001
```

That workaround is a smell, not a fix. Half-open intervals `[low, high)` would let contiguous
partitions be stated directly.

## Q2 — What event semantics does scoring assign to overlapping branches?

Moot for overlap, since overlap cannot be constructed. But the underlying question — what does an
`Interval` *mean* to a scorer — has **two different answers in the same crate**:

| scorer | how it reads an `Interval` | implied semantics |
|---|---|---|
| `Brier`, `LogScore` | `probability_of` → `filter(\|b\| &b.outcome == target)` | an opaque **atom/label**, matched by exact equality |
| `Crps` | `interval_midpoint()` → `iv.midpoint()` | a **numeric location** |

`probability_of`'s own doc comment concedes the problem: *"Exact-equality matching — meaningful
for `Boolean`/`Discrete`, **not** for `Interval`."*

## Q3 — Can overlapping probability mass be double-counted?

**Not today — and notably not because of the overlap check.**

Under exact-equality matching a target matches only branches *identical* to it, and identical
branches are already rejected by `DuplicateOutcomeRegion`. So double-counting is prevented by the
**duplicate** check; the **overlap** check is not currently load-bearing for it.

The overlap check is defensive against a future change to containment matching — which is exactly
where it *would* become load-bearing, and is the change Q5 recommends. It is correct to keep it.

## Q4 — Are Brier, log score and CRPS consistent about overlap?

**No.** They are not even consistent about what an interval is. Concretely, take a forecast branch
`[0, 10]` with p = 0.9, and an actual outcome of `[4, 4]` — a value that fell *inside* the
forecast interval:

**Executed, not derived** (temporary test, since removed):

```
actual 4.0 INSIDE forecast [0,10] p=0.9, unsupported_mass 0.1
  Brier = 1.82
  Log   = 20.72
  CRPS  = 0.9
```

| scorer | result | reading |
|---|---|---|
| Brier | **1.82** — the maximum for a two-class problem is 2.0 | near-maximal penalty |
| Log | **20.72** — this is exactly `-ln(1e-9)`, the epsilon floor | **the worst score the rule can emit** |
| CRPS | **0.9** | moderate penalty |

Same forecast, same outcome, opposite verdicts. The log score does not merely penalise a correct
interval forecast — it clamps to its own floor, the value reserved for a forecaster that assigned
essentially zero probability to what happened. A forecaster *well calibrated over intervals* is
scored as maximally wrong by two of the three rules.

The executed numbers are worse than the reasoned estimate that preceded them; the derivation had
predicted a large penalty but not a floor-clamped one.

This is not a rounding disagreement; it is a disagreement about the event space.

## Q5 — Bins, arbitrary events, or empirical atoms?

Three models are currently present at once:

- `try_new`'s non-overlap check enforces **mutually exclusive bins** (a partition).
- `probability_of`'s exact match implements **empirical atoms**.
- `Crps`'s midpoint implements **numeric locations**.

**Recommendation: mutually exclusive bins**, for three reasons.

1. It is the only model consistent with the gate that already exists. The non-overlap check is
   already enforcing partition semantics; the scoring layer just does not honour it.
2. It is the model under which Brier and log score are *proper* for continuous outcomes.
   Containment matching over a partition is an ordinary multi-class proper score. Exact-equality
   matching over intervals is not a scoring rule at all — it is a lookup that almost always misses.
3. It preserves the existing `unsupported_mass` field as the explicit declaration of the
   uncovered region, which is already the right shape for a partition that does not span the
   whole space.

**This is a scoring-semantics change, not a cleanup.** Changing `probability_of` to containment
would alter every Brier and log score this crate has ever produced. It should be pre-registered
and the prior figures explicitly superseded — the same discipline applied to the ARC 2-AFC
retractions, and for the same reason.

## Q6 — Constructor and property tests required by that choice

**Constructor**

- reject overlap — *exists*
- make intervals **half-open `[low, high)`** so contiguous partitions are expressible, and delete
  the `5.000001` workaround from `accepts_adjacent_nonoverlapping_intervals` — *not done*
- keep `unsupported_mass` as the explicit gap declaration — *exists*
- reject non-finite bounds and inversion — *exists*

**Property tests** (none of these exist today)

1. *Containment partition*: for any constructed distribution and any finite `x`, at most one
   branch contains `x`. Under half-open bins this is exactly what non-overlap buys, and it is
   currently untested because nothing in scoring calls a containment predicate.
2. *Mass conservation under containment*: summing `probability_of` over a partition plus
   `unsupported_mass` equals 1 within `MASS_TOLERANCE`.
3. *Cross-rule agreement in ordering*: for two forecasts and one outcome, Brier, log and CRPS
   agree on which forecast is better. **This will fail against today's code** — that is the point,
   and it is the paired control that makes the change to containment matching meaningful.
4. *A value inside a forecast interval is not scored as a total miss* — the direct regression
   guard for the Q4 defect.

## What is NOT claimed here

No production code was changed. Q1–Q3, Q5 and Q6 are a source read of
`symthaea-futures-core/src/validated.rs` and `symthaea-futures-calibration/src/lib.rs`.

**Q4 was executed**, not reasoned. An earlier draft of this document derived the Q4 numbers from
the two code paths and flagged them as unverified; running them made the finding sharper (the log
score clamps to its floor, which the derivation had not predicted). The temporary test was removed
after use. Recording this because a reasoned-but-plausible mechanism is exactly what went wrong in
the ArcChain diagnosis two days ago, and the correction cost one test run.

The narrow question that prompted the audit is settled: **partially overlapping intervals cannot
be constructed.** The reason to keep reading past that is Q4.
