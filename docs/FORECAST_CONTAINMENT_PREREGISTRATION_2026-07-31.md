# Pre-registration: interval scoring moves from exact-equality to containment

Written **before** the change, per the discipline applied to the ARC 2-AFC retractions: a
scoring-semantics change alters every number the rule has ever produced, so the predicted effect
is recorded first and checked against the observed effect afterwards.

Audit that motivated this: `FORECAST_INTERVAL_OVERLAP_AUDIT_2026-07-31.md`, Q4/Q5.

## The defect

`probability_of` matches outcome regions by **exact equality**. For `Interval` regions that is not
a scoring rule — it is a lookup that almost always misses. Measured, for an outcome that fell
*inside* the forecast interval:

```
actual 4.0 INSIDE forecast [0,10] p=0.9, unsupported_mass 0.1
  Brier = 1.82     (max for two classes is 2.0)
  Log   = 20.72    (exactly -ln(1e-9): the epsilon floor)
  CRPS  = 0.9
```

Brier and log treat a correct interval forecast as maximally wrong; CRPS, which uses the interval
midpoint, treats it as nearly right. Three rules, two incompatible event models.

## The change

Introduce one predicate and route all region matching through it:

```rust
fn region_contains(region: &OutcomeRegion, actual: &OutcomeRegion) -> bool {
    match (region, actual) {
        (Interval(bin), Interval(a)) => bin.contains(a.midpoint()),
        (x, y) => x == y,   // Boolean / Discrete unchanged
    }
}
```

- `probability_of` sums branches whose region **contains** the actual, instead of equalling it.
- `BrierScore`'s class set becomes the **branch regions** (the bins). The realized outcome is
  mapped to whichever bin contains it, rather than being appended as a separate class — appending
  it would double-count the same mass across two classes that are not disjoint.
- Midpoint is the representative point because `Crps` already uses it. Adopting the existing
  convention is what makes the three rules share one event model.

Safety of containment rests on a gate that already exists: `try_new` rejects overlapping
intervals, so at most one bin can contain a given point. Q3 of the audit noted the overlap check
was not load-bearing under exact equality. **This change is what makes it load-bearing.**

## Predicted effects — to be checked after

1. **No current consumer's numbers move.** Two independent reasons:
   - The only `BrierScore` call outside the calibration crate is the test
     `generated_forecast_is_scoreable_by_the_calibration_crate`, which scores a
     `OutcomeRegion::Boolean`, and Boolean matching is unchanged.
   - `interval_atoms_distribution` (the ensemble's only interval producer) emits **point atoms**
     `[t, t]`. A point atom contains midpoint `y` iff `t == y`, which is exactly what equality
     already did. The change is a **no-op on degenerate intervals** by construction.
2. **No persisted forecast is retroactively invalidated.** A search for stored artifacts
   containing `unsupported_mass` found none — nothing has been written to the evidence ledger.
3. **The Q4 case is repaired.** For forecast `[0,10]` p=0.9 with actual 4.0, log score should fall
   from the 20.72 epsilon floor to `-ln(0.9) ≈ 0.105`, and Brier from 1.82 to a small value.
4. **Cross-rule ordering agreement becomes assertable** — the property test that currently must
   fail should pass.

If prediction 1 is wrong — if any existing test's numbers move — that is a finding, not a nuisance,
and it means a consumer was relying on the broken semantics. It gets reported rather than
absorbed.

## What would falsify the design

- If a bin can be found that contains the midpoint of an actual which lies partly outside it, the
  midpoint convention is too lossy for non-degenerate actuals and should be replaced by an
  explicit rejection of non-degenerate actual outcomes.
- If two bins can both contain a point, the non-overlap gate is insufficient and containment must
  not ship.

## Deliberately out of scope

**Half-open intervals `[low, high)`.** The audit recommends them so contiguous partitions
(`[0,5), [5,10)`) become expressible — today `overlaps()` rejects touching intervals and the test
suite works around it with a literal `5.000001`. That is a change to construction validity with a
wider blast radius than scoring, and it is not needed to fix Q4. Recorded as a separate item.

## Tests to be added

1. `value_inside_forecast_interval_is_not_a_total_miss` — the direct Q4 regression guard.
2. `cross_rule_ordering_agrees` — Brier, log and CRPS agree which of two forecasts is better.
   Expected to fail against the pre-change code; that is the paired control.
3. `containment_matches_at_most_one_bin` — the partition property the overlap gate now buys.
4. `point_atoms_score_identically_to_exact_equality` — pins prediction 1 so a future change cannot
   silently break the degenerate case.
