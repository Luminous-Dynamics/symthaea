# Butlin PR B — Minimal Qualification Runner: Design Review + Plan

**Status**: proposal, not implemented. Freezes `qualification_design.rs` as the closure of the
static qualification-design phase — no more taxonomy added there. This document is the pre-runner
design review requested before writing any PR B code, plus a scope for the smallest runner that
proves the qualification contract works end to end.

**Terminology update (post-review, now landed in `qualification_design.rs`):** Findings A and B
below were corrected directly in the static module, not left as a plan-only note — the two
conflated questions they identified are now two separate methods:
`PositiveControlPlan::control_design_qualifies()` (control credibility only) and
`QualificationDesign::static_design_qualifies()` (control credibility AND probe validity
together). `ControlReadiness::Verified` was also renamed to `FormulaVerified`, since "Verified"
was found to overclaim what had actually been checked (see Finding B). The sequencing below (fix
static predicate → rename → runtime qualification design → conformance fixtures → prove fail-closed
→ only then run a real row) reflects that ordering; the static-predicate fix and the rename are
now **done**; everything from `RuntimeQualification` onward is still proposal-only.

## Design review of the 5 nominally-qualifying rows

Per the (now-superseded) single combined check, five rows appeared to pass (`GWT-2`, `GWT-4`,
`HOT-3`, `PP-1`, `AE-2`). Applying the six review questions to each surfaced **two findings that
weren't visible from the static types alone** — both are now fixed in code, described here in
their original form since the fix is the direct answer to the finding:

### Finding A (fixed): the combined check didn't separate control credibility from probe validity — GWT-2 shouldn't fully qualify

`GWT-2`'s positive control is genuinely credible (forcing `gwt_coalition_size = 0` does violate
the real `size > 0` predicate) — it correctly passes `control_design_qualifies()`. But `GWT-2`'s
`probe_validity` is `ExecutionProxy` — the same coarse "did the module run" limitation flagged for
`GWT-3`/`RPT-2` — so it correctly *fails* `static_design_qualifies()`, the stricter combined check.
**Under the corrected definition, only 4 of 5 — `GWT-4`, `HOT-3`, `PP-1`, `AE-2` — pass
`static_design_qualifies()`; `GWT-2` now carries an explicit, code-enforced caveat instead of a
plain pass** (`test_gwt2_control_qualifies_but_static_design_does_not`).

### Finding B (fixed): every "Verified" control was formula-verified, not achievability-verified

The readiness value formerly named `Verified` was assigned whenever the *formula* was confirmed to
produce the claimed effect *given* the manipulation (e.g. "if `phi_attention_weight` were pinned to
1.0, the deviation formula would read 0.0"). None of the 5 had their manipulation's *achievability*
independently confirmed — i.e., whether `CognitiveLoopConfig`/`CycleMetadata` actually expose a
real mutation hook to pin `phi_attention_weight`, zero `embodied_agency`, or pin
`actual_effective_lr` from outside the normal cycle. This is the identical class of gap already
disclosed for `RPT-1`/`RPT-2`/`HOT-1`/`AST-1`/`AE-1` (no input-override hook), just one layer lower
(field-level pin vs. input-level stimulus). **Renamed to `FormulaVerified`** so the variant can
never be over-read as "this control has been run" once it ends up in a serialized evidence bundle
— the achievability/specificity gap itself is not fixed by the rename, only made honestly visible,
which is exactly what `RuntimeQualification` below exists to actually close.

### Per-row findings

| Row | (1) Target scoped? | (2) Control moves metric? | (3) Sham comparable? | (4) Construct measure? | (5) Independent? | (6) Inconclusive trigger |
|---|---|---|---|---|---|---|
| **GWT-2** | Questionable — `enable_gwt=false` is the same broad flag GWT-3 uses, not capacity-specific | Formula-verified only | Yes (`disable_online_learning`, different subsystem) | **No — `ExecutionProxy`** (Finding A) | No — shares target w/ GWT-3, benchmark w/ 5 others, sham w/ GWT-3 | Manipulation not achievable, or coalition-size override has no real hook |
| **GWT-4** | Plausible, not exhaustively traced | Formula-verified only (Finding B) | Yes (`disable_embodied_cognition`) | Yes — `DirectMeasure` | No — shares benchmark w/ AE-2; **cross-role w/ AST-1** (AST-1's sham = GWT-4's target) | Pin has no real hook, or AST-1's cross-role sham run contaminates interpretation |
| **HOT-3** | Plausible, not exhaustively traced | Formula-verified only (Finding B) | Yes (`disable_cross_modal_binding`) | Nominally `DirectMeasure`, **but shares the exact probe signal with PP-1** | **No — shares `probe_group`, `positive_control_protocol` with PP-1** | HOT-3 and PP-1's ablated `actual_effective_lr` values turn out statistically indistinguishable across seeds |
| **PP-1** | Plausible, not exhaustively traced | Formula-verified only (Finding B) | Yes (`disable_embodied_cognition`) | Nominally `DirectMeasure`, **same shared-signal caveat as HOT-3** | Same as HOT-3 | Same as HOT-3 |
| **AE-2** | Plausible, not exhaustively traced | Formula-verified only (Finding B) | Yes (`disable_predictive_processing`) | Yes — `DirectMeasure` | No — shares benchmark w/ GWT-4; **cross-role w/ GWT-4 and PP-1** (both reuse `disable_embodied_cognition` as their sham) | Zero-field override has no real hook |

**Net effect of this review**: of the 5, only `GWT-4` and `AE-2` are *reasonably* clean single-row
candidates (still carrying the achievability caveat and mutual cross-role/benchmark dependency on
each other), `HOT-3`/`PP-1` are a real, deliberately-dependent pair (valuable to run *together*,
not as if independent), and `GWT-2` needs its `ExecutionProxy` caveat surfaced wherever it's
reported. This review did not reduce the runnable set below zero, but it did make every row's
actual evidentiary weight more precise than "5 qualify" implied — consistent with what a review
before PR B is supposed to do.

## Static vs. runtime qualification

The remaining part of Finding B (achievability, not just formula correctness) isn't more static
taxonomy (per the "stop adding taxonomy" guidance) — it's a **runtime** concept the static design
can't resolve on its own, since achievability and specificity can only be established by actually
running something. Store the components, not just a final boolean, so a failure is diagnosable
rather than a single opaque `false`:

```rust
// Sketch only -- not implemented. Lives in the future PR B runner crate/module,
// not in qualification_design.rs.
pub struct RuntimeQualification {
    pub static_design_qualifies: bool,       // QualificationDesign::static_design_qualifies()
    pub intervention_applied: bool,          // did the targeted ablation lever actually execute?
    pub intervention_specificity_passed: bool, // did it change ONLY the intended state, not unrelated fields?
    pub positive_control_effect_observed: bool, // did the control's manipulation produce the claimed effect THIS run?
    pub sham_behaved_as_expected: bool,      // did the sham NOT produce the targeted effect, confirming specificity?
    pub probe_signal_usable: bool,           // probe_validity re-checked at runtime, not just assumed from the static design
    pub identity_and_config_match: bool,     // does this run's actual config match what the design declared? (registry/identity check)
    pub failure_reasons: Vec<QualificationFailure>, // every reason qualifies_run() is false, not just the first
}

impl RuntimeQualification {
    pub fn qualifies_run(&self) -> bool {
        self.static_design_qualifies
            && self.intervention_applied
            && self.intervention_specificity_passed
            && self.positive_control_effect_observed
            && self.sham_behaved_as_expected
            && self.probe_signal_usable
            && self.identity_and_config_match
    }
}
```

A run that fails any one of these stays `Inconclusive`, even when the row's static design is
otherwise sound (`GWT-4`'s design is fine, but if the pin turns out unachievable at runtime, that
specific run is still `Inconclusive` — `static_design_qualifies() == true` doesn't get inherited for
free). `intervention_specificity_passed` matters on its own, separate from `intervention_applied`:
successfully changing `actual_effective_lr` isn't enough if the same hook also perturbs several
unrelated state variables — that would be a real intervention with a fake specificity claim.

## Runner conformance fixtures vs. empirical evidence — two distinct layers

The synthetic malformed-manipulation case in the minimal scope below is a **runner-conformance**
check, not scientific evidence about Symthaea, and the two must not be reported in the same table:

- **Runner conformance fixtures**: deterministic synthetic cases proving the software maps
  conditions to outcomes correctly — failed control → `Inconclusive`; execution proxy →
  `Inconclusive`; malformed identity → hard failure; shared-signal rows → linked reporting;
  qualified null → `NotDemonstrated`; qualified directional effect → eligible support. These prove
  the runner *enforces the contract*, nothing about Symthaea itself. Report under something like
  "qualification-runner contract validation," never mixed into the evidence bundle.
- **Empirical runs**: real `AE-2`, `HOT-3`, `PP-1`, and later rows, run against the real
  `CognitiveLoopService`. These produce the actual evidence bundle. The runner must not manufacture
  outcome-category coverage here — empirical results are whatever the system actually produces,
  including an unplanned `Inconclusive` if that's what happens.

One more conformance fixture is worth adding beyond the malformed-manipulation case already
scoped: a control whose formula response looks correct in a fabricated result, but whose
instrumentation reports the mutation hook never actually executed (mutation counter stays zero).
This specifically tests `intervention_specificity_passed`/`intervention_applied` independent of
`positive_control_effect_observed` — guarding against accepting a coincidental metric movement as
proof the intended manipulation occurred. Expected result: hard failure or `Inconclusive`, never a
qualified pass, even though the metric "looks right."

## Minimal PR B scope

Not all 12 rows. The smallest runner that proves the contract works end to end, using exactly:

- **One strong qualifying row**: `AE-2` (cleanest of the 4 `static_design_qualifies()` rows —
  `GWT-4` carries the AST-1 cross-role wrinkle, so `AE-2` is the simpler first case despite its
  own GWT-4/PP-1 cross-role note).
- **One shared-signal pair**: `HOT-3`/`PP-1`, run together, reported as a linked pair via
  `shared_groups()`, never as two independent confirmations.
- **One deliberately unqualified row**: `GWT-3` (`Unverified` positive control + `ExecutionProxy`
  probe — should report `Inconclusive` on both grounds).
- **One positive control, one sham**: reuse `AE-2`'s own (already declared).
- **One malformed/failed manipulation case**: synthetically force `intervention_applied = false`
  for a row whose static design passes (e.g. simulate "the pin didn't actually take effect") and
  confirm the runner still reports `Inconclusive`, not a false positive/negative.
- **One "metric moved, hook didn't fire" case**: a fabricated result where the expected metric
  change is present but instrumentation shows the mutation hook's counter stayed at zero — proves
  the runner doesn't accept a coincidental movement as proof the manipulation occurred.

Both of the last two are runner-conformance fixtures (see above), reported separately from the
real `AE-2`/`HOT-3`/`PP-1`/`GWT-3` empirical runs.

The runner must demonstrate all six of:

1. Qualified support (a real `CausallySupported`/`FunctionallySupported`-style outcome) — from `AE-2`.
2. A real `NotDemonstrated` (qualified probe, genuine null).
3. `Inconclusive` because the control failed at runtime (the synthetic malformed case).
4. `Inconclusive` because the probe is an execution proxy (`GWT-3`).
5. Dependency-aware reporting that does not double-count `HOT-3`/`PP-1` as independent
   corroboration.
6. A hard failure (not a silent `Inconclusive`) on registry/identity mismatch — e.g. the runner is
   handed a row whose declared `target_lever` doesn't match what `ablation_specs()` actually says
   right now, mirroring `report.rs`'s existing `EvidenceMergeError::ClassificationMismatch`
   philosophy at the qualification layer.

## Explicitly not in this PR B milestone

- Running the other 7 rows (`RPT-1`, `RPT-2`, `GWT-2`, `HOT-1`, `HOT-2`, `AST-1`, `AE-1`) — these
  stay `Inconclusive`-by-design (`Unverified`/`DegenerateGuardTest`) until their own achievability
  gaps are separately closed.
- Multi-seed statistical campaigns, dose-response curves, or the full probe-qualification-v1
  vision's later phases.
- Any of the larger landscape-integration ideas (architecture-neutral adapters, cross-architecture
  comparison, welfare-policy layer) — those remain deferred pending real data from this minimal
  runner, per the prior discussion.

- A large future indicator-repair campaign (typed stimulus/override API, per-indicator construct
  fixes for `HOT-2`/`AE-1`/`AST-1`/`RPT-1`, replacing `GWT-3`/`RPT-2`/`GWT-2`'s execution proxies,
  splitting `HOT-3`/`PP-1` into distinct probes, selective rescue tests) — real, well-motivated
  future work, but a separate, much larger undertaking than this runner. See
  `BUTLIN_INDICATOR_REPAIR_CAMPAIGN_2026-07-27.md` for that plan; do not start it before this
  minimal runner exists and has actually run.

## Recommended sequencing

1. ~~Correct the static qualification predicate~~ — **done** (`static_design_qualifies()` split
   from `control_design_qualifies()`).
2. ~~Rename `Verified` to reflect formula-level verification~~ — **done** (`FormulaVerified`).
3. Design `RuntimeQualification` for real (the sketch above), sized to the minimal scope's rows
   only — not all 12.
4. Implement the runner-conformance fixtures (both malformed-manipulation cases) and prove they
   resolve to `Inconclusive`/hard-failure as designed, entirely before touching a real
   `CognitiveLoopService`.
5. Only then execute `AE-2` as the first genuine empirical row, followed by `HOT-3`/`PP-1`
   (linked) and `GWT-3` (deliberately unqualified).

This sequencing exists so the first real run doesn't have to simultaneously test the scientific
hypothesis *and* the runner's basic correctness — by the time `AE-2` actually runs, the contract
itself will already be proven to hold via the synthetic fixtures.

## Recommendation

The next code change in this area should be the minimal PR B runner scoped above, following the
sequencing just given — not a bigger one, and not more static types. The goal of that first runner
is to prove the fail-closed contract actually holds in practice (broken controls, execution
proxies, shared signals, and malformed interventions all resolve to `Inconclusive` or a hard error,
never an attractive false positive), not to collect indicator results.

**The honest state right now**: twelve direct designs exist. Five positive controls are
formula-verified. Four complete probe designs (`GWT-4`, `HOT-3`, `PP-1`, `AE-2`) are statically
interpretation-eligible, and two of those four (`HOT-3`/`PP-1`) share a raw probe signal and so
constitute fewer than four independent evidence units. No control is yet runtime-achievability
verified, because the runner and mutation instrumentation don't exist yet.
