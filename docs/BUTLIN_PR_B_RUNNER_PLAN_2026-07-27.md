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
now **done**, and steps 3-4 (`RuntimeQualification` + conformance fixtures) are now **done** too —
see `qualification_runtime.rs`, described below. Only step 5 (a real `AE-2` run against
`CognitiveLoopService`) remains.

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
running something. **Implemented** in `crates/domains/symthaea-psych-bench/src/benchmarks/butlin/qualification_runtime.rs`
(always compiled, no `symthaea-backend` feature needed — same rationale as `qualification_design.rs`
itself: this is pure decision logic, not a call into `symthaea::cognitive_loop`):

```rust
// As landed -- one deliberate change from the original sketch: failure_reasons is a
// computed method (Vec<QualificationFailure>), not a stored field, so there's a single
// source of truth instead of two things that could drift apart.
pub struct RuntimeQualification {
    pub static_design_qualifies: bool,
    pub intervention_applied: bool,
    pub intervention_specificity_passed: bool,
    pub positive_control_effect_observed: bool,
    pub sham_behaved_as_expected: bool,
    pub probe_signal_usable: bool,
    pub identity_and_config_match: bool,
}

impl RuntimeQualification {
    pub fn from_static_design(design: &QualificationDesign) -> Self { /* runtime fields start false */ }
    pub fn failure_reasons(&self) -> Vec<QualificationFailure> { /* every failing dimension, in order */ }
    pub fn qualifies_run(&self) -> bool { self.failure_reasons().is_empty() }
}

pub fn resolve_outcome(
    qualification: &RuntimeQualification,
    indicator_effect_observed: bool,
    functional_effect_observed: bool,
) -> EvidenceOutcome { /* !qualifies_run() -> Inconclusive; else NotDemonstrated / Supported(tier) */ }

pub enum QualificationRunError {
    RegistryIdentityMismatch { indicator: &'static str, field: &'static str, declared: String, actual: String },
}
pub fn check_identity_against_registry(
    design: &QualificationDesign, live_target_lever: &str, live_functional_benchmark: &str,
) -> Result<(), QualificationRunError> { /* hard error, never silently downgraded to Inconclusive */ }
```

A run that fails any one of these stays `Inconclusive` (via `resolve_outcome`), even when the
row's static design is otherwise sound (`GWT-4`'s design is fine, but if the pin turns out
unachievable at runtime, that specific run is still `Inconclusive` — `static_design_qualifies() ==
true` doesn't get inherited for free; `from_static_design()` deliberately defaults every
runtime-only field to `false`). `intervention_specificity_passed` matters on its own, separate from
`intervention_applied`: successfully changing `actual_effective_lr` isn't enough if the same hook
also perturbs several unrelated state variables — that would be a real intervention with a fake
specificity claim. A registry/identity mismatch is deliberately modeled as a hard `Result::Err`
(`QualificationRunError`), not folded into `qualifies_run()`'s boolean space, mirroring `report.rs`'s
existing `EvidenceMergeError::ClassificationMismatch` — a stale/self-contradictory row must be
rejected outright, not silently graded `Inconclusive` alongside legitimate runtime failures.

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

**All of the above are now implemented and green**, as synthetic fixtures in
`qualification_runtime.rs`'s `#[cfg(test)]` module (12 tests, `cargo test -p symthaea-psych-bench
--lib -- qualification_runtime`): `test_gwt3_static_design_alone_forces_inconclusive` (demonstration
4, needs no runtime check at all — the static failure alone is sufficient),
`test_runtime_control_failure_forces_inconclusive_despite_qualifying_static_design` (demonstration
3), `test_metric_moved_but_intervention_not_applied_is_not_a_qualified_pass` (the "hook never fired"
fixture), `test_qualified_probe_with_no_effect_is_not_demonstrated` (demonstration 2),
`test_qualified_probe_with_effect_resolves_to_supported` (demonstration 1, **fixture only** — proves
the mapping logic, explicitly not a claim about a real `AE-2` measurement),
`test_identity_check_hard_fails_on_target_lever_mismatch` /
`_functional_benchmark_mismatch` (demonstration 6), and
`test_hot3_pp1_qualified_pair_must_not_be_reported_as_independent` (demonstration 5 — reuses
`qualification_design.rs`'s already-verified `fully_independent_of()`/`shares_probe_signal()`
rather than re-deriving the dependency logic).

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
3. ~~Design `RuntimeQualification` for real~~ — **done** (`qualification_runtime.rs`), sized to the
   minimal scope's rows only (`AE-2`, `HOT-3`, `PP-1`, `GWT-3`), not all 12.
4. ~~Implement the runner-conformance fixtures~~ — **done**, all resolve to `Inconclusive`/hard-
   failure exactly as designed (12/12 tests green), entirely without touching a real
   `CognitiveLoopService`.
5. ~~Execute `AE-2` as the first genuine empirical row~~ — **done**
   (`ae2_empirical_runner.rs`), AE-2 only per explicit scope direction. `HOT-3`/`PP-1` (linked) and
   `GWT-3` (deliberately unqualified) remain for a later, separate run.

This sequencing existed so the first real run wouldn't have to simultaneously test the scientific
hypothesis *and* the runner's basic correctness — by the time `AE-2` actually ran, the contract
itself was already proven to hold via the synthetic fixtures (steps 3-4).

## Step 5 result: the first real AE-2 empirical run (2026-07-27)

`ae2_empirical_runner.rs` runs four arms against a real `CognitiveLoopService`. All seven
pre-registered questions resolved cleanly, a real bug in the runner's own specificity check was
caught and fixed on the first live run (not before it), and the result reproduced on a second
independent run. **Full numbers, correction history, positive-control scope caveat, the 6 gating +
17 non-gating diagnostic fields, and known limitations are frozen in
`BUTLIN_AE2_FIRST_EMPIRICAL_RESULT_2026-07-27.md`** — not duplicated here to avoid two documents
drifting apart.

**Precise headline** (narrower than the bare `EvidenceOutcome` label): the embodied-cognition
ablation causally eliminated the internal AE-2 probe signal (`embodied_agency`) while the sham and
measured unrelated state remained stable; no degradation was detected on the current downstream
proxy benchmark (a ceiling effect), so functional embodied-agency consequences — and a fortiori any
claim about consciousness — remain unestablished. `Ae2EmpiricalRun::claim_scope_note()` encodes
this distinction in code rather than leaving it as prose that could drift from what the outcome
actually says.

## Recommendation

Per the explicit direction that produced this result: **stop here for review.** The next steps
in order, none started: (1) `AE-2` repeated across fresh seeds — the health-panel tolerance above
is a disclosed, uncalibrated first-pass guess and needs a real baseline-variance study; (2) `AE-2`
under a second stimulus schedule; (3) `HOT-3`/`PP-1` together, explicitly linked as one shared-signal
evidence unit; (4) `GWT-3` as the deliberate real-world fail-closed case; (5) only then the broader
indicator-repair campaign (`BUTLIN_INDICATOR_REPAIR_CAMPAIGN_2026-07-27.md`).

**The honest state right now**: twelve direct designs exist. Five positive controls are
formula-verified. Four complete probe designs (`GWT-4`, `HOT-3`, `PP-1`, `AE-2`) are statically
interpretation-eligible, and two of those four (`HOT-3`/`PP-1`) share a raw probe signal and so
constitute fewer than four independent evidence units. The runtime-qualification contract
(`RuntimeQualification`, `resolve_outcome`, `check_identity_against_registry`) is implemented and
proven correct against synthetic fixtures. `AE-2` is now the first row with a genuine, single-seed,
reproduced empirical result — causal support scoped to its internal probe signal only, with
functional (downstream-benchmark) support explicitly not established (ceiling effect) and no claim
made about the broader theoretical capacity or consciousness. See
`BUTLIN_AE2_FIRST_EMPIRICAL_RESULT_2026-07-27.md` for the full frozen record.
