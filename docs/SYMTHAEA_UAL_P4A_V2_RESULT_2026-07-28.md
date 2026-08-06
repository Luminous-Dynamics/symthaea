# UAL-P4a-v2 — Compositional Generalization Redesign: Real Result

**Status: frozen result record.** This is the Phase 5 (P4a redesign) work unit
scoped in `/home/tstoltz/.claude/plans/ethereal-wandering-dewdrop.md` ("UAL-P4a
Redesign: Compositional Generalization with Ground-Truth Held-Out Answers"),
run after the HDC Binding Algebra Qualification and Migration Plan (Phases
0-4, see `HDC_BINDING_ALGEBRA_MIGRATION_INVENTORY_2026-07-28.md`) closed.
Recorded verbatim, before any parameter was touched a second time — see "What
was NOT done" below for the discipline this followed.

## Source commit

- `68446de937` — `feat(ual): P4a redesign with ground-truth held-out answers`.

## What changed from the original P4a

The original `p4a_recombination.rs` (kept in-tree as
`benchmarks::ual::p4a_recombination`, historical record, its own
`Inconclusive` verdict unchanged) compared `WZ_readout - YZ_readout` with no
independently-known correct answer per held-out item — a real
construct-validity gap, disclosed and frozen in
`SYMTHAEA_UAL_FROZEN_EVIDENCE_2026-07-27.md`.

`p4a_compositional_generalization.rs` (UAL-P4a-v2) replaces this with a
genuinely relational **matching rule**: two independent 3-valued factors
(shape `s0..s2`, texture `t0..t2`), compound stimulus `bind(shape_i,
texture_j)`, ground truth `reward iff i == j`. 6 of 9 compounds are trained
(`(0,0)`,`(1,1)` matching; `(0,2)`,`(2,0)`,`(1,2)`,`(2,1)` non-matching); 3
are held out (`(2,2)` matching; `(0,1)`,`(1,0)` non-matching) — each with a
deterministic, known-in-advance correct answer, unlike the original design.
Modeled directly on `benchmarks::executive::ravens`'s proven
extract-rule-via-inverse-unbind-then-score-against-ground-truth pattern.

A 5-rung baseline ladder is validated *before* trusting the candidate rung:
marginal/constituent-value baseline, nearest-neighbor similarity baseline,
exact-lookup (memorization) baseline, a shuffled-relation control (same
mechanism, reward labels shuffled — destroys the true `i==j` structure), and
the candidate mechanism itself. Per-baseline expected values are checked
individually (marginal/nearest-neighbor/shuffled ≈ 0.5 chance;
exact-lookup ≈ 2/3 by construction of the 1-matching-vs-2-non-matching
held-out class imbalance — a real bug in an earlier draft applied a blanket
chance-tolerance check to all four and would have incorrectly flagged
exact-lookup's correct 2/3 result as a design failure).

Preregistered bar for `Demonstrated`: candidate accuracy bootstrap CI lower
bound > 65%, separated from every baseline rung, with the baseline ladder
itself confirmed to behave as predicted (an independent positive control —
trained-set matching-vs-nonmatching mean value gap > 0.1 — must also pass,
verifying the mechanism learns *something* before its generalization result
is trusted at all).

## Exact configuration

- `dimension = 512` (via `generate_near_chance_hv`, same helper as P1/P2).
- `n = 60` independent runs per schedule arm (fresh random factor-value HVs
  each run), 3 held-out items scored per run.
- Both `UalSchedule` arms run and combined via `combine_schedule_reports`
  (Blocked: 6 trained-compound trials grouped by compound; Interleaved:
  randomly ordered).
- Benchmark crate: `crates/domains/symthaea-psych-bench`, module
  `src/benchmarks/ual/p4a_compositional_generalization.rs`.

## Real empirical result

```text
UAL-P4a-v2 functional outcome: Inconclusive — ScheduleScoped
System under test: benchmark-local candidate HDC learner (NOT live Symthaea)
Internal association formation: Observed
Behavioral expression: NotObserved
```

**Blocked schedule — resolves cleanly to `NotDemonstrated`:**
- Baseline ladder behaved exactly as predicted (qualification passed).
- Positive control passed: mean matching-vs-nonmatching trained value gap =
  **0.5400** (well above the 0.1 floor) — the mechanism does learn real
  structure on the trained set.
- Held-out accuracy (n=60, 3 items/run): candidate = **0.5889**
  `[0.4944, 0.6611]`; marginal = 0.6333; nearest-neighbor = 0.5722;
  exact-lookup = 0.6667; shuffled-relation-control = 0.6056.
- Candidate's CI does not clear the 0.65 floor and is not separated from the
  baseline rungs (exact-lookup and marginal both sit at or above it) →
  qualification is fully satisfied and the conjunctive behavioral criterion
  genuinely was not met — a real `NotDemonstrated`, not a blocked/Inconclusive
  outcome.

**Interleaved schedule — resolves to `Inconclusive`:**
- Positive control again passed (0.5400).
- Held-out accuracy: candidate = **0.5389** `[0.4611, 0.6056]`; marginal =
  0.6333; nearest-neighbor = 0.5722; exact-lookup = 0.6667;
  shuffled-relation-control = **0.6611**.
- The shuffled-relation-control landed at 0.6611 — outside the pre-registered
  ±0.15 tolerance around its 0.5 chance expectation
  (`baseline_ladder_behaves_as_predicted` fails for this arm only) → per the
  preregistered criteria, a failing baseline-ladder check on the *control
  designed to catch spurious structure* forces `Inconclusive` rather than
  `NotDemonstrated`, since it means this arm's task design itself did not
  validate as intended, independent of the candidate's own result.

**Combined report:** `Inconclusive`, correctly derived via the existing
`UalRuntimeQualification`'s symmetric `.and()` combinator (the same fix from
the prior P2/P4a claim-integrity pass) — one qualified-and-failing arm
(Blocked: `NotDemonstrated`) plus one disqualified arm (Interleaved:
`Inconclusive`) combine to `Inconclusive` overall, not silently to whichever
arm ran first.

## Interpretation

- The candidate mechanism demonstrably learns the trained associations
  (positive control passes cleanly, both schedules) but does **not**
  generalize the relational rule to held-out compounds above a practically
  meaningful margin — in the one schedule where the task design itself
  validated cleanly (Blocked), the result is an honest, qualified
  `NotDemonstrated`.
- The Interleaved arm's `Inconclusive` status is a **task-design validation
  failure on the negative control**, not a claim about the candidate
  mechanism one way or the other. The most likely explanation is sampling
  noise: with only 3 held-out items per run, per-run accuracy is one of 4
  discrete values (0, 1/3, 2/3, 1), so `n=60`'s aggregate mean has coarse
  granularity and a single-schedule tolerance miss at the ±0.15 boundary is
  plausible without any real confound. This is a hypothesis, not a finding —
  it was not investigated further.

## What was NOT done (and why)

- **No threshold was loosened after seeing the Interleaved result.** The
  shuffled-relation-control tolerance was set at ±0.15 before either schedule
  arm was run and is identical to the marginal/nearest-neighbor tolerance
  used throughout this file. Widening it specifically because the observed
  value (0.6611) sat just outside it would be exactly the post-hoc
  parameter-tuning this project's standing discipline forbids.
- **Sample size (n) was not increased** to try to resolve the Interleaved
  ambiguity. Increasing `n` immediately after observing an unfavorable result
  carries the same post-hoc-tuning risk even though it is statistically
  more defensible in the abstract than threshold-widening — a larger `n`
  would need to be pre-registered as its own separate work unit, run without
  knowledge of whether it would help, to be trustworthy.
- **No claim of "UAL demonstrated" is licensed by this result** — the
  outcome is `Inconclusive`, and even a clean pass would only license
  "initial compositional associative-learning profile demonstrated" per the
  design doc's disclosure requirement.

## What "frozen" means here

Any future work that touches this probe's mechanism, baseline ladder, or
thresholds should diff its new results against exactly the numbers above,
run at the same configuration (`dimension=512`, `n=60`). A change in these
numbers is expected and useful evidence about what changed — but only if
this record stays untouched as the "before" side of that comparison. A
follow-up `n`-scaling or Interleaved-arm re-investigation, if ever done,
should be scoped as its own separate, preregistered work unit, not a
quiet re-run of this one.
