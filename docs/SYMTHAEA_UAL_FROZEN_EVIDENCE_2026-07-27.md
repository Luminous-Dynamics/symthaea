# UAL P2/P4a — Frozen Evidence Record (pre-HDC-audit baseline)

**Status: frozen baseline. Do not edit the recorded numbers/verdicts below.**
This is Phase 0 / Commit A of `HDC Binding Algebra Qualification and
Migration Plan` (see `MASTER_ROADMAP.md`'s UAL row). Its sole purpose is to
give the upcoming `symthaea-core` HDC binding-algebra audit a clean,
timestamped before/after comparison point. **No UAL threshold, parameter, or
mechanism change happens during that audit's Phases 1–4** — any change to
these recorded numbers is explicitly out of scope until a separate,
preregistered Phase 5 (P4a redesign) or Phase 6 (P2 re-evaluation) work unit
is scoped and run.

## Source commits

- `7784cab403` — fail-closed qualification model + P1 metric/schedule repair
  (first claim-integrity pass, part 1).
- `99382557e5` — retract inert P2 schedule claim, fix behavioral criterion,
  rename `FullSymthaea`, add HDC binding audit (first pass, part 2).
- `da96bf5b9b` — second claim-integrity pass: symmetric qualification
  combination, P4a fail-closed via `construct_validity_passed`, P2 requires
  genuine value transfer via a real choice mechanism.
- `4d26f1be52` — MASTER_ROADMAP correction to match `da96bf5b9b`'s verdicts.

## Exact configuration

- `dimension = 512`
- `trials_per_condition (n) = 40`
- Both `UalSchedule` arms run and combined via `combine_schedule_reports`
  (Blocked + Interleaved for P2/P4a).
- Benchmark crate: `crates/domains/symthaea-psych-bench`, module
  `src/benchmarks/ual/`.

## UAL-P2 (`p2_second_order.rs`)

```text
UAL-P2 functional outcome: NotDemonstrated — ScheduleScoped
System under test: benchmark-local candidate HDC learner (NOT live Symthaea)
Internal association formation: Observed
Behavioral expression: NotObserved
```

- Relational-identity (does querying with B rank A over reward-matched C?)
  rate = **1.0000** `[1.0000, 1.0000]` under both Blocked and Interleaved.
- Value-transfer (forced choice: B vs. neutral D, via `softmax_choice`) rate
  = **0.6250** `[0.4500, 0.7500]` under both schedules — **the bootstrap CI
  includes chance (0.5)**.
- `mean(sim_A - sim_D) = 0.7106`; `mean A_value = 0.6729`.
- Functional outcome is `NotDemonstrated`, not `Inconclusive` — this run's
  qualification (`UalRuntimeQualification::all_passed()`) is fully satisfied;
  the conjunctive behavioral criterion (relational identity AND value
  transfer) genuinely was not met at this `n`/temperature.
- Reported note, verbatim: *"relational retrieval observed WITHOUT
  value-transfer/behavioral expression — this supports only 'P2 relational
  retrieval', not 'second-order value conditioning'."*
- Schedule caveat (always attached): P2's accumulator is proven
  algebraically schedule-invariant for this mechanism (`bind_hv` is the same
  fixed vector on every pairing step) — "replicated across schedules" here
  means the two orderings are algebraically equivalent for this learner, not
  that a meaningful manipulation was varied and survived.

## UAL-P4a (`p4a_recombination.rs`)

```text
UAL-P4a functional outcome: Inconclusive — ScheduleScoped
System under test: benchmark-local candidate HDC learner (NOT live Symthaea)
Internal association formation: Observed
Behavioral expression: NotObserved
```

- Forced to `Inconclusive` via `UalStaticQualification::construct_validity_passed
  = false` — the HDC binding/unbinding semantics underlying this probe's
  retrieval mechanism are an open question (see `hdc_binding_properties.rs`).
- Underlying diagnostic (preserved as a note, NOT a claim): mean(WZ_readout −
  YZ_readout) = **0.2027** `[0.1816, 0.2245]` under Blocked, **0.1362**
  `[0.1131, 0.1588]` under Interleaved (both CIs exclude zero — a real,
  reproducible effect exists in the raw numbers). Mean internal margin diff:
  0.4053 (Blocked), 0.2724 (Interleaved).
- This numeric effect is explicitly **not** evidence of compositional
  recombination until the binding-algebra audit resolves whether `.bind(
  compound_hv)` (as opposed to `.bind(&compound_hv.inverse())`) measures
  genuine compositional structure or a geometric/self-product artifact of
  `ContinuousHV::random`'s real uniform distribution.

## What "frozen" means here

Any future work that touches `symthaea-core`'s `ContinuousHV`/`BinaryHV`
binding algebra, or `p2_second_order.rs`/`p4a_recombination.rs` themselves,
should diff its new results against exactly the numbers above, run at the
same configuration. A change in these numbers after the HDC audit lands is
expected and useful evidence about what the audit changed — but only if this
record stays untouched as the "before" side of that comparison.
