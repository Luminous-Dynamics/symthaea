# Symthaea — UAL Phase-1 Protocols: P1, P2, P4a (Gate B specification)

**Status: Pre-registered specification — no code written. Gate B (this document) and
Gate A (see below) are both now closed as of 2026-07-27.** This document satisfies
Gate B of `SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md` (same directory): "at least three
Phase-1-specified probes (P1, P2, P4a) with baseline ladders, controls, and
multi-schedule replication plans fully written out — not just prose sketches." Every
type, function signature, and field name below is quoted from the real
`crates/domains/symthaea-psych-bench` codebase (verified 2026-07-27) so this spec is
directly implementable, not aspirational. Pseudocode blocks are **specification, not
implementation** — nothing here has been compiled or run. Do not treat this document as
evidence a UAL probe exists; it defines what Gate B's "fully specified" requirement
means before Phase 1 implementation starts.

Read the parent design doc first for the full rationale (why UAL sits outside Butlin,
the P1–P6 family, the capacity/explanatory-novelty split, the internal/behavioral
reporting split, and the schedule-robustness rule this document operationalizes).

**IMPLEMENTATION UPDATE (2026-07-27, same day)**: P1/P2/P4a were implemented per this
spec, then subjected to a claim-integrity repair pass after an independent review plus
this codebase's own direct verification found real defects — some in this spec itself,
not just the code. Two are load-bearing enough to flag here rather than only in the
commits: (1) **P2's originally-specified behavioral criterion
(`b_inferred_value - c_value > 0`) was mathematically incapable of detecting success**
(A and C are reward-matched, so their expected values are equal regardless of retrieval
quality) — replaced with a retrieval-identity criterion (rate at which querying with B
ranks A above C, vs. chance) plus a value-transfer check against a new neutral stimulus
D; (2) **P2's confidence-weighted pairing mechanism, meant to satisfy the
schedule-robustness rule below, was proven algebraically inert** (an executable proof
test now pins this) and has been removed — P2's honest finding is that this specific
mechanism is schedule-invariant by construction, unlike P4a's. See commits `7784cab403`
and `99382557e5` for full detail, and the two probes' own module docs in
`crates/domains/symthaea-psych-bench/src/benchmarks/ual/` for the authoritative current
design — the P2 section below describes the ORIGINAL (partially incorrect) spec and is
kept for historical record, not as a description of what shipped.

## Common apparatus (shared by all three protocols)

**RNG/seeding**: reuse the existing convention exactly —
`config.trial_seed(benchmark, condition, trial_idx)` (`harness/config.rs:191-198`),
XORed with `0x9E3779B97F4A7C15` and fed to the shared xorshift PRNG already used
identically in `reward_learning.rs:19-24` and `srtt.rs:43-47`. New benchmark name:
`"ual"`. Condition strings per probe/phase are given in each protocol below, e.g.
`config.trial_seed("ual", "p2_phase1", trial_idx)`.

**Representation**: stimuli are `symthaea_core::hdc::ContinuousHV` (`srtt.rs:24`),
generated via `ContinuousHV::random(dim, seed)`. This is a deliberate departure from
`reward_learning.rs`'s plain `[f64; 2]` Q-value array — P2 and P4a both require
compositional binding (`ContinuousHV::bind`, `weighted_bundle`, `similarity`), which a
scalar Q-value array cannot represent. P1 is the exception (see below): it stays on the
existing `[f64; 2]` mechanism deliberately, to isolate "does the harness/reporting
plumbing work" from "does the new HDC-based association mechanism work."

**Output type**: every probe returns a `BenchmarkResult` (`benchmark: String,
config_label: Option<String>, metrics: BTreeMap<String, MetricValue>, elapsed_ms: u64,
conditions: usize, trials_per_condition: usize, trial_trace: Vec<TrialOutcome>, notes:
Vec<String>`). `MetricValue` fields come from `MetricValue::from_samples_bootstrap(samples,
seed)` (BCa bootstrap — preferred over the plain SE×t-critical `from_samples` given small
per-condition N expected here).

**Schedule field — does not exist yet, must be added before implementation**: grep
confirmed `BenchmarkConfig` has no `schedule`/`ordering`/`interleave` field anywhere, and
`AblationConfig` is just `{ name: String, base: BenchmarkConfig }` with no scheduling
concept. This spec assumes a new field (not yet written):

```text
enum UalSchedule { Blocked, Interleaved }   // one new field on a UAL-local config
```

Adding this field is in-scope for Phase 1 implementation, not a blocker to this
specification — it is called out explicitly here per the schedule-robustness standing
rule, so it is not silently forgotten when implementation starts.

**Reporting format** (per the design doc's mandatory three-field + schedule-qualifier
format):

```text
UAL-P<n> functional outcome: <Demonstrated | NotDemonstrated>  — <schedule status>
Internal association formation: <Observed | NotObserved>
Behavioral expression: <Observed | NotObserved>
```

mapped onto the existing `report.rs` vocabulary: `SupportTier::Observed` for
mechanism-only evidence, `SupportTier::FunctionallySupported` reserved for a UAL
functional-outcome `Demonstrated` result specifically (never for internal-only evidence
— see the design doc's Learning-versus-expression section).

---

## Protocol P1 — Reversal learning

### Rationale for minimal design effort
P1 exists primarily to prove the harness/reporting pipeline end-to-end using
**already-implemented** infrastructure before any new mechanism is introduced. It
deliberately reuses `reward_learning.rs`'s existing `[f64; 2]` Q-value / `softmax_choice`
mechanism (`reward_learning.rs:19-44`) rather than switching to `ContinuousHV` — changing
both the mechanism and the harness plumbing in the same probe would make a failure
ambiguous between "harness bug" and "mechanism bug."

### Stimuli & phases
- Two stimuli, `A` and `B` (choice indices 0/1, as in the existing benchmark).
- Phase 1 (acquisition): `A → reward` (p=0.8), `B → nothing` (p=0.2), for 40 trials.
- Phase 2 (reversal): contingency flips, 40 trials.
- No test phase needed — reversal itself is the measured event, unlike P2/P4a.

### Two schedules (multi-schedule replication)
- **Abrupt**: reversal occurs at a fixed trial index (trial 41), matching the existing
  benchmark's current behavior exactly.
- **Probabilistic**: after a 10-trial warm-up, each subsequent trial has a fixed 5%
  hazard of triggering the reversal (seeded via `config.trial_seed("ual", "p1_hazard",
  trial_idx)`), so the reversal point varies by seed but the total trial budget and
  contingency probabilities stay identical.

### Metrics (already implemented, reused as-is)
`trials_to_criterion` (first trial where a 5-trial rolling fraction of correct choice
exceeds 0.8, `reward_learning.rs:134-140`) and `lose_shift_ratio`
(`lose_shifts / losses`, `reward_learning.rs:143-147`).

### Baseline ladder
1. **Value table**: static, unlearned `[0.5, 0.5]` — never updates. *Expected*: never
   reaches criterion in either phase — this is the "did the benchmark's criterion
   threshold even discriminate learners from non-learners" sanity check.
2. **First-order learner**: standard delta-rule Q-update (`Q += α(reward − Q)`), no
   change from the existing implementation. *Expected*: reaches criterion in both phases
   — this is the pre-existing behavior and must not regress.
3. **Graph propagation**: not meaningfully distinct from rung 2 for a 2-stimulus task —
   report as "not discriminating for P1," not force a synthetic difference.
4. **Static HDC binding**: not applicable to a scalar Q-value task — report as N/A for
   P1 specifically (this is expected and fine; P1's job is harness validation, not
   baseline discrimination).
5. **Full Symthaea**: whatever value-update mechanism the live cognitive loop actually
   uses when routed through this benchmark (not yet wired — implementation-time decision,
   out of scope for this spec).

### Controls
- **Positive control** (`PositiveControlPlan`): `id: "p1-reversal-signal"`, `purpose:
  ControlPurpose::StimulusResponsiveness`, `protocol_group: "ual-p1"`, `description:
  "unmodified first-order learner must reach criterion in both phases and register a
  reversal"`, `expected_effect: <criterion reached both phases, lose_shift_ratio > 0
  post-reversal>`.
- **Sham control** (`ShamControlPlan`): `lever: "unrelated-motor-noise-injection"`,
  `group: "ual-p1-sham"`, `rationale: "a manipulation unrelated to value-update should
  not change trials_to_criterion"`, `matched_dimensions: [trial_count, rng_seeding]`,
  `maximum_allowed_global_impairment: Some(0.05)`.

### Internal vs. behavioral
- **Internal**: the Q-value (or full-Symthaea equivalent) for the current best stimulus
  must have measurably crossed the other stimulus's value within the phase.
- **Behavioral**: `trials_to_criterion` must be finite (criterion reached) in both
  phases — a purely internal crossing with no behavioral criterion-reaching is reported
  per the mandatory split, not conflated.

### Leakage test
Assert the benchmark's trial generator never reads `phase` or `trial_idx` inside the
choice-scoring function itself (only inside contingency assignment) — i.e., the agent's
choice mechanism has no access to ground-truth phase boundaries, only to its own
accumulated value estimate.

---

## Protocol P2 — Second-order conditioning

### Stimuli & representation
Three `ContinuousHV`s, `dim` TBD at implementation (match `srtt.rs`'s existing dimension
choice for consistency): `A_hv = ContinuousHV::random(dim, seed_a)`, `B_hv =
ContinuousHV::random(dim, seed_b)`, `C_hv = ContinuousHV::random(dim, seed_c)` (C is the
**shuffled-pairing control stimulus**, see negative control below). Seeds must be drawn
such that `A_hv.similarity(&B_hv)` and `A_hv.similarity(&C_hv)` are both within a
preregistered near-chance band (e.g. `|similarity| < 0.1`) — checked at generation time
and re-drawn on failure, so representational similarity cannot itself explain any later
transfer (directly addresses the "representational similarity between A and B" alternative
explanation named in the design doc).

### Value & relational-memory mechanism (full-Symthaea condition)
- **Value**: scalar value keyed by hypervector identity (a `HashMap<StimulusId, f64>`
  analogous to `reward_learning.rs`'s Q-values, not a new mechanism).
- **Relational link**: on each Phase-2 trial, `bind_hv = B_hv.bind(&A_hv)` is formed and
  accumulated into a relational memory via `ContinuousHV::weighted_bundle(&[memory,
  bind_hv], &[0.9, 0.1])` (same EMA-accumulation pattern as `srtt.rs:83-86`), stored
  separately from the value table.
- **Test-time query**: given `B_hv` alone, compute `query = memory.bind(&B_hv)` (unbind,
  since `bind` is self-inverse per the codebase's existing binding algebra), then
  retrieve the stimulus in `{A_hv, C_hv}` with highest `similarity()` to `query`, and
  look up *that* stimulus's stored value. This is a genuine relational-chaining
  mechanism, not direct value copying — it can fail structurally (e.g. if the retrieved
  neighbor is wrong) in a way plain value propagation cannot.

### Phases (40 trials each unless noted) + test
- **Phase 1**: `A_hv → reward` (p=0.8, matching P1's contingency for consistency).
  `C_hv → reward` at an *independent*, separately-seeded p=0.8 as well — C must acquire
  its own real value, not remain unlearned, so the shuffled-pairing control is a true
  peer of A, not a strawman.
- **Phase 2**: `B_hv` paired with `A_hv` (no reward presented directly) — the
  second-order link.
- **Test** (extinction, 20 trials, **no reward ever delivered**, both to prevent
  test-phase learning from contaminating the measurement and to isolate transferred
  value from real-time reinforcement): present `B_hv` alone, measure choice/value
  read-out.

### Two schedules
- **Blocked**: all 40 Phase-1 trials, then all 40 Phase-2 trials, then test.
- **Interleaved**: Phase-1 and Phase-2 trial types randomly interleaved (same 40+40
  totals), order seeded via `config.trial_seed("ual", "p2_interleave_order", i)`.

### Baseline ladder
1. **Value table**: no relational memory at all — `B_hv` has no stored value.
   *Expected*: fails (no transfer possible by construction).
2. **First-order learner**: learns direct pairings only, no chaining mechanism.
   *Expected*: fails — `B_hv` was never directly reinforced.
3. **Graph propagation**: explicit directed graph, edge `B→A` added with weight 1.0 on
   every Phase-2 trial, second-order value estimated via 1-hop BFS from `B` to `A`'s
   value node. *Expected*: **passes** — this is the calibration case, confirming the
   task is solvable by a mechanism with zero "understanding," per the design doc's
   explicit purpose for this rung.
4. **Static HDC binding, no learned temporal dynamics**: bind/bundle only, no value
   update at all (`bind_hv` accumulated exactly as above but the value table is frozen at
   initialization). *Expected*: fails on the value read-out (there is no value to
   retrieve) — isolates representational transfer from value transfer.
5. **Full Symthaea**: the mechanism above. *Empirical.*

### Controls
- **Positive control**: `id: "p2-direct-pairing-signal"`, `purpose:
  ControlPurpose::StimulusResponsiveness`, `description: "direct A_hv→reward pairing
  alone (no B) must produce a normal acquisition curve"`, `expected_effect: <A_hv value
  rises to reward-matching asymptote within Phase 1's 40 trials>`.
- **Negative control 1 (shuffled-pairing / nonspecific-drift)**: `C_hv`, matched on
  independent direct reward history, receives **no** Phase-2 pairing with anything.
  *Expected*: `C_hv`'s test-time value read-out stays at its Phase-1 asymptote — no
  drift toward `B_hv`'s or `A_hv`'s territory. A `C_hv` value shift would indicate
  generalized value inflation, not specific second-order transfer.
- **Negative control 2 (lookup-table / memorization)**: a table literally keyed by
  `(pair_seen_directly: bool)` — by construction it cannot assign `B_hv` any value since
  `B_hv` was never in a directly-rewarded pair. This is baseline rung 1, restated as an
  explicit control rather than only a ladder entry, per the design doc's note that "fixed"
  alone is an insufficient lookup-table control — this version is intentionally
  incapable of relational extension at all, unlike rung 3's graph, to bound the weakest
  possible non-learner.
- **Sham** (`ShamControlPlan`): `lever: "unrelated-dimension-perturbation"`, `group:
  "ual-p2-sham"`, `rationale: "perturbing an HV dimension unrelated to the A/B/C triple
  should not move B_hv's test-time value"`, `matched_dimensions: [hv_dim, bundle_decay,
  trial_count]`, `maximum_allowed_global_impairment: Some(0.05)`, `reuses_target_of:
  None`.

### Internal vs. behavioral
- **Internal**: `memory.bind(&B_hv)`'s similarity to `A_hv` must exceed similarity to
  `C_hv` by a preregistered margin (e.g. Δ > 0.15) after Phase 2.
- **Behavioral**: the test-phase choice/value read-out for `B_hv` must be statistically
  distinguishable from `C_hv`'s (paired bootstrap CI excluding zero, `n≥20` test trials,
  `MetricValue::from_samples_bootstrap`). Per the mandatory split, an internal margin
  without a behavioral difference is reported as `NotDemonstrated` with `Internal
  association formation: Observed`.

### Leakage test
Assert the test-phase trial generator has zero code path that reads Phase-2's stored
`(B,A)` pairing directly when scoring `B_hv`'s test value — the *only* permitted path is
through the `memory` hypervector's similarity retrieval. (Concretely: a unit-level check
that deleting/zeroing `memory` collapses `B_hv`'s test value to the value-table default,
proving the retrieval path is load-bearing, not bypassed.)

---

## Protocol P4a — Held-out compositional recombination

### Stimuli & representation
Four element `ContinuousHV`s: `W, X, Y, Z` (`ContinuousHV::random`, mutually
near-chance similarity per the same generation-with-rejection rule as P2). Two of the
six possible pairs are trained; **one specific pair is held out entirely and never
appears, in any bound form, during training** — this is the anti-memorization
requirement the design doc names as load-bearing.

### Phases
- **Training** (60 trials total, split evenly): `bind(W,X) → reward` (p=0.8),
  `bind(Y,Z) → nothing` (p=0.2). Individual elements `W, X, Y, Z` are never presented
  alone during training — only as the two trained compounds — so a per-element value
  table cannot be built by construction (this is what forces rung 2's expected failure
  below).
- **Test** (extinction, no reward, 3 conditions × 15 trials each, order randomized):
  1. `bind(W,X)` — a **seen** compound (positive control, confirms the readout pathway
     works at all).
  2. `bind(W,Z)` — the **held-out novel compound** (the critical measurement — never
     presented, bound or unbound, during training).
  3. `bind(Y,X)` — a second held-out compound, included so the critical result is a
     pattern across two novel compounds, not a single lucky draw.

### Two schedules
- **Blocked-by-element**: all `bind(W,X)` training trials first, then all `bind(Y,Z)`
  trials.
- **Interleaved-by-element**: the two trained-compound trial types randomly interleaved
  (same 30+30 totals), order seeded via `config.trial_seed("ual", "p4a_interleave_order",
  i)`.

### Baseline ladder
1. **Value table**: keyed by exact compound identity (a hash of the bound vector, or
   equivalently the `(element-pair)` tuple). *Expected*: **fails** on both held-out
   compounds by construction — no entry exists.
2. **First-order learner**: learns a value for each *trained compound as a whole*, no
   per-element decomposition. *Expected*: fails identically to rung 1 — same reason.
3. **Graph propagation, adapted for compounds**: value of a novel compound estimated as
   the mean of its two elements' *marginal* values, where each element's marginal value
   is estimated as the mean value of the trained compounds it appeared in (`W`'s marginal
   ≈ `bind(W,X)`'s value; `Z`'s marginal ≈ `bind(Y,Z)`'s value). *Expected*: **passes** —
   this is the calibration case for P4a, analogous to rung 3's role in P2: confirms the
   task is solvable by naive additive/marginal composition, without requiring the system
   to represent compounds as anything beyond independent parts.
4. **Static HDC binding, no value learning**: compute `bind(W,Z)` at test time and find
   its nearest neighbor by `similarity()` among the *trained* bound vectors
   (`bind(W,X)`, `bind(Y,Z)`); if a genuine representational-transfer confound exists
   (the alternative explanation named in the P4 section of the design doc), this rung
   should show above-chance similarity between the novel and a trained compound purely
   from vector-space geometry, with no value component at all. *Expected*: this rung's
   result calibrates how much of any "full Symthaea" pass is attributable to geometry
   alone — a required comparison, not a pass/fail target.
5. **Full Symthaea**: the mechanism under test — must combine `bind(W,Z)` at test time
   with a *compositional* value-integration step (implementation-time decision: e.g., a
   trained value-integration function over bound-vector components, not simply rung 3's
   marginal-mean) such that it is distinguishable from rung 3 by design, not just by
   outcome — otherwise a "full Symthaea" pass would be indistinguishable from naive
   additive composition and could not support an explanatory-novelty claim. *Empirical.*

### Controls
- **Positive control**: seen compound `bind(W,X)` at test — `expected_effect: <test
  value read-out matches Phase-1 asymptotic value within CI>`. If this fails, the readout
  pathway itself is broken and the held-out results are uninterpretable.
- **Negative control (memorization)**: rung 1/2 above, restated as an explicit control —
  their guaranteed failure on held-out compounds is the anti-memorization floor.
- **Negative control (pure geometric transfer)**: rung 4 above — if "full Symthaea"'s
  held-out performance is statistically indistinguishable from rung 4's, the result must
  be reported as "representational transfer, not integrated compositional valuation,"
  per the design doc's explanatory-novelty criterion (capacity may still be
  `Demonstrated`, but explanatory novelty is `NotSupported`).
- **Sham** (`ShamControlPlan`): `lever: "irrelevant-dimension-noise-on-held-out-query"`,
  `group: "ual-p4a-sham"`, `rationale: "noise on dimensions uncorrelated with W/X/Y/Z
  should not change held-out compound value read-out"`, `matched_dimensions: [hv_dim,
  trial_count, reward_probability]`, `maximum_allowed_global_impairment: Some(0.05)`.

### Internal vs. behavioral
- **Internal**: the value-integration function's output for `bind(W,Z)` must differ
  measurably from its output for `bind(Y,Z)` (the trained low-value compound), showing
  the internal computation is sensitive to which *elements* compose the novel query.
- **Behavioral**: the test-phase choice/value read-out for `bind(W,Z)` must be
  statistically distinguishable from `bind(Y,Z)`'s (same bootstrap-CI standard as P2).

### Leakage test
Assert the held-out compounds `bind(W,Z)` and `bind(Y,X)` are checked, at trial-generation
time, against a set-membership test over every compound presented during training
(exact-vector equality, not similarity) — a hard assertion failure if a held-out compound
was accidentally also trained, rather than a silent data-quality issue discovered only
after results look surprising.

---

## What this document does not do

It does not implement any of the above. It does not add a 15th Butlin indicator, an
`ablation_specs()` row, or any `PsychBenchmark` impl.

**Gate A status (updated 2026-07-27, same day as this document)**: Gate A — a real
end-to-end run proving the Butlin qualification pipeline works — closed independently,
via `ae2_empirical_runner.rs` and documented in
`symthaea/docs/BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md` ("Step 5 result"). A real
`CognitiveLoopService` run (200 cycles, 4 arms) produced `Supported(CausallySupported)`
for AE-2, with a real bug found and fixed mid-run. This closure was discovered, not
performed, by the UAL track — see the parent design doc's Gate A section for full
detail and the single-seed caveat. Both gates named in the parent design doc are now
closed; this document's own specification work is unaffected by that discovery — the
protocols above were written independent of Gate A's status and remain what
implementation should follow when it begins.

## See also
- `SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md` (same directory) — parent design,
  rationale, P1–P6 family, Phase 1/2 split, reporting rules.
- `crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/reward_learning.rs` — the
  mechanism P1 reuses directly.
- `crates/domains/symthaea-psych-bench/src/benchmarks/motor/srtt.rs` — the
  `ContinuousHV` bind/bundle/similarity pattern P2 and P4a extend.
- `crates/domains/symthaea-psych-bench/src/benchmarks/butlin/qualification_design.rs` —
  `PositiveControlPlan`/`ShamControlPlan`/`ControlPurpose`/`ProbeValidity` definitions
  used verbatim above.
- `crates/domains/symthaea-psych-bench/src/benchmarks/butlin/report.rs` —
  `IndicatorEvidence`/`SupportTier` definitions the reporting format maps onto.
