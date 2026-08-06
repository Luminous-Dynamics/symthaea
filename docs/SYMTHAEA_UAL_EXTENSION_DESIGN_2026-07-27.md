# Symthaea — Unlimited Associative Learning (UAL) Extension Design

**Status: Candidate extension — technically feasible, not operationally qualified.**
Created 2026-07-27. This is a design document only — **no UAL probe, indicator, or
ablation row has been implemented.** Do not cite this document as evidence Symthaea
possesses or lacks UAL; it exists to define what would have to be true before such a
claim could be made honestly.

**Revision note (2026-07-27, same day)**: incorporates five corrections from an
independent second-pass review — separating capacity from explanatory-novelty claims,
operationalizing the (non-)enumerability requirement via combinatorial scaling rather
than an unprovable logical claim, splitting P4 into a probe family so "unlimited" is
supported by a scaling trend rather than one data point, tightening internal-vs-
behavioral reporting semantics, and phasing implementation into capability qualification
before mechanism localization — plus two added standing rules (multi-schedule
replication, cross-stage dependency accounting).

## Why UAL sits outside the frozen Butlin denominator

The canonical Butlin-14 indicator suite
(`crates/domains/symthaea-psych-bench/src/benchmarks/butlin/indicators.rs`) is a fixed,
literature-anchored set (RPT-1/2, GWT-1/2/3/4, HOT-1/2/3/4, PP-1, AST-1, AE-1/2). IIT was
deliberately excluded from it because Chalmers/Butlin et al.'s report assumes
computational functionalism, which conflicts with IIT's intrinsic-causal-structure
claims — exclusion by design, not oversight (`indicators.rs` framing comment).

UAL (Birch, Ginsburg & Jablonka's proposed transition marker) is a different kind of
claim than any of the 14: it is a **functional learning-capacity theory**, not an
architectural-mechanism theory. Folding it into the Butlin count would silently change
what the "14/14" denominator means — mixing "does this architecture exist" indicators
with "can this system learn in an unbounded, compositional way" indicators. It is kept
as a separate, explicitly labeled extension track precisely so the frozen suite's
existing evidence stays interpretable, and so a future UAL verdict is legible as its own
thing rather than diluting or inflating the canonical count.

## Primary-source definition of the claimed capacity

UAL, per its originating literature, is not "the system can learn associations." It is
the capacity for **open-ended, compositional, revisable associative learning** —
operationally: the ability to form associations that (a) generalize to novel
combinations never directly reinforced, (b) can be revised when contingencies change,
(c) can chain through unobserved intermediate relations (second-order and higher), and
(d) continues to generalize as the combinatorial space of stimuli/relations grows,
rather than degrading toward a size where an explicit lookup table becomes a plausible
account of the observed behavior. A system that performs well on any *one* of these
sub-capacities has not thereby demonstrated UAL — UAL is the conjunction, and
demonstrating the conjunction requires demonstrating each component with controls that
rule out the other, simpler explanations for that specific component's result.

**On (d) specifically**: no finite experiment can literally prove behavior is "not
reducible to any enumerable table" — any finite experiment can in principle be
represented by a sufficiently large table. (d) is therefore operationalized empirically,
not asserted logically: as combinatorial-holdout performance under controlled scaling
(see the UAL-P4 family below), compared explicitly against explicit table-based and
graph-propagation baselines. The defensible claim this document supports is "performance
cannot be explained by a lookup table over the observed training support without
explicitly enumerating the held-out combinatorial space" — not "performance is not
reducible to any table."

This document treats UAL as a **capability profile**, not a single pass/fail bit.

## What the code audit actually established

(From the grounded investigation of `symthaea-psych-bench` and the main cognitive loop,
2026-07-27 — full citations below.)

- `benchmarks/butlin/report.rs` already separates `architectural_score` from
  `live_score` per indicator, and never blends them — this is the correct evidentiary
  shape for UAL sub-probes too: a probe passing should register as live/functional
  evidence for *that probe*, not get algebraically folded into a single UAL number.
- `benchmarks/butlin/ablation.rs` and `qualification_design.rs` already implement a
  positive-control / sham-control / validity-tier pattern
  (`ControlPurpose`: Instrumentation < StimulusResponsiveness < MechanisticResponsiveness;
  `ProbeValidity`: DirectMeasure/BehavioralProxy/ExecutionProxy/Unverified). Any UAL probe
  must be designed against this same taxonomy before it's implemented, not after.
- `benchmarks/neuromod/reward_learning.rs` already implements a reversal-learning
  paradigm (A→reward, B→nothing, then reverse; trials-to-criterion, lose-shift ratio).
- `benchmarks/motor/srtt.rs` already implements HDC bind-and-accumulate transition
  memory (structured-vs-random sequence RT comparison) — a template for compound-stimulus
  binding.
- `crates/core/symthaea-core/src/hdc/hebbian.rs` exists as a lower-level associative
  primitive but is not currently consumed by either benchmark above.
- `symthaea-alife`'s MA-001 finding (`memory/symthaea_alife_ma001_jul26.md`) is directly
  relevant and must inform every UAL probe design: learning can occur internally
  (measurable weight/state change) while the action-selection/output layer fails to
  express it behaviorally. A UAL probe that only checks internal state, not behavior,
  would reproduce this exact false-positive risk.

None of the above constitutes a UAL probe. It constitutes reusable scaffolding.

## Staged probe family

UAL is represented as an escalating family, not one indicator. Each stage is
independently falsifiable and independently reportable — passing an earlier stage is
necessary but not sufficient for claiming the later ones.

### UAL-P1 — Reversal learning
Learned value must update when reward contingency flips.
**Status**: infrastructure largely exists (`reward_learning.rs`).
**Alternative explanations to rule out**: simple recency-weighted value decay with no
real "reversal" representation — a system that just forgets old associations fast enough
would pass this without any flexible relational structure.

### UAL-P2 — Second-order conditioning
Learn A→reward, then B→A (no direct B→reward pairing); test whether B alone acquires
value.
**Alternative explanations**: ordinary value propagation through a fixed transition
table; spreading activation; representational similarity between A and B producing
transfer without any real relational chaining; direct-readout adaptation at the output
layer rather than compositional association.

### UAL-P3 — Compound discrimination
E.g., A rewarded, B rewarded, A+B not rewarded (or the reverse — a nonlinear/negative
patterning design). Tests whether the system represents compounds as more than the sum
of marginal feature values.
**Alternative explanations**: independent per-feature value accumulation that happens to
average out; overfitting to compound-as-a-whole without genuine compositional structure
(distinguishable via P4).

### UAL-P4 — Novel recombination (a probe family, not a single test)
Train component relationships separately; test compositions that were never directly
reinforced in any form during training. **This is the load-bearing probe family for
distinguishing UAL from memorization and for supporting the word "unlimited"** — a
single novel-compound test shows compositional transfer, not open-endedness. Structured
as an escalating sub-family so open-endedness is demonstrated by a scaling trend, not
asserted from one data point:

- **P4a** — one unseen compound of already-known elements.
- **P4b** — an unseen application of an already-learned relation.
- **P4c** — two-step relational composition (chaining two learned relations at test time).
- **P4d** — increasing symbol vocabulary / combinatorial-space size at fixed training
  coverage fraction.
- **P4e** — increasing relational-composition depth.

The held-out unit at every sub-stage must be a whole compound/relation, never a trial
instance of an already-seen compound. The strongest result is a **scaling curve**, not a
single score: training-example count, total possible combinations, held-out relation
classes, accuracy by composition depth, and memory/compute growth — each plotted against
the explicit table-based and graph-propagation baselines from the ladder below. A flat or
degrading curve as the combinatorial space grows should be reported as "flexible
associative learning, bounded scaling regime demonstrated," not "unlimited."

**Alternative explanations (all sub-stages)**: any degree of representational overlap
between a novel compound and a seen one could produce transfer via similarity rather
than composition — requires a similarity-matched-but-untrained-relation control (see
baselines below).

### UAL-P5 — Trace association
Separate paired events by a temporal gap (no simultaneous presentation); test whether
association still forms.
**Alternative explanations**: this specifically stresses whatever temporal/working-memory
substrate bridges the gap — a pass here could reflect a general working-memory capacity
unrelated to associative learning per se, and must be reported as such if so.

### UAL-P6 — Cross-context transfer
Learn a relationship in one context; test whether it generalizes appropriately (transfers
when it should, stays context-bound when it shouldn't).
**Alternative explanations**: this is a two-sided test — a system that transfers
everything indiscriminately fails as badly as one that never transfers. The control must
include cases where transfer would be a bug.

The final UAL report is a **structured per-stage profile** (e.g. "P1 functional outcome:
Demonstrated, BehavioralProxy. P2 functional outcome: NotDemonstrated (internal
association formation: Observed). P4a: not yet designed...") — never a single UAL
present/absent verdict, and never presented as a 15th Butlin indicator. See "Track
dependencies between stages" below for how passes across multiple stages must be
collapsed to independent evidence units before being summarized.

## Baselines required for every probe (not just the lookup-table check)

A single "fixed lookup table" negative control is insufficient — a lookup table can be
extended to encode second-order relations, so "fixed" alone doesn't guarantee it can't
solve the task. Each probe must be run against a baseline ladder, and Symthaea's result
must be reported relative to *each* rung, not just against the weakest one:

1. Direct stimulus-value table, no relational propagation.
2. First-order associative learner (learns direct pairings only, no chaining).
3. Graph-based value propagation (explicit BFS/spreading-activation over a learned
   graph — deliberately capable of solving P2/P3 without any "understanding," to
   calibrate how much of a pass is explainable by simple propagation).
4. HDC binding without learned temporal dynamics (bind/bundle only, no CfC/Hebbian
   update) — isolates how much of a pass is representational (the binding algebra
   itself) vs. genuinely learned.
5. Benchmark-local candidate learner (`SystemUnderTest::BenchmarkLocalHdcLearner` in the
   real implementation — see the 2026-07-27 claim-integrity repair pass; the label "Full
   Symthaea" used here originally was itself found to be a naming trap and retracted, not
   just a placeholder name).

A probe-level **capacity** claim requires Symthaea to pass its preregistered held-out
behavioral generalization criterion and its manipulation checks (leakage, drift,
memorization) — it does not additionally require beating every baseline. Baseline parity
or superiority answers a separate question:

- **Capacity criterion**: did Symthaea pass the preregistered held-out test while the
  controls ruled out leakage, memorization, and nonspecific drift? This can be true even
  if a simpler baseline (e.g. graph propagation) also passes.
- **Explanatory-novelty criterion**: did Symthaea outperform or qualitatively exceed the
  simpler baselines that can also produce the behavior? This determines how
  parsimoniously the result can be explained, and whether any Symthaea-specific
  mechanistic contribution is supported.

Baseline parity defeats a claim of *mechanistic novelty* — it does not erase an *observed
capacity*. Conflating the two would effectively redefine "UAL" as "whatever Symthaea does
that no simple model can also do," rather than testing a stable, independently defined
construct. Both criteria must be reported separately for every probe.

## Target mechanism map (prerequisite for Phase 2 / any ablation row — not for Phase 1)

Unlike a Butlin indicator tied to one architectural mechanism, UAL is a capacity that
could route through several candidate mechanisms. This map, and the causal-intervention
requirement below, apply to **Phase 2 (mechanism localization)** — see "Implementation
is two phases" below. A Phase 1 (capability qualification) probe requires none of this;
it should not be blocked on picking a mechanism before it has even been shown that the
capacity exists behaviorally. Before any `ablation_specs()` row is written for Phase 2,
the design must name which of these it targets, because disabling the wrong one either
proves nothing (targets a mechanism the probe doesn't actually depend on) or proves too
much (disabling a broad learning pathway trivially destroys all learning, Butlin or UAL
alike):

- Hebbian update (`symthaea-core/src/hdc/hebbian.rs`)
- HDC bind/bundle operations (representational substrate, not learning per se)
- Recurrent/CfC state carryover (temporal bridging — relevant to P5 specifically)
- Value propagation / reward-prediction-error pathway
- Neuromodulatory learning-rate gating
- Episodic/associative memory retrieval
- Policy/action-selection expression layer (the MA-001 failure point — see below)

A defensible ablation for a given UAL probe needs, at minimum: (1) a declared target
mechanism from this list, (2) a matched sham ablation (disables something structurally
similar but not expected to move this probe), (3) a positive control proving the probe
*can* move at all under the unablated system, (4) a simpler-task control showing basic
(non-UAL) learning survives the ablation intact, (5) a downstream behavioral-expression
check, and (6) a rescue/alternate-path test where architecturally feasible.

## Learning-versus-expression distinction (mandatory per probe)

Per the MA-001 precedent, every UAL probe must report three separate fields, not a
single tier:

```text
UAL-P<n> functional outcome: <Demonstrated | NotDemonstrated>
Internal association formation: <Observed | NotObserved>
Behavioral expression: <Observed | NotObserved>
```

**The probe's functional-outcome field may only read `Demonstrated` when behavioral
expression is `Observed`.** Internal change alone — association formation without a
corresponding change in action, prediction, or output — is real mechanism evidence and
should be recorded as such (it may independently carry a `report.rs`-style
`SupportTier::Observed` tag on the *mechanism* record), but it must never be attached to
the UAL-stage's functional-outcome field in a way a reader could interpret as partial UAL
support. E.g., an internal-only result for P2 is reported as:

```text
UAL-P2 functional outcome: NotDemonstrated
Internal association formation: Observed
Behavioral expression: NotDemonstrated
```

never as "UAL-P2: Observed."

## Qualification requirement — the rule this document exists to enforce

**No single passed probe (P1 alone, or P2 alone) may be reported, cited, or logged in
MASTER_ROADMAP/memory as "UAL support" or "UAL present."** Any report must either (a)
name the specific stage(s) passed with their functional-outcome / capacity /
explanatory-novelty fields (per the sections above), explicitly caveated against the
stages not yet run, or (b) withhold any UAL-level claim until P1, P2, and P4a (the
minimal packet for *beginning* a UAL-level assessment — see "Minimal implementation
sequence" below) have been run with matched baselines and behavioral-expression checks
per this design. **Even a clean P1/P2/P4a pass does not license "UAL demonstrated"** —
P4a alone is one held-out compositional case, not evidence of open-endedness; report it
as "initial compositional associative-learning profile demonstrated." The words
"unlimited"/"open-ended" require the P4d/P4e scaling evidence specifically (see UAL-P4
above).

**Standing rule — schedule robustness**: every functional-outcome field must carry an
explicit schedule-status qualifier, not just a bare Demonstrated/NotDemonstrated:

```text
Demonstrated — schedule-scoped (single presentation ordering tested)
Demonstrated — replicated across schedules (≥2 structurally different orderings)
NotDemonstrated — schedule-scoped
Inconclusive
```

A probe-level finding is schedule-scoped until replicated under at least one
structurally different presentation ordering with identical semantic content (per
[[feedback_recall_harm_is_schedule_dependent]] — identical content and seeds under a
different tier-count/interleaving previously inverted an unrelated harm finding
elsewhere in this codebase). This applies with particular force to P2 (chain formation
may depend on interleaving), P4 (recombination may depend on curriculum order), P5
(temporal gaps are the manipulated variable), and P6 (context composition changes with
scheduling). The explicit qualifier prevents a later summary from silently dropping the
schedule caveat.

## Implementation is two phases — capability first, mechanism second

A full causal ablation should not gate a first behavioral probe. A valid behavioral
benchmark can be built, and can produce a real positive or null result, before anyone
knows which mechanism carries the capability — it just cannot yet support a
mechanism-specific causal claim. Splitting the work this way avoids guessing an ablation
target too early and then unconsciously designing the task around that guess.

### Phase 1 — Capability qualification
Requires: held-out generalization, positive and negative controls, a behavioral-
expression check, internal-state measurement, the baseline ladder, leakage tests, and
multi-schedule replication (per the standing rule above). Establishes functional
capacity **and** explanatory novelty relative to the baseline ladder (the
`Demonstrated`/`NotDemonstrated`, capacity-criterion, and explanatory-novelty fields
defined above) — but does not establish *which internal Symthaea mechanism* caused the
result. No target mechanism or ablation required.

### Phase 2 — Mechanism localization
Requires: a declared target mechanism (from the map above), a true on/off intervention,
a matched sham, a simpler-task-preservation check, a specificity panel, and a rescue path
where architecturally feasible. Establishes **which mechanism causally carries the
capacity, and whether that mechanism is necessary, specific, or replaceable** — it
populates the target-mechanism and causal-support fields. Explanatory novelty relative
to simpler baselines is already assessed in Phase 1; Phase 2 does not re-derive it, only
attributes it to a mechanism. Attempted only for probes that have already cleared
Phase 1.

## Gates before implementation begins

1. **Gate A — CLOSED 2026-07-27** (by separate, undiscovered-until-now work; not
   originally scoped for this design). The real runner turned out not to be
   `qualification_runtime.rs` itself — that module's own doc comment discloses it
   "does not touch a real `CognitiveLoopService`," and is a pure contract layer tested
   only against 12 synthetic fixtures. The actual end-to-end proof is
   `crates/domains/symthaea-psych-bench/src/benchmarks/butlin/ae2_empirical_runner.rs`,
   documented in `symthaea/docs/BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md` ("Step 5 result,"
   2026-07-27): a real `CognitiveLoopService` (`ConsciousnessProfile::Standard`, 200
   cycles, 20-cycle warmup), 4 arms (baseline / target ablation / sham / positive
   control), all 7 pre-registered qualification questions resolved from live
   measurements. **Outcome: `Supported(CausallySupported)`** —
   `embodied_agency` collapsed 1.0→0.0 under `disable_embodied_cognition`, unmoved
   (1.0) under the sham. A real bug was caught and fixed mid-run (an overly strict
   health-panel check on a module never enabled by `ConsciousnessProfile::Standard`).
   This closes Gate A's requirement — the harness genuinely works end-to-end on a real
   indicator — but the result is explicitly **single-seed**; the source plan's own
   recommendation is to repeat AE-2 across fresh seeds and a second stimulus schedule
   before treating the health-panel tolerance as calibrated. Treat Gate A as
   "mechanism proven," not "AE-2 fully characterized," when citing this closure.
2. **Gate B — CLOSED 2026-07-27** (`SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md`): P1,
   P2, and P4a are fully specified — baseline ladders, controls, and multi-schedule
   replication plans written out against real code, not prose sketches. A named target
   mechanism and causal intervention are required only before a probe enters Phase 2,
   not before Phase 1 implementation begins.

**Both gates are now closed. Implementation of the minimal P1/P2/P4a Phase-1 packet
(per the protocols doc) is no longer blocked on this design's own terms** — though
starting it is a separate decision from having designed it, and the codebase-wide
"verify before duplicating" norm applies: Gate A's closure was discovered, not
performed, by this track, underscoring that this fast-moving monorepo can close a
gate out from under a design before anyone reads it.

## Minimal implementation sequence

Once Gate A is met, start with a three-probe Phase-1 packet rather than P2 alone. This
packet begins a UAL-level assessment — separating basic reversal, indirect transfer, and
one held-out compositional case — it does not by itself establish the full UAL
conjunction (see the qualification requirement above):

1. **P1 — Reversal.** Mostly existing infrastructure
   (`benchmarks/neuromod/reward_learning.rs`). Establishes that the harness, output
   path, and behavioral-expression check all work end-to-end before anything harder is
   attempted.
2. **P2 — Second-order transfer.** Establishes indirect association; remains explainable
   by graph propagation alone, which is an expected outcome to report, not a failure.
3. **P4a — Held-out recombination.** The critical anti-memorization test.

Run the full baseline ladder (value table, first-order learner, graph propagation,
static HDC binding, full Symthaea) across all three probes together — not per-probe in
isolation — to get a capability gradient rather than three disconnected numbers:

| System              | P1                | P2                | P4a                                 |
|---------------------|-------------------|-------------------|--------------------------------------|
| Value table         | possible          | expected fail     | expected fail                        |
| First-order learner | possible          | expected fail     | expected fail                        |
| Graph propagation   | possible          | possible          | design-dependent                     |
| Static HDC binding  | design-dependent  | design-dependent  | possible representational transfer   |
| Benchmark-local candidate | empirical    | empirical         | empirical                            |

These expected outcomes are predictions to confirm, not assumptions to encode into the
harness — but the table is what makes each baseline diagnostic rather than decorative: a
result that violates an "expected fail" cell (e.g. the value table passing P4a) would
indicate a leaky task design, not a surprising baseline capability.

Even after this packet clears Phase 1, it should be reported per-probe with schedule
qualifiers (e.g. "UAL-P1: Demonstrated — replicated across schedules. UAL-P2: [capacity/
explanatory-novelty fields, schedule status]. UAL-P4a: [...]"), rolled up at most to
"initial compositional associative-learning profile demonstrated" — never to "UAL:
present" or "UAL demonstrated."

## Track dependencies between stages

P1–P6 are not six independent pieces of evidence — several plausibly share a mechanism,
a stimulus family, or a behavioral readout, and the final profile must say so explicitly
rather than implying six independent confirmations:

- P2 and P4 may share the same relational-propagation mechanism.
- P5 may depend primarily on the HDC–LTC temporal substrate rather than on associative
  learning per se (already flagged in P5's alternative-explanations note above).
- P6 may reuse the same context encoder as P4.
- Multiple stages could share one action-expression bottleneck — a single MA-001-style
  choke point would fail every stage's behavioral-expression check identically, which
  should be read as one finding, not N.

Each probe's design must declare, before running: shared stimulus family, shared probe
signal, shared target mechanism (Phase 2 only), shared behavioral readout, shared
schedule, and shared training data, relative to every other probe in the set. The final
profile must report both how many stages passed and how many *independent evidence
units* those passes represent — e.g. "four stages passed, representing two independent
evidence units" — following the same dependency-collapsing discipline already applied to
HOT-3/PP-1 in the existing Butlin report.

## See also

- `THE_SUBSTRATE_ROADMAP.md` / `CORE_SUBSTRATE.md` — for how other capability-vs-
  architecture distinctions are already handled in this codebase (Φ vs Ψ, e.g.).
- `crates/domains/symthaea-psych-bench/src/benchmarks/butlin/{indicators,report,ablation,
  qualification_design,qualification_runtime,ae2_empirical_runner}.rs` — the pattern
  this design deliberately follows rather than reinvents; `ae2_empirical_runner.rs` is
  the file that actually closed Gate A (see above).
- `symthaea/docs/BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md` — the Gate A closure evidence:
  the first real, single-seed AE-2 empirical run, plus the broader Butlin
  indicator-repair campaign this document's runner work belongs to.
- `symthaea/docs/SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md` — the Gate B
  specification (P1/P2/P4a fully specified protocols).
- `memory/symthaea_alife_ma001_jul26.md` — source of the learning-vs-expression
  requirement above.
- `memory/symthaea_structural_phi_inverse_response_jul25.md` — prior Butlin-suite work,
  for context on how indicator findings get corrected/re-verified in this codebase.
