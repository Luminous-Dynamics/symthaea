# EFE Fleet Dispatch: Pre-registration and Precondition Failure

**Status: HALTED at the precondition check. Do not build the simulator.**
**Date: 2026-07-31 · Owner: tstoltz**

---

## Summary

The proposed gate was: *does Expected-Free-Energy-driven idle-vehicle repositioning beat
min-cost-flow-with-a-historical-demand-prior on a NYC TLC replay?* Budget 18 developer-days.

It never got to the simulator. A static precondition check — read the EFE implementation and
confirm its terms can vary across candidate actions — **failed against the code**, in about two
hours. Two independent reasons also emerged, each sufficient on its own, for not running the study
even with that defect fixed.

This is the intended behaviour of a pre-registration. The gate returned a verdict far cheaper than
planned, and the verdict is negative.

**Consequence, pre-committed in the design and now triggered:** the deliverable is a fleet
coordination layer — demand prior + assignment + rebalancing LP — **and Symthaea is not in it.**

---

## Blocker 1 (fatal, verified): the terms that make it active inference are structurally inert

`ExpectedFreeEnergyComputer::compute` (`crates/core/symthaea-fep/src/free_energy.rs:399-440`)
returns `total = pragmatic_weight·pragmatic + epistemic_weight·epistemic − novelty_weight·novelty`.
The pragmatic term is expected cost. **Epistemic and novelty are what distinguish this from a
weighted greedy heuristic. Neither varies across actions.**

### The epistemic term is bit-identical for every action

Three facts compose:

1. `GenerativeModel::predict_next_state` (`generative_model.rs:130-158`) makes only `next_mean`
   action-dependent. The precision update contains **no action term**:
   ```rust
   let next_precision: Vec<f64> = state.precision.iter()
       .map(|p| (p * self.transition_precision) / (p + self.transition_precision))
       .collect();
   ```
   `transition_precision` is a single scalar field on the model (`generative_model.rs:52`).
2. `HiddenState::entropy` (`types.rs:102-108`) is a function of `precision` and `mean.len()` only:
   ```rust
   let log_det: f64 = self.precision.iter().map(|p| -p.max(0.001).ln()).sum();
   0.5 * (dim + dim * (2.0 * PI).ln() + log_det)
   ```
   The *values* in `mean` never enter.
3. `compute_epistemic_value` (`free_energy.rs:462-474`) is `predicted_entropy − current_entropy`.

Action affects only `mean`. Entropy ignores `mean`. Therefore the epistemic value is identical for
every candidate — not approximately, exactly.

**Supplying a custom generative model does not fix this.** `predict_next_state` is an inherent
method on the concrete `GenerativeModel` struct, not a trait method. There is no override point.
Fixing it means forking or rewriting the crate's belief update — at which point the honest
provenance claim ("we reused ~20 lines of the repo's EFE arithmetic") no longer holds, and the
experiment is testing new code rather than Symthaea's.

### The novelty term is also near-constant

`compute` takes `&mut self` and unconditionally records **every action it scores**
(`free_energy.rs:422-426`). Selection is `argmin` over all candidates, so every candidate is
scored and pushed each decision epoch. Counts across candidates stay near-identical by
construction, so `compute_novelty = 1/(1+count)` (`free_energy.rs:476-479`) is near-constant.
The 100-entry buffer holds ~8 epochs at 12 candidates/epoch, so it has almost no memory either.

Two side-effects worth recording independently:

- **EFE scoring is impure and enumeration-order-dependent.** The score assigned to candidate 12
  depends on candidates 1–11 having been scored first. Deterministic, so a determinism harness
  passes while the arm is silently path-dependent. This is the same defect class as the standing
  rule that `HdcLtcBridge::train_step`/`predict_forward` must stay pure w.r.t. live state
  (MASTER_ROADMAP §Signal integrity) — that rule should extend here.
- A prior audit attributed the observed "2 distinct EFE values across 263 actions" solely to
  `action_idx % 2` in the transition construction (`generative_model.rs:86-101`, confirmed real).
  That diagnosis was **incomplete**: it explains degeneracy in the *pragmatic* term only. The
  epistemic term was never action-dependent at all, at any action count.

### What this means

With both terms inert, the "EFE arm" reduces to `pragmatic_weight × expected cost` — a greedy cost
minimiser with active-inference vocabulary attached. Running an 18-day study to discover that a
greedy cost minimiser ties a min-cost-flow LP would have been an expensive way to learn nothing.

---

## Blocker 2 (independent): the headroom being measured is ~1%, and the design gated on ≥4%

The primary estimand was ρ = fraction of the min-cost-flow → clairvoyant-oracle headroom captured,
with a proceed gate of h ≥ 4%.

[arXiv:2604.03883](https://arxiv.org/abs/2604.03883) (Kumar & Tiwari, *Regime-Calibrated Fleet
Repositioning with a Spatial Queue-Regret Decomposition*, May 2026) measures **exactly this
comparison** — a Zhang-&-Pavone share-target transportation LP against a clairvoyant oracle MPC, on
NYC TLC replay demand in a common simulator:

| Controller | Mean wait |
|---|---|
| Oracle MPC (clairvoyant) | 91.3 s |
| Share-target transportation LP | 92.2 s |
| Scenario chance-MPC | 92.2 s |
| GPR chance-MPC-lite | 94.4 s |
| Wen-style rebalancing | 100.1 s |

h = (92.2 − 91.3)/92.2 = **0.98%** — four times *below* the design's own proceed gate. Tuning the
baseline can only shrink it further.

**And my original ≥10% threshold was wrong in the other direction, for a reason worth recording.**
The 3.5–4.1% figure (82.3 vs 85.3 vs 85.8 s) that first looked like the relevant effect ceiling is
from a *different experiment in the same abstract*: it compares **demand-prior/retrieval variants**
feeding one fixed controller. It is the size of the **prior** lever, not the **controller** lever.
In that paper's own numbers the prior lever is ~4.2× the perfect-information controller lever.

That is the substantive finding underneath the arithmetic: **in fleet repositioning, the demand
prior is worth several times more than the controller.** A study that freezes a shared prior across
arms and varies only the controller is, by construction, measuring the small lever.

---

## Blocker 3 (independent): the estimator and the budget were both unsound

Recorded so they are not rediscovered if this is ever revived.

- **ρ̂ was a mean of ratios over a near-zero, sign-indefinite denominator.** At h ≈ 1%, the per-date
  denominator is ~0.9 s on a ~92 s scale. Nothing guarantees per-date oracle dominance on held-out
  dates; any date with h_d ≤ 0 makes ρ_d unbounded and sign-flipped, and bootstrap intervals on
  that statistic have no valid coverage. It can manufacture a large positive ρ̂ from pure noise.
  A ratio of means with a Fieller interval is the minimum repair.
- **Propagated variance made the study structurally incapable of concluding.** σ_ρ ≈ σ_τ/h; with a
  modal σ_τ ≈ 0.10 and h ≈ 0.01, σ̂_ρ ≈ 10 — an order of magnitude past the design's own re-sizing
  cap, which downgrades everything to INCONCLUSIVE. Both DEMONSTRATED *and* EQUIVALENT become
  unreachable simultaneously, so even the "useful tie" outcome was unavailable.
- **Tuning compute was under-counted ~19×.** The plan budgeted ~4,650 evaluation episodes (~2 h on
  12 cores) but omitted tuning entirely: 200 trials × 5 arms × 90 tuning episodes = 90,000
  episodes ≈ 62 h. On a host that routinely sits at load 26–62 with 15+ concurrent sessions, that
  is a multi-day job absent from an 18-day plan.
- **A portfolio confound made the primary comparison non-identifying.** The EFE arm's candidate set
  included the baseline's own share-target plan, so `argmin` over a superset containing the
  baseline's action cannot do worse in-model. Any win would conflate a portfolio effect with active
  inference.
- **Checkpoint 0 gated on an unfrozen simulator.** The proceed decision depended on h measured in a
  simulator whose internals (demand volatility, warm-up, travel noise) were still tunable — a
  researcher-controlled knob on the primary estimand's denominator, evaluated before freeze.

---

## What would have to be true to revive this

All four, in order. Any one failing halts again.

1. **Make the epistemic term action-dependent, and prove it.** Requires a belief update where
   predicted precision depends on the action — i.e. replacing `predict_next_state`, not configuring
   it. Gate: `SD(epistemic) / SD(pragmatic) ≥ 0.02` across candidates on ≥80% of epochs, measured on
   *unweighted* term values (normalising by weighted `EFE_total` would let a tuned `pragmatic_weight`
   fail the check mechanically).
2. **Make `compute` pure.** Snapshot/restore `action_history` around candidate enumeration, or take
   `&self` and return the history delta. Add a test that scoring candidates in reverse order yields
   identical scores.
3. **Add a novelty manipulation check.** Gate on across-candidate variation in the novelty term
   specifically. Without it, a positive result would be reported as "attributable to epistemic and
   novelty" when novelty contributed nothing.
4. **Find a question where the lever is large.** Given h ≈ 1% for controller-vs-oracle, the
   controller is the wrong lever. If active inference has a real edge it is far more likely to be in
   *demand prediction under regime shift* — the prior, which the literature says is worth ~4× more —
   than in the repositioning controller. That is a different study with a different estimand, and it
   should be designed from scratch rather than patched onto this one.

---

## Assets that survive regardless

Found while scoping, useful independent of any of this:

- A correct rectangular Hungarian/Jonker-Volgenant solver with dummy columns and an `INVALID_COST`
  sentinel, mis-filed at `crates/domains/symthaea-vision-manifold/src/manifold.rs:5631`. Worth
  lifting to a shared crate; a prior pass wrongly declared no assignment algorithm existed anywhere.
- `hdc/grid_encoder.rs:35-41` has **no spatial locality** — `ContinuousHV::random` per index, so
  position 5 and 6 are exactly as dissimilar as 5 and 50. Any spatial HDC work is currently
  meaningless. The correct construction exists for *time* at `hdc/temporal_encoder.rs:130-172`
  (integer bounded frequencies, sin/cos pairs, max freq √dim/2) and needs porting to 2D.
- **NYC TLC FHVHV data trap, undocumented anywhere public:** `on_scene_datetime` is 99.6–99.8% null
  for Lyft (HV0005) before 2025-04 and 0% null after; Uber (HV0003) is 0% null throughout. The
  official data dictionary (dated 2025-03-18) still calls the field "Accessible Vehicles-only",
  which is false in both directions. **Filtering on non-null `on_scene` over any pre-April-2025
  window silently yields an Uber-only dataset**, dropping ~27% of rows in perfect correlation with
  operator. Any future TLC work here must start at 2025-04.

---

## What this says about the method

The pre-registration did its job by failing early, and the specific way it failed is the recurring
pattern in this repo: **capability and vocabulary are indistinguishable from the inside.** The crate
is named `symthaea-fep`, the struct is `ExpectedFreeEnergyComputer`, the fields are `epistemic` and
`novelty`, there are 174 passing tests — and two of the three terms cannot vary. Nothing in the type
system, the test suite, or the naming would tell you.

The generalisable defence is the one used here: before building the harness, write down the
mechanism the hypothesis requires, then check the mechanism exists in the code. That check cost two
hours and saved eighteen days.

### The findings are now measured, not argued

`crates/core/symthaea-fep/tests/efe_term_variation.rs` (4 characterization tests + 1 ignored
aspirational gate, all green 2026-07-31) executes what this document originally derived statically:

| Test | Asserts |
|---|---|
| `characterize_epistemic_term_is_action_invariant` | epistemic value is bit-identical across all 12 candidates |
| `characterize_pragmatic_term_varies_only_by_action_parity` | exactly 2 distinct pragmatic values across 12 actions |
| `characterize_novelty_term_flattens_after_one_enumeration` | novelty is uniform across candidates after one epoch |
| `characterize_novelty_counts_considered_not_taken` | a never-taken candidate's novelty decays to 1/6 after 5 scorings |
| `aspirational_epistemic_term_varies_across_actions` | `#[ignore]`d — the SD(epistemic)/SD(pragmatic) ≥ 0.02 revival gate |

These assert the **defective** current behaviour deliberately. If one fails, someone has fixed the
underlying term — delete the characterization, promote the aspirational test, and revise this
document. (Verified not silently dead: `symthaea-fep/Cargo.toml` has no `autotests = false` and no
`[[test]]` block, so autodiscovery runs the file. This repo has 166 test files that fail that check.)

**One claim in this document's first draft was wrong, and writing the test is what caught it.** It
asserted that EFE scoring is *enumeration-order-dependent within an epoch*. It is not:
`compute_novelty` counts only occurrences of the queried action, and scoring other candidates does
not change that count. The probe asserting order-dependence failed on first run. The real defect is
narrower and sharper — novelty counts actions **considered**, not actions **taken**, so a candidate
that is evaluated and rejected every epoch becomes progressively less "novel" without ever being
executed, exactly inverting the term's purpose. The impurity is real; the order-dependence was not.

That is the second time in this document that a plausible reading of the code was wrong in a way
only execution caught. Treat every remaining static claim here accordingly.

---

## Related

- `mycelix-workspace/mycelix-commons/zomes/DORMANT_TRANSPORT.md` — the Mycelix side, frozen 2026-07-30.
- `docs/KEYSTONE_AB_PROTOCOL_2026-07-17.md` — the house pre-registration format this follows.
- MASTER_ROADMAP §Signal integrity — the purity rule that should extend to `ExpectedFreeEnergyComputer`.
