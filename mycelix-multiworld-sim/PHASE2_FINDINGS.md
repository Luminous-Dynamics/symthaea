# Phase 2 Findings — Mycelix Multiworld Simulator

*2026-04-17. Consolidated write-up of the Phase 1 survey → Phase 2a/b/c
implementation → A1–A3 + geometric-CVS validation arc.*

## Summary

**Claim tested:** "Consciousness-gated governance with 8D sovereign profiles
+ restorative justice + correction rate-limits produces better
civilizational outcomes under adversarial pressure than a simpler
Phi-only baseline."

**Result:** **Null.** Across 10 seeds × 50 years × 4 attacker doses × 2
aggregation methods, the Phase 2 defenses produce a mean CVS delta that is
indistinguishable from zero (arithmetic −0.004 ± 0.008; geometric
−0.007 ± 0.007). Phase 2 wins 1–3 out of 10 seeds.

**Not a bug in defense code.** Phase 2 demonstrably contains its target
attacks: CorrectionFarmer rejection rate is 61% across all seeds;
TierBuyers dilute to negative SAP delta by year 50; GuildColluders'
mycel-boost scales correctly with colluder count.

**Root cause:** the attack model as simulated changes *adversary state*
(SAP balances, justice counters, mycel scores) but does not propagate into
any of the five inputs that compose CVS (genetic diversity, economic
sustainability, harmony scores, oppression index, collective Phi). Both
arithmetic and geometric aggregation are insensitive because the inputs
themselves don't move differentially between conditions.

**Path forward:** model attacks that cascade into CVS dependencies, not
just adversary state. See [Recommendations](#recommendations).

---

## What Phase 2 built

Phase 2 closed the [MYCELIX_SIM_PHASE1_SURVEY](../MYCELIX_SIM_PHASE1_SURVEY.md)
gaps 1, 2, and 5. ~2,500 LOC across three commits plus wiring and
validation:

| Phase | What | Commit | Tests |
|---|---|---|---|
| 2a | 8D Sovereign Profile + CivAgent field + per-tick refresh + Monte Carlo init at 5 founder sites | `e300f724cc` | 11 |
| 2b | `RestorativeJustice` — violations, corrections, tier_penalty with cooldown. Sanctions record violations; per-tick corrections pass | `03a8f13e3b` | 13 |
| 2c | 5 Mycelix attack strategies + `MAX_CORRECTIONS_PER_TICK = 2` rate limit + `correction_farming_score` detector + `MycelixResilience` metric | `8b15bd92aa` | 18 |
| wiring | Per-tick `apply_mycelix_adversarial_tick` + A/B integration tests | `c50c72a12d` | 4 |
| bypass | CrossClusterAmplifier gate-lowering + 50yr equilibrium test | (bundled) | 2 |
| multi-seed | 10-seed × 50yr validation | `0951d95ce0` | 4 |
| Phase 3 | DKG cost model for committee-size calibration | `4bd5a34230` | 15 |

The code does what the Mycelix production spec describes. The
implementation is correct. All tests green.

---

## The experiments (SIMULATOR_ROADMAP A1–A3)

### A1 — Expose resilience in the report (`657baeefd4`)

Added `CivilizationReport.mycelix_resilience: Option<MycelixResilience>`
populated by `red_team::compute_resilience_from_worlds` at sim end.
Empirical (10 seeds × 50yr):

- Resilience mean **0.689**, range 0.461–0.822
- Seed 271 low outlier (0.461) — TierBuyers survived dilution
- All other seeds ≥ 0.55

A1 was just observability plumbing, but the per-seed variance surfaced
at this point was the first hint that per-surface attack success isn't
uniform.

### A2 — Counterfactual: does Phase 2 outperform baseline? (`d9cd07af3d`)

Added `PolicyConfig.phase2_enabled` flag gating the four code paths the
machinery introduces. With the flag off, the simulator reverts to the
pre-2a governance (scalar Phi + MYCEL 50/50 blend, no refresh, no
restorative justice, no correction rate limit). Attackers engage
identically in both conditions.

10 seeds × 50 years:

| metric | phase2=on | phase2=off | delta |
|---|---|---|---|
| mean CVS | 0.720 | 0.724 | **−0.004 ± 0.008** |
| Phase 2 wins | 3/10 | 7/10 | — |
| resilience | 0.689 | 0.683 | +0.006 |

First surprise: Phase 2 actually *loses* in 7 seeds, though the magnitude
is within one σ of zero. Resilience is nearly identical — the attacks
don't succeed more against baseline than against Phase 2.

### A3 — Is the null dose-specific? (`1ba6175863`)

A2 used 3 attackers per strategy (15 total). Maybe that dose is too low
to stress the defense. Sweep 3 / 10 / 20 / 30 per strategy, 3 seeds × 50yr
each:

| dose | cvs_on | cvs_off | delta | survival |
|---|---|---|---|---|
| 3 | 0.722 | 0.723 | −0.000 | 6/6 |
| 10 | 0.714 | 0.724 | −0.010 | 6/6 |
| 20 | 0.724 | 0.726 | −0.002 | 6/6 |
| 30 | 0.726 | 0.729 | −0.003 | 6/6 |

Phase 2 is slightly negative at every dose. All 24 runs survived 50
years. Dose isn't the problem.

### Geometric CVS — is the null an aggregation artifact? (`a0df345c40`)

Current CVS is an arithmetic mean of 5 components weighted 0.2 each
(`0.2·genetic + 0.2·economic + 0.2·harmonies + 0.2·(1−oppression) + 0.2·phi`).
Arithmetic means are fully substitutable — a civilization with
`oppression = 1.0` can still score 0.8 if the other four are maxed.
Kosmic-lab's historical K-index uses geometric mean for this reason:
"weakest link" aggregation where one collapsed dimension crushes the
whole score.

Added `EpochManager::compute_cvs_geometric` alongside arithmetic.
Re-ran A2 with both variants side-by-side:

| metric | Arithmetic | Geometric |
|---|---|---|
| Phase 2 delta | −0.004 ± 0.008 | **−0.007 ± 0.007** |
| Phase 2 wins | 3/10 | **1/10 (seed 137 only)** |
| absolute scale | 0.71–0.73 | 0.59–0.61 |

The geometric variant makes the null *stronger*, not weaker. If
arithmetic CVS had been hiding defense signal by letting strong
dimensions compensate for attacker-damaged ones, geometric CVS would
have revealed it. It didn't. Both aggregations agree: Phase 2 doesn't
move the inputs differentially between conditions.

---

## Diagnosis

The five CVS inputs come from independent pipelines:

| input | source |
|---|---|
| genetic_diversity | `population.rs` inheritance + reproduction |
| economic_sustainability | `economy.rs` sector production vs consumption |
| harmony_scores | `harmony.rs` Eight-Harmony tracker |
| max_oppression | `governance.rs::compute_oppression` from tier distribution |
| collective_phi | `consciousness.rs::tick_consciousness_all_worlds` |

The five Mycelix attack strategies mutate *adversary state*:

| strategy | mutates |
|---|---|
| TierBuyer | `agent.sap_balance` |
| DemurrageEvader | `agent.sap_balance` (churn, tracked only) |
| CorrectionFarmer | `agent.justice.corrections` / `rejected_corrections` |
| CrossClusterAmplifier | reduces per-dim civic floor in `civic_fraction_meeting` |
| GuildColluder | `agent.mycel_score` |

None of these mutations reach the CVS input pipelines. SAP balance
doesn't affect genetic diversity. Justice counters don't affect economic
sustainability. MYCEL score doesn't enter harmony scores. Civic
eligibility fraction is used *within* governance but is averaged out of
`oppression_index` (which reads only the tier distribution, not
eligibility).

**The sim has two disjoint state spaces:**

1. A **civilizational-outcomes** space that CVS reads (demographics,
   economy, harmony, oppression, phi).
2. A **governance-internals** space that Phase 2 defends (profiles,
   justice records, vote eligibility, MYCEL reputation).

The defenses work in space 2. CVS reads from space 1. There is no
pathway for space-2 changes to cascade into space-1 state. Any
aggregation method — arithmetic, geometric, max, min — will report
zero effect because the inputs are literally the same.

---

## What this does NOT say

- **Phase 2 defenses are broken.** They are not. They empirically
  contain their target attacks (farming rejection 61%, TierBuyer
  dilution, bypass gating correctly applied). The resilience metric
  reports high defense quality in both conditions.
- **Phase 2 is wasted work.** The code is a faithful simulation of
  production Mycelix's spec. In production, the defenses protect
  against real attacks (cryptographic integrity, audit-log legal
  standing, operator trust) — none of which enter CVS.
- **Geometric CVS is wrong.** It is strictly more stringent than
  arithmetic and captures an intuitively-correct property ("a totally
  oppressive society is not viable"). Future sim work should probably
  use it as the primary metric. It just doesn't rescue this
  particular null.

---

## Recommendations

### Near term (1–3 sessions)

**Wire attack consequences into CVS input pipelines.** Examples:

1. **TierBuyer → oppression**: if adversarial agents control > X% of
   voting-tier seats, increase `oppression_index` proportionally. The
   defense of Phase 2 is exactly preventing this control, so damage
   would differentiate conditions.
2. **CorrectionFarmer → harmony decay**: successful corrections from
   farmers (credited, not rejected) corrupt the "Radical Translucency"
   or "Emergent Ethical Wisdom" harmony score. Farming rejection would
   then directly protect harmony scores.
3. **GuildColluder → economic sustainability**: if colluders capture
   vote-weighted proposal outcomes, bias resource-allocation votes
   toward their faction, reducing economic diversity.
4. **Adversarial inheritance**: in `population.rs` birth processing,
   children of adversarial parents inherit adversarial status with some
   probability. Currently adversaries are one-shot injected and dilute
   naturally over generations.

Any one of these would make the sim capable of detecting Phase 2's
contribution. All four would make it a credible governance stress-test.

### Mid term (SIMULATOR_ROADMAP B)

Only pursue after the near-term cascade work is done. Unified
`SimulatorReport` schema in `luminous-sim-core` enables composite
experiments but is blocked on having something worth composing.

### Do not pursue

- More aggregation-method sweeps. Geometric already showed the null is
  aggregation-invariant.
- More attacker-dose sweeps. A3 already showed dose-invariant null.
- More A/B variations without cascade wiring. The effect sizes will
  remain within noise until the attacks actually damage CVS inputs.

---

## Honest framing for publication

If this work were written up externally, the claim is:

> *"Implementing Mycelix's governance defenses (8D sovereign profile,
> restorative justice, correction rate-limiting) as a 2,500-LOC
> simulation module did not produce a measurable civilizational-outcome
> benefit over a simpler baseline across 10 seeds × 50 years × 4
> attacker doses × 2 CVS aggregation methods. The defenses empirically
> contain their target attacks (farming rejection 61%, TierBuyer
> dilution to negative SAP delta); they just don't shift outcome
> metrics. The diagnosed cause is that the simulator's attack model
> and its CVS computation operate on disjoint state spaces — the
> defenses protect governance internals that CVS does not read. Future
> work modeling attack cascades into CVS inputs would be required to
> test the defenses' civilizational value."*

That's a negative result worth publishing, and a clear research
direction. Both are more valuable than a forced positive result would
be.

---

## Artifacts

- **Code:** `mycelix-multiworld-sim/src/{sovereign_profile, sub_passport, red_team, dkg}.rs`, `mycelix-multiworld-sim/tests/mycelix_{attack_ab, red_team, attack_metrics}.rs`
- **Examples:** `examples/{mycelix_attack_metrics, mycelix_multiseed_sweep, phase2_counterfactual_sweep, dose_sensitivity_sweep, dkg_committee_tradeoff}.rs`
- **Tests:** 716+ lib + 24 integration, all green
- **Survey:** `../MYCELIX_SIM_PHASE1_SURVEY.md`
- **Roadmap:** `../SIMULATOR_ROADMAP.md`
- **This document:** `PHASE2_FINDINGS.md`
