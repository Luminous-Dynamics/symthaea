# Luminous Dynamics: Simulator Roadmap

*2026-04-17 — grounded in the current state of seven active simulators.*

This is not a wishlist. Every entry either (a) is a direct extension of
existing code, or (b) answers a specific question the current code can't
answer. Aspirational work is explicitly called out.

## Current state (audit)

| Simulator | LOC | Scale | Status | Core question answered |
|---|---|---|---|---|
| [mycelix-multiworld-sim](../mycelix-multiworld-sim/) | ~60K | Civilizational (150yr) | **Shipping** — 779 tests, Phase 1 survey fully closed | "Does consciousness-gated governance survive adversarial pressure across seeds?" |
| [phi-lab](../phi-lab/) | ~360K | Individual consciousness | **Shipping** — 32 tests | "What topology maximizes λ₂ (algebraic connectivity) across 19 networks?" |
| [luminous-sim-core](../crates/luminous-sim-core/) | ~300 | Shared types | **Shipping** | Foundation for everything else |
| [symthaea-{flight,vehicle,auv}](../symthaea/crates/) | ~30K | Embodiment | **WIP** — scenario tests exist | Per-platform dynamics + extreme-case survival |
| [sol-atlas-core](../sol-atlas-core/) | ~4K | Planetary (Earth) | **Stub** — shared types only | — (awaiting USACE pipeline) |
| [spark-engine](../spark-engine/) | ~12K | Physics (LCF fusion) | **WIP** | Gamow tunneling + uncertainty |
| [kosmic-lab](../kosmic-lab/) | ~1.1M (Python) | Deep time | **WIP** | K-index retrodiction across 5,000yr |

**Observation:** the simulators are at different scales and in different
languages. They don't yet share a telemetry contract or a composite
experiment harness. Each lives in its own epistemic bubble.

## Priority A — CLOSED (2026-04-17)

All four A-series items shipped. Summary:

| Item | Status | Commit | Outcome |
|---|---|---|---|
| A1: resilience in report | ✓ | `657baeefd4` | 10-seed resilience mean 0.689 |
| A2: counterfactual A/B | ✓ | `d9cd07af3d` | **Null** (Δ −0.004 ± 0.008) |
| A3: sensitivity sweep (dose) | ✓ | `1ba6175863` | Null is dose-invariant |
| Geometric CVS | ✓ | `a0df345c40` | Null is aggregation-invariant |
| A4: write-up | ✓ | (this commit) | `PHASE2_FINDINGS.md` |

**Headline finding:** Phase 2 machinery does not produce a measurable
civilizational-outcome benefit over the Phi+MYCEL baseline in this
simulator. Diagnosed cause: the attack model operates on governance-
internal state that CVS does not read. See `PHASE2_FINDINGS.md` for
the full story.

Near-term work now shifts to **attack-cascade modeling** — making the
sim capable of detecting defense effects — before any further A/B
variation is worth running.

## Priority A — ORIGINAL PLAN (historical, superseded by closure above)

We just finished Phase 1 survey + Phase 3 DKG + 10-seed validation. The
natural next moves aren't new phases — they're payoff from the machinery
we already built.

### A1. Expose `MycelixResilience` in `CivilizationReport` (≤1 session)

**Why:** the resilience metric is currently only reachable through unit
tests or the metrics example binary. If it's not in the report struct,
scenarios can't query it, and we can't plot it over time.

**Work:** add `mycelix_resilience: Option<MycelixResilience>` to
`CivilizationReport`; populate in `run()` by reading
`MycelixAdversarialTelemetry` accumulated across ticks. Backward-compat
via `#[serde(default)]`.

**Acceptance:** `report.mycelix_resilience.unwrap().mean() >= 0.7` in the
existing A/B tests.

### A2. Counterfactual A/B: does Phase 2 machinery actually help? (1 session)

**Why:** the current A/B tests show "defense works under attack." They
don't show "defense-with-8D outperforms defense-with-scalar-Phi under
attack." That's the *scientific* claim underlying the Phase 2a survey
rationale — and we haven't tested it.

**Work:**
- Add a `PolicyConfig::phase2_disabled()` variant: skips
  `refresh_sovereign_profiles`, uses Phi-only in `tick_governance_full`
  (the old `eligible_fraction` branch), disables `record_violation` hook
  in `sanctions::apply_sanctions`.
- Run 10 seeds × 50yr with Phase 2 enabled vs disabled, under mixed
  attack. Report CVS delta, farming_score delta, survival delta.

**Acceptance (honest):** even if the delta is small or zero, we publish
the number. Negative result is as valuable as positive.

### A3. Sensitivity sweep on the free parameters (1-2 sessions)

**Why:** several constants were picked by analogy to canonical values,
not by optimization:
- `MAX_CORRECTIONS_PER_TICK = 2` (Phase 2c)
- `VIOLATIONS_PER_DEGRADE = 3`, `CORRECTIONS_PER_RESTORE = 10` (Phase 2b)
- The 1/3+1/3+1/3 Phi/MYCEL/8D blend weight in `tick_governance_full`

**Work:** grid search over these parameters at a handful of seeds.
Report which values maximize long-run CVS *under attack*, and which
destabilize it. This is calibration data for production Mycelix —
potentially citable.

**Acceptance:** a markdown summary of the Pareto frontier, with the
handful of dominating parameter sets.

### A4. Research write-up (1-2 sessions, deferred)

**Why:** the Phase 2 + multi-seed findings are a natural companion
piece to the psych-bench paper. Main claim: *"Consciousness-gated
governance with 8D profiles + restorative justice survives adversarial
injection across seeds, with defense signatures distinguishable in
telemetry."*

**Work:** 3-5 page write-up citing empirical numbers already produced.
Drop into `papers/` or `symthaea/papers/`. Honest framing: this is
simulation evidence, not a field trial.

**Blocking:** A1 + A2 first (we need the counterfactual for the claim
to stand up).

## Priority B — Mid-term: cross-simulator coherence

The real unlock isn't "more simulators" — it's "these simulators
answer one unified research question." Currently they can't.

### B1. Unified telemetry schema (2-3 sessions)

**Why:** mycelix-multiworld-sim emits `CivilizationReport`, phi-lab
emits λ₂, symthaea-psych-bench emits a composite z-score. A scenario
that runs all three produces three disjoint JSON blobs, not a
single queryable trace.

**Work:** define `SimulatorReport` in `luminous-sim-core` — a
sum-type or trait with `simulator_id`, `config_hash`, `scalar_metrics`,
`time_series`, `narrative_events`. Migrate the three shipping
simulators first. WIP ones can adopt when ready.

**Acceptance:** a scenario binary that runs all three and emits one
`.jsonl` with aligned timestamps.

### B2. Composite experiment: individual consciousness → civilization outcome (aspirational, 3+ sessions)

**Why:** mycelix-multiworld-sim agents currently have a scalar `phi`
plus 8D sovereign profile. Their actual *neural* consciousness (phi-lab's
topology λ₂) is not modeled. A composite experiment would wire phi-lab
topology to per-agent Phi, then see if topological differences propagate
to civilizational outcomes.

**Concrete proposal:** replace `ConsciousnessState::phi()` for a subset
of agents with a λ₂ draw from phi-lab. Does that change tier
distribution or CVS vs. the current homogeneous formula?

**Blocking:** B1 must land first; without a shared schema the composite
pipeline is glue code forever.

### B3. Sol Atlas as the civilizational instrument (aspirational)

**Why:** memory notes "Sol Atlas project — civilizational instrument:
Bevy holographic + Leptos WebGL globe" and mycelix-multiworld-sim is
the intended backing simulator. The link hasn't been built.

**Work:** expose mycelix-multiworld-sim state via a Sol Atlas-readable
interface (WebSocket? embedded Leptos sim? shared SQLite?). This is
weeks of product work, not a hack session — flagging it honestly.

## Priority C — Parked (not recommending yet)

- **kosmic-lab Rust bridge**: memory says kosmic-lab has a "planned
  Rust integration." No demonstrated need yet. Wait for a research
  question that requires it.
- **spark-engine sim harness**: physics-first, not civilization-first.
  Separate track.
- **Symthaea robotics scenario expansion**: these sims are embodiment-
  focused, not civilizational. Own roadmap lives in the symthaea
  robotics docs.

## Sequencing recommendation

Single-track, highest-payoff first:

1. **A1** — expose resilience in report (≤ 1 session, unlocks everything downstream)
2. **A2** — counterfactual A/B (1 session, answers the scientific claim)
3. **A3** — sensitivity sweep (1-2 sessions, produces citable data)
4. **A4** — write-up (if user wants a paper)
5. **B1** — unified telemetry schema (if cross-simulator work becomes priority)

A1 is ~2 hours. After A1 every scenario can observe the Phase 2
machinery, and the value of A2/A3 grows. B1 is where real ambition
starts — but only worth the effort after the local consolidation
produces something worth comparing.

## Honest gotchas

- The mycelix-multiworld-sim attribution-bundling incidents from this
  session (two commits where staged work was swept into concurrent
  session commits) argue for **always using a worktree for simulator
  work**, not just for finishing phases.
- The 10-seed sweep showed seed 13 as a case where TierBuyers
  actually outperform baseline (+1.1 SAP delta), yet civilization still
  survived. The defense invariants we rely on (survival, farming
  rejection rate) are more robust than any single-metric improvement.
  Keep reporting multiple metrics, not a single "defense works" pass/fail.
- Phase 2 results should be labeled "simulation evidence" in any write-
  up. The real DHT, the real identity bridge, the real demurrage ledger
  aren't in the loop yet.

---

*This roadmap assumes a single-session-at-a-time cadence. If priorities
shift (e.g., Sol Atlas goes to production, a grant deadline appears),
the sequencing above should be revisited, not followed mechanically.*
