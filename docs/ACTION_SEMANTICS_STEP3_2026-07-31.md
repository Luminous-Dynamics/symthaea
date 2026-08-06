# Step 3 — action semantics and perception-to-action sensitivity

**Date:** 2026-07-31
**Harness:** `examples/action_semantics_probe.rs`
**Question:** what does the main cognitive loop treat as an action, and can perception change it?

---

## Part 1 — Action semantics, established by code audit

The live action pipeline is `step_fep_active_inference`
(`src/cognitive_loop/helpers/cycle_extracted.rs:734`), called **ungated, every cycle**
from `src/cognitive_loop/cycle_phase_dynamics/mod.rs:2156`. It performs a genuine
`perceive → select_action → act`.

**Observation space — 4 scalars, all introspective:**
`prediction_error`, `coherence`, `prediction_confidence`, `effective_learning_rate`.

**Action space — 4 discrete internal control adjustments:**

| idx | effect |
|---|---|
| 0 | boost learning rate (scaled by free energy) |
| 1 | reset sensory precision toward 1.0 |
| 2 | boost exploration (larger nudge when surprised) |
| 3 | tighten trust via precision |

**No action touches the world, or even the representation.** Every one adjusts the
loop's own learning hyperparameters. This is self-regulation, not agency.

### Consequence for the intervention-generation problem

An agent whose actions cannot change what it observes next *from the world* cannot
manufacture interventional data, and so cannot learn distinctions that only
interventions reveal. On the default path, no such action exists. This is
**structural, not a tuning issue** — the fix is a design and safety question, not
wiring.

It also sharpens the AGW blocker. The roadmap recorded that
`reasoning_engine.reason()` hardcodes `tool: None` and that the prerequisite was "a
design decision on what the autonomous loop treats as an action." That decision is
already made in code: the action space exists and is four gain adjustments on the
learner. Causal reasoning over it would be reasoning about the loop's own learning
dynamics, not about the world — narrower than the module names suggest.

### Defect found: argument/parameter-name misalignment

`Observation::from_consciousness_state(phi, integration, coherence, attention)`
(`crates/core/symthaea-fep/src/types.rs:37`) is called with
`(prediction_error, coherence, prediction_confidence, effective_lr)`. The arguments
are positionally misaligned with their parameter names. Behaviourally harmless — the
constructor packs them into an opaque `values` vector — but a reader of the call site
would reasonably believe the agent observes Φ and integration. It observes neither.

### Test-coverage gap

`test_step_fep_repeated_calls_stable` sweeps `(pe, coh)` across 50 iterations but
asserts only bounds and finiteness. **No existing test would catch an inert actuator.**

---

## Part 2 — Measurements

### L1: actuator sensitivity (fresh service per grid point)

18 grid points over `PE ∈ {0.0…1.0} × coherence ∈ {0.0, 0.5, 1.0}`:

> **CORRECTION 2026-07-31 (same day).** Two of the three findings originally listed
> here were **artifacts of my own probe design** and are retracted below. The
> retraction is kept visible rather than edited away, because the original text was
> committed and pushed.

**RETRACTED — "1 of 4 distinct actions selected."** Action 3 was chosen at every
point, but that is not a property of the actuator. `ActiveInferenceAgent::select_action`
samples *stochastically* from the softmax using an xorshift64 RNG whose state is a
**fixed constant** at construction (`agent.rs:132`, `0x9E3779B97F4A7C15`). L1 used a
fresh service per grid point, so every point drew the identical first random number and
therefore the identical action. The design guaranteed the result.

**RETRACTED — "selected action is not the probability argmax; candidate real bug."**
There is no bug. Selection is deliberately stochastic sampling (`agent.rs:316-332`),
not argmax, so drawing a lower-probability action is correct behavior. With
p = [0.2808, 0.2192, 0.2808, 0.2192] the cumulative bins are [0.281, 0.500, 0.781,
1.000], and the fixed seed's first draw lands in the last bin every time.

**STANDS — probability response is near-flat.** Movement across the *entire* PE range
0.0 → 1.0 is **0.2808 → 0.2606**, about 2%. The probability vector is deterministic
given the inputs, so the RNG artifact does not touch this. This is the real
actuator-sensitivity result.

**STANDS, and is now root-caused — rank-2 degeneracy.** `p₀ ≡ p₂` and `p₁ ≡ p₃`,
bit-identical in every row. Cause is `generative_model.rs:92`:

```rust
let bias_direction = if action_idx % 2 == 0 { -1 } else { 1 };
```

The per-action transition matrix depends on the action index **only through its
parity**, and `transition_bias` initializes to all zeros for every action. So actions
0 and 2 receive identical transition matrices, as do 1 and 3. Identical transitions →
identical predicted state → identical EFE → identical softmax probability.

**A 4-action space is initialized as 2 distinguishable actions.** The matrices are
learnable (`generative_model.rs:202-268` updates them per action), so this is a
starting condition rather than a permanent identity — but whether the main loop's usage
ever differentiates them is an open question, and nothing in the observed data suggests
it does.

### L2: perception → action channel (120 cycles per regime)

| regime | PE mean | PE sd | action histogram (0/1/2/3) |
|---|---:|---:|---|
| repetitive | 0.2825 | 0.1488 | 30 / 34 / 35 / 21 |
| varied | 0.5005 | 0.1139 | 29 / 35 / 35 / 21 |
| alarming | 0.5163 | 0.1166 | 29 / 35 / 35 / 21 |

- **The channel carries signal**: PE responds strongly to perceptual regime, nearly
  doubling from repetitive to varied (range 0.2338).
- **The action distribution does not.** `varied` and `alarming` are **bit-identical**
  (29/35/35/21) despite differing PE; `repetitive` differs by one count in two bins
  despite PE differing by ~0.22.

## Verdict, corrected

The harness printed *"CHANNEL PRESENT BUT NOT DECISIVE"* and, for L1, *"ACTUATOR
RESPONDS to its inputs."* **Both are too generous, and the fault is mine.** The L1
threshold was `max_spread > 1e-6`, which any nonzero response clears — so it duly
reported "responds" for a grid that picks one action everywhere with 2% probability
movement. This is the same rule-design failure as Step 1.5's half-vs-full recovery
metric: *a threshold set where anything clears it measures nothing.*

**Honest reading: perception does not influence action selection in the main loop.**
PE moves substantially with perceptual regime and the action distribution is invariant
to it — bit-identical across two regimes with different PE. The per-cycle variation in
which action fires is real but is driven by something internal and perception-
independent, since it reproduces almost exactly across regimes.

The threshold has been tightened in the harness (`distinct_actions > 1 ||
max_spread > 0.10`), and argmax-mismatch and degenerate-pair detectors added. The
tables above are from the run *before* that change; only the verdict string differs,
not any measurement.

## The coherent mechanism, after correction

The two levels now tell one story rather than two:

1. The generative model can only represent **2 distinct actions** out of 4 at
   initialization (parity-only `bias_direction`).
2. The EFE computed over those actions is **nearly insensitive to the observation** —
   ~2% probability movement across the full prediction-error range.
3. Therefore the softmax distribution is near-constant regardless of perception.
4. Therefore the sampled action histogram is **perception-invariant**, which is exactly
   what L2 measured: bit-identical histograms for `varied` and `alarming` despite PE
   differing.

Perception does not influence action because the action-value computation barely
depends on the observation, not because anything is disconnected.

## Open items

1. **Rank-2 degeneracy is the highest-value follow-up.** Is parity-only
   `bias_direction` intentional (a deliberate two-directional prior) or an oversight?
   If actions 0/2 and 1/3 are meant to differ, the model cannot express that until
   learning separates them.
2. **Does the main loop's usage ever differentiate the paired actions?** The transition
   matrices are learnable; nothing observed suggests they diverge in practice.
3. **Why is EFE so insensitive to the observation?** ~2% across the entire input range
   is the substantive actuator finding and is not yet explained.
4. Fix the `from_consciousness_state` argument/name misalignment.
5. Add a sensitivity assertion to the FEP test module, which currently cannot catch
   an inert actuator.
6. **Harness fix**: L1 must not report the sampled action as evidence. With a fixed
   construction seed, one draw per fresh service is not a sample of anything. Report the
   probability vector (deterministic) or draw many times per point.

## Scope

This says nothing about whether the loop *should* have world-directed actions. A
self-regulating meta-learner is a coherent design. The finding is that the
surrounding vocabulary — active inference, agent, perceive/act — invites a
world-directed reading that the code does not support, and that perception-to-action
sensitivity is absent even within the introspective action space it does have.
