# Step 4 — temporal credit assignment: verified premise and design

**Date:** 2026-07-31
**Status:** premise verified at source; design only, nothing implemented.

---

## Part 1 — The premise, verified

I had been asserting all through this arc that temporal credit assignment is
structurally absent, on the strength of a roadmap note. Given that this same arc
produced three retractions from exactly that habit — believing a claim before
checking the mechanism behind it — the premise was read at source before building on
it.

**It holds.** In `crates/domains/symthaea-broca/src/controller.rs:1069`
(`backward_step`):

- The composed input is **reconstructed** (`thought_hv.bind(&token_emb).bind(&pos_emb)`),
  not read from a stored history.
- It reads `self.network.output()` and `neuron.state()` — the network's **current**
  state, not an activation saved from step *t*.
- The update target is local: `target = neuron.state() − d_per_neuron`.
- There is no loop over time and no gradient carried from position *t* to *t−1*.
- `bptt_window` bounds only how many tokens are trained per pair
  (`training.rs:1007`).

This is **truncated teacher forcing with a local target rule**, not BPTT. The naming
implied a sequence-learning capacity the trainer does not have — consistent with the
Keystone A/B conclusion that training signal, not architecture, is the binding
constraint.

### Correction to the existing in-source note

`training.rs` previously stated that cross-layer credit "uses a hand-tuned
`gradient_attenuation.powi(depth)` decay, **not a chain rule**." That overstates.
`neuron.backward()` returns a real `d_input`, which is accumulated and used to build
the target for the next layer down — a genuine backward error signal *does* flow
across layers. `gradient_attenuation` is damping applied on top of that chain, to both
the propagated signal and the per-layer learning rate.

**The distinction is load-bearing for the fix:**

| | status |
|---|---|
| **Depth** credit | exists (target-propagation-like), heuristically damped |
| **Time** credit | **absent entirely** |

Only the second is the blocker. A fix should not disturb the first.

---

## Part 2 — Design

### The architectural constraint that decides this

The trainer is built around **no stored state history**. `backward_step` deliberately
reconstructs its input and reads current state. Any mechanism requiring an unrolled
activation tape is therefore not a modification but a rewrite of the training path's
central assumption.

That rules out true BPTT as the first move, and points at eligibility traces.

### Options, against this architecture

**A. True truncated BPTT** — store activations for the window, unroll, chain gradients
back through time.
*Fits:* exact credit over the window.
*Costs:* requires a state tape of `window × layers × 16,384` dims; contradicts the
design's explicit no-history property; largest blast radius on the hottest path.
*Verdict:* correct destination, wrong first step.

**B. Eligibility traces / e-prop** — each neuron maintains an O(1) running trace of
its own recent input-output sensitivity; the local error signal at time *t* multiplies
the trace to credit earlier activity.
*Fits:* no stored history (trace is per-neuron state); local per-neuron update, which
is exactly the shape `neuron.backward(input, target, dt)` already has; O(1) memory.
*Costs:* approximate; traces need a decay constant, which is one more tuned scalar —
and this codebase has a documented weakness for hand-tuned scalars going unvalidated.
*Verdict:* **recommended first implementation.** It is the only option that respects
the existing architecture rather than replacing it.

**C. Explicit local predictive objective** — train each step to predict its own next
input, making temporal structure a per-step target rather than a credit-assignment
problem.
*Fits:* no history, no traces, simplest.
*Costs:* changes what is being optimized, so it is not a fix to credit assignment but
a different objective. Would need its own justification.
*Verdict:* worth considering as a complement, not a substitute.

### Preconditions before implementing any of them

1. **A metric that can detect success.** The Keystone A/B work established the loop
   "does not learn sequences AT ALL in any configuration," and the HDC-LTC ablation
   later found no demonstrated predictive superiority over an EMA baseline. Adding
   credit assignment without a validated sequence task would produce another
   uninterpretable result. `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md` exists, is frozen,
   and includes an information-theoretic task validator — use it, and run its validator
   *first*. The predecessor benchmark failed because its corpus was exactly periodic,
   making next-item prediction solvable with zero history.
2. **A negative control that is mechanically guaranteed.** The `Static` arm reading
   exactly 0.0000 on swap-sensitivity is the model to copy.
3. **Baselines that are hard to beat.** An EMA bank, not a strawman — EMA already
   showed stronger regime separation than the HDC-LTC coupling.
4. **Do not disturb depth credit.** It works; the change is orthogonal and should be
   shown not to regress it.

### Pre-registration sketch

Any implementation should fix, before running: the task (from V2's validated ladder),
the baseline set, the success threshold, and an explicit statement of what result
would count as *failure*. Given this arc's history — three retractions, two thresholds
set where anything cleared them — the threshold discipline matters more than the
mechanism choice.

## What is not claimed

Nothing here says adding temporal credit assignment will make the loop learn
sequences. The Keystone result is consistent with training signal being the binding
constraint, but that is a hypothesis this work would test, not a prediction it
assumes.
