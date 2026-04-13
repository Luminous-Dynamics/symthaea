# Symthaea Experimental Pre-Registrations

**Purpose**: Freeze hypotheses, predicted directions, predicted magnitudes,
and pre-specified statistical tests **before** any experimental harness is
implemented or any data is collected. This is the discipline that keeps the
"publishable null result" promise (Phases III + IV of the Holon-Soma
roadmap) honest — without pre-registration, post-hoc rationalization would
let any finding be claimed as confirmation.

**Format**: Each pre-registration is one section with eight required
fields. Once written, a pre-registration is **immutable** (commits that
modify a frozen pre-registration must rename the section to add a
`-revised-N` suffix and explain the revision in a `## Revision history`
appendix at the bottom — original text never overwritten).

**Provenance**: Each pre-registration must be committed to git **before**
its corresponding experimental harness exists. Use `git log
papers/preregistration.md` to verify the timestamp ordering against the
harness commit.

---

## Required fields

Every pre-registration MUST have these eight fields:

1. **Hypothesis** (one declarative sentence, falsifiable)
2. **Independent variable(s)** (what you'll vary, with discrete levels or range)
3. **Dependent variable(s)** (what you'll measure, with units and how)
4. **Predicted direction** (which way each DV moves with each IV — sign of effect)
5. **Predicted magnitude** (effect size, with units and tolerance)
6. **Pre-specified statistical test** (named test + alpha + minimum n)
7. **Pre-specified failure conditions** (what observed values would falsify the hypothesis — be explicit)
8. **What we will publish** (commit to publishing all four quadrants: confirms direction + magnitude / confirms direction only / confirms magnitude only / null)

---

## PR-001: Phase III — Bandwidth-Quality-Φ Sweep

**Status**: FROZEN 2026-04-13. Harness does not yet exist (will land in
Phase III after Phase I.A.5 + I.A.2 + I.B complete).

**Source**: `plans/shiny-wibbling-quail.md` Phase III (lines 297-326).

### 1. Hypothesis

Consciousness-gated frame encoding (the `consciousness_level` parameter
flowing through `SomaRdpServer.tick()` into `AdaptiveQualityEngine` tier
selection) reduces transport bandwidth without proportional cost to task
success rate.

Restated as a falsifiable statement: at simulated Φ = 0.65 vs Φ = 0.35 on
the same task trace, mean bandwidth is at least 30% lower **and** task
success rate is within 5 percentage points.

### 2. Independent variable

Simulated Φ (consciousness level) ∈ {0.15, 0.25, 0.35, 0.50, 0.65, 0.85}
— six discrete levels spanning the Red/Orange/Yellow/Green tier
boundaries from the existing `MotorSafetyLevel` model.

### 3. Dependent variables

- **Bandwidth** (bytes per second through the sealed binary wire,
  measured at `HolonHttpState.rdp_outbound` drain rate, sampled at
  100 ms intervals over the task duration)
- **Task success rate** (fraction of trials where the task completes
  according to the predefined `phone_agent` task-completion checker)
- **Prediction error** (`VisionTelemetry.prediction_error`, mean and
  max over the trial)
- **Working memory saturation** (fraction of cycles where
  `working_memory_load >= 4`)
- **End-to-end task latency** (wall time from `PhoneAction::Tap` issue
  to detected screen transition, milliseconds)

### 4. Predicted direction

| DV | Direction with increasing Φ |
|----|----------------------------|
| Bandwidth | **Decreases** (higher Φ → tighter `AdaptiveQualityEngine` tier → fewer/smaller patches) |
| Task success | **Stays flat** above Φ ≥ 0.35; drops sharply below |
| Prediction error | **Decreases** smoothly (more accurate perception) |
| WM saturation | **Increases** (more attention → more tracked objects) |
| Latency | **Stays flat** within ±20% across the Φ range |

### 5. Predicted magnitude

- **Bandwidth at Φ=0.65 vs Φ=0.35**: ≥30% reduction. Effect size: Cohen's
  d ≥ 0.5 on paired bandwidth measurements over the same task trace.
- **Task success at Φ ≥ 0.35**: ≥95% (95% CI lower bound).
- **Task success "knee"**: somewhere in [0.20, 0.30] — below this point,
  success collapses to <50%.
- **PE at Φ=0.65 vs Φ=0.35**: 10-20% lower (smaller effect, noisier).
- **Latency variance across Φ levels**: <20% coefficient of variation.

### 6. Pre-specified statistical test

- **Primary**: paired t-test on per-task bandwidth measurements at
  Φ=0.35 vs Φ=0.65, α = 0.01, two-tailed, minimum n = 60 task runs per
  condition (3 tasks × 20 trials).
- **Secondary**: 95% CI on task success rate at Φ ∈ {0.35, 0.50, 0.65,
  0.85}; if any CI lower bound falls below 0.90, the success-stays-flat
  prediction is partially falsified.
- **Knee location**: piecewise linear regression on success vs Φ; report
  knee location with bootstrap 95% CI.

### 7. Pre-specified failure conditions

The hypothesis is **falsified** if **any one** of the following holds in
the data:

- Bandwidth at Φ=0.65 is not lower than bandwidth at Φ=0.35 (no effect or
  wrong direction): the gating doesn't reduce bandwidth at all.
- Bandwidth reduction <10% with p > 0.05: the effect is real but trivial.
- Task success at Φ=0.35 falls below 0.90: the gating causes failures
  even at moderate Φ.
- Knee location is above Φ=0.40: the safe operating range we need
  (Yellow tier and up) is too narrow.
- PE *increases* with Φ: the gating is making perception worse, not
  better — the signal we're using is wrong.

### 8. What we will publish

Whichever of these the data lands on, we publish:

- **(A) Confirmed direction + magnitude**: paper claims consciousness-
  gated encoding is a functional advantage. Phase III becomes the
  empirical floor for Phase IV.
- **(B) Confirmed direction, weaker magnitude**: paper claims the effect
  is real but smaller than designed; Phase IV proceeds with calibrated
  expectations.
- **(C) Null result**: paper claims the framework's consciousness gating
  is **not** functionally distinguishable from a fixed-quality control.
  This is a falsification of the design assumption and a publishable
  negative result. The design must be revisited before Phase IV.
- **(D) Wrong direction**: paper claims the gating actively harms task
  performance. This is a strong falsification. Phase IV is suspended
  pending root-cause analysis.

All four outcomes are publishable. The pre-registration is what prevents
post-hoc reframing of (D) as (A).

---

## PR-002: Phase IV — Markov Blanket Test

**Status**: TEMPLATE ONLY 2026-04-13. To be frozen before Phase IV
harness implementation begins (Phase IV is gated on Phase III completion).

**Source**: `plans/shiny-wibbling-quail.md` Phase IV (lines 327-376).

### 1. Hypothesis

The phone Symthaea ("brainstem") and desktop Symthaea ("prefrontal")
instances form a single Markov blanket spanning the network boundary,
operationally defined as: total system prediction error in the coupled
condition is **strictly less than** the sum of decoupled prediction
errors.

$$PE_{\text{coupled}} < PE_{\text{phone alone}} + PE_{\text{desktop replay}}$$

### 2. Independent variable

Coupling condition (3 levels):
- **Phone-only**: phone runs autonomously, no desktop input
- **Desktop-replay**: desktop runs on a recorded phone trace,
  open-loop (no feedback)
- **Coupled**: full bidirectional pipe, both instances live

### 3. Dependent variables

- **PE_phone** mean over a 60s task run (phone instance's
  `VisionTelemetry.prediction_error`)
- **PE_desktop** mean over the same 60s window (desktop instance's PE
  on its semantic state)
- **PE_combined** = PE_phone + PE_desktop in the coupled condition,
  measured as the sum of both instances' PEs at matched time points

### 4. Predicted direction

PE_coupled < PE_phone-only + PE_desktop-replay (hypothesis is one-tailed
in the coupled-is-lower direction).

### 5. Predicted magnitude

Coupled PE drop ≥ 15% below the sum of the decoupled PEs. Effect size
Cohen's d ≥ 0.4.

### 6. Pre-specified statistical test

Paired t-test (coupled vs sum-of-decoupled) on per-task PE means,
α = 0.01, one-tailed, minimum n = 30 task runs per condition.

Tasks: "navigate Settings → Display → Brightness", "open YouTube and
search NixOS", "open Clock". Each task contributes n = 10 trials per
condition.

### 7. Pre-specified failure conditions

Falsified if **any one** holds:

- PE_coupled ≥ PE_phone-only + PE_desktop-replay (no integration
  benefit).
- PE_coupled is lower but the difference is not statistically
  significant at α = 0.01.
- PE_coupled is lower in some tasks and higher in others (no consistent
  Markov-blanket effect).
- Desktop instance contributes nothing measurable (PE_desktop ≈ 0 in
  all conditions, indicating its perception module isn't engaging).

### 8. What we will publish

- **(A) Confirmed**: coupled instances are operationally one cognitive
  system across the network boundary. Phase V (cross-body dreaming)
  starts.
- **(B) Null**: coupled and decoupled are statistically indistinguishable.
  The Markov blanket framing for distributed Symthaea is falsified.
  Phase V is cancelled.
- **(C) Wrong direction**: coupling makes both worse. The bidirectional
  pipe is leaking attention away from each instance's local task.
  Phase V cancelled, design revisit required.

All three outcomes publishable.

---

## Revision history

(none — both pre-registrations are at version 1, as initially frozen
on 2026-04-13)
