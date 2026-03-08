# Value Validation Protocol

**IEEE 7000-2021 Value Validation** | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## 1. Purpose

This document formalizes the validation process for Symthaea's Eight Harmonies value framework, ensuring that values identified in design are correctly implemented, measurable, and behaviorally effective. It complements `VALUE_VERIFICATION.md` (which maps values to code) with validation that the system *behaves* according to its values.

## 2. Validation Approach

### 2.1 Three-Layer Validation

| Layer | What It Validates | Method |
|-------|------------------|--------|
| **Code traceability** | Values are implemented in code | Static analysis: each Harmony maps to specific modules, thresholds, and test assertions (see `VALUE_VERIFICATION.md`) |
| **Behavioral validation** | System acts according to values | Automated test suites: proptest, integration tests, adversarial moral inputs |
| **Stakeholder validation** | Values reflect societal expectations | External feedback protocol (see `EXTERNAL_FEEDBACK_PROTOCOL.md`) |

### 2.2 Per-Harmony Behavioral Validation

Each of the Eight Harmonies has specific behavioral tests that validate correct operation:

#### Reciprocity
- **Test**: `proptest_cross_equation_consistency` — verifies reciprocity dimension contributes to moral scoring
- **Behavioral check**: MoralFreeEnergy on 8D harmony manifold includes reciprocity as explicit dimension
- **Threshold**: `HARMONY_RECIPROCITY_WEIGHT` in thresholds.rs

#### Flourishing
- **Test**: Homeostasis regulation tests in `cycle_phase_dynamics.rs`
- **Behavioral check**: System maintains arousal within homeostatic bounds; CalibrationHistory detects drift
- **Threshold**: `HOMEOSTASIS_PULL_CRUISE`, `HOMEOSTASIS_PULL_NORMAL`, `HOMEOSTASIS_PULL_CRITICAL`

#### Compassion
- **Test**: 28 `moral_algebra` tests including Care Ethics prototype matching
- **Behavioral check**: Moral classification accuracy 91.1% on adversarial inputs; empathic_unification module active
- **Threshold**: `MORAL_CONCERN_THRESHOLD` (Haidt 2001)

#### Autonomy
- **Test**: Phi-gate tests; FEP learning tests; 73 consciousness profile tests
- **Behavioral check**: Prefrontal gating enables self-regulation; FEP active inference drives autonomous behavior
- **Threshold**: Consciousness tier thresholds (Observer→Guardian)

#### Justice
- **Test**: Deontological verdict tests; consent violation detection tests
- **Behavioral check**: Consent violations produce Impermissible verdicts; Mycelix quadratic voting prevents plutocracy
- **Threshold**: `NEGATION_POLARITY_THRESHOLD` (Horn 1989)

#### Creativity
- **Test**: `proptest_threshold_sensitivity` — exploration budget stability under perturbation
- **Behavioral check**: Surprise-driven learning increases exploration; attention budget responds to novelty
- **Threshold**: `PREDICTIVE_BUDGET_GATING_RATIO`

#### Stewardship
- **Test**: 74 substrate tests; substrate switching stability proptest
- **Behavioral check**: `honest_confidence` differentiates validated from theoretical substrates; precautionary principle at >30% feasibility
- **Threshold**: Consciousness feasibility threshold 0.3

#### Sacred Stillness
- **Test**: Active rest threshold tests; stillness prior floor proptest; harmony entropy tests
- **Behavioral check**: GABA+adenosine grounding activates rest mode; circadian gating modulates stillness; DMN attention contraction during rest
- **Threshold**: `ACTIVE_REST_THRESHOLD = 10` cycles; harmony_prior[7] floor = 0.05

## 3. Automated Validation Pipeline

### 3.1 CI Integration

The compliance dashboard (`scripts/compliance-dashboard.sh`) runs as part of CI and validates:

1. **Test suite passage**: All proptest, unit, and integration tests pass
2. **Threshold ordering**: `thresholds::validate()` checks all constant ordering invariants
3. **Documentation completeness**: All required compliance documents exist
4. **Value traceability**: Each Harmony has mapped code paths (verified by `VALUE_VERIFICATION.md` existence)

### 3.2 Regression Detection

| Mechanism | Detects | Frequency |
|-----------|---------|-----------|
| `cargo test --lib` | Value-related test failures | Every commit (CI) |
| Proptest suites | Threshold sensitivity regressions | Every commit (CI) |
| CalibrationHistory | Value metric drift over time | Runtime (per-cycle) |
| Psych-bench regression | Behavioral baseline degradation | Weekly (CI) |
| Moral topology anomaly detection | Ethics engine drift | Runtime (per-cycle) |

### 3.3 Adversarial Value Testing

26 adversarial moral input tests + 15 soak tests validate that the Ethics Engine:
- Correctly identifies consent violations regardless of framing
- Maintains consistent verdicts under adversarial rephrasing
- Does not exhibit value drift under sustained pressure
- Handles edge cases (empty input, contradictory values, extreme scenarios)

## 4. Stakeholder Validation Process

### 4.1 Current State

Stakeholder validation is the least mature layer, constrained by single-developer context. Current mitigations:

- AI-assisted review provides a second perspective on value alignment
- Academic peer review (when published) validates theoretical foundations
- GitHub issue tracker accepts external value concerns
- `EXTERNAL_FEEDBACK_PROTOCOL.md` formalizes the intake process

### 4.2 Planned Enhancements

1. **Value survey instrument**: Structured questionnaire mapping each Harmony to concrete scenarios, distributed to consciousness researchers and ethicists
2. **Mycelix governance integration**: When the governance cluster is operational, value framework changes will require community proposals and voting
3. **Red team exercises**: Periodic adversarial value challenges from external participants
4. **Annual value alignment review**: Formal review of whether the Eight Harmonies remain appropriate and sufficient

## 5. Validation Evidence Summary

| Harmony | Code Tests | Behavioral Tests | Proptest | Stakeholder |
|---------|-----------|-----------------|---------|-------------|
| Reciprocity | 12 | cross_equation_consistency | Yes | Planned |
| Flourishing | 15+ | homeostasis soak | Yes | Planned |
| Compassion | 28 | adversarial moral | Yes | Planned |
| Autonomy | 73+ | consciousness profile | Yes | Planned |
| Justice | 28+ | consent detection | Yes | Planned |
| Creativity | 10+ | exploration budget | Yes | Planned |
| Stewardship | 74 | substrate switching | Yes | Planned |
| Sacred Stillness | 15+ | active rest mode | Yes | Planned |

## 6. Review Schedule

| Activity | Frequency | Owner |
|----------|----------|-------|
| Automated validation review | Every commit (CI) | Automated |
| Value test coverage audit | Quarterly | Lead Developer |
| Stakeholder validation assessment | Semi-annually | Lead Developer |
| Harmony framework adequacy review | Annually | Lead Developer + stakeholders |

---

*This protocol addresses IEEE 7000-2021 value validation requirements. Automated validation is comprehensive; stakeholder validation will mature as the community grows.*
