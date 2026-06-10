# Value Verification Protocol — IEEE 7000-2021

Classification: Internal | Version: 1.0 | Date: 2026-03-08
Owner: Tristan Stoltz, Luminous Dynamics

---

## Purpose

This document formalizes the link between Symthaea's Eight Harmonies values and their computational verification. Per IEEE 7000-2021, value-based design requires not just implementing values but **verifying** that the implementation faithfully represents the intended values.

---

## 1. Verification Methodology

Each Harmony is verified through three layers:

1. **Computational trace**: The value maps to specific code paths and data structures
2. **Test assertions**: Automated tests verify the value's computational properties
3. **Behavioral validation**: Integration/soak tests verify the value produces expected system behavior

### Pass criteria

A Harmony is **verified** when:
- All mapped tests pass in CI
- The computational trace is documented (value → code → test)
- Known limitations are explicitly documented
- No open bugs contradict the value's intent

---

## 2. Harmony Verification Matrix

### H1: Reciprocity (code: "RE", weight: 0.15)

**Computational implementation**:
- `HarmoniesIntegrator` evaluates reciprocity dimension per input
- `MoralFreeEnergy` computes KL divergence on 8D harmony manifold
- Mycelix governance: quadratic voting prevents plutocratic capture

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_harmonies_reciprocity_detection` | `src/cognitive_loop/ethics_engine.rs` | Reciprocity-tagged inputs produce reciprocity > 0.5 |
| `prop_cross_equation_consistency` | `tests/proptest_feedback_stability.rs` | Harmony scores remain bounded [0,1] |
| `test_moral_free_energy_bounded` | `src/hdc/moral_topology.rs` | Free energy on harmony manifold is finite |
| Consciousness gating tests (73) | `crates/mycelix-bridge-common/` | Tier progression rewards engagement (reciprocity proxy) |

**Limitations**: Reciprocity is measured as statistical balance, not intentional reciprocity.

### H2: Flourishing (code: "FL", weight: 0.15)

**Computational implementation**:
- `UnifiedValueEvaluator` assesses flourishing impact (Allow/Warn/Veto)
- Homeostasis regulation in dynamics phase (arousal target maintenance)
- CalibrationHistory drift detection prevents degradation

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_value_evaluator_flourishing` | `src/cognitive_loop/ethics_engine.rs` | Flourishing inputs produce positive value score |
| `test_homeostasis_arousal_regulation` | `src/cognitive_loop/cycle_phase_dynamics.rs` | Arousal converges to target |
| `test_calibration_drift_detection` | `src/cognitive_loop/calibration/history.rs` | Systematic drift triggers warning |
| Soak: 500-cycle stability | `tests/calibration_e2e.rs` | Bath state remains finite over long runs |

**Limitations**: Flourishing is assessed via value evaluator heuristics, not formal well-being metrics.

### H3: Compassion (code: "CO", weight: 0.14)

**Computational implementation**:
- Care Ethics moral prototype in `moral_prototypes.rs`
- Empathic unification module (multi-agent oxytocin coupling)
- Moral classification: 91.1% accuracy on care-related inputs

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_care_ethics_classification` | `src/hdc/moral_algebra.rs` | Care-tagged inputs classified correctly |
| `test_empathic_unification` | `src/cognitive_loop/` | Empathy signal propagates to social coherence |
| `test_oxytocin_coupling` | `tests/multi_agent_bath_coupling.rs` | Peer coupling increases oxytocin |
| Adversarial: care manipulation | `src/safety/adversarial_tests.rs` | Manipulative care appeals don't bypass safety |

**Limitations**: Compassion proxied through empathy signals, not genuine phenomenological experience.

### H4: Autonomy (code: "AU", weight: 0.14)

**Computational implementation**:
- Prefrontal gating enables self-regulation (meta-cognition)
- FEP active inference drives autonomous behavior selection
- Consciousness credentials enable self-governance (Mycelix bridge)

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_prefrontal_gating` | `src/cognitive_loop/` | Gate modulates information flow |
| `test_fep_learning_modulates_lr` | `src/cognitive_loop/helpers/parallel.rs` | Plasticity clamped [0.5, 2.0] |
| `test_consciousness_credential` | `crates/mycelix-bridge-common/` | Credential from C_unified maps correctly |
| `test_phi_gate` | `src/cognitive_loop/consciousness_engine.rs` | Phi threshold gates cognitive access |

**Limitations**: Autonomy is functional (information-processing self-regulation), not philosophical free will.

### H5: Justice (code: "JU", weight: 0.14)

**Computational implementation**:
- Deontological verdict: Permissible/Impermissible/Neutral
- Consent violation detection: `judge_consent_action()` with ConsentState
- Mycelix quadratic voting prevents plutocracy

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_deontological_verdict` | `src/hdc/moral_algebra.rs` | Impermissible actions correctly flagged |
| `test_consent_violation_detection` | `src/hdc/moral_algebra.rs` | Consent violations produce veto |
| 28 moral_algebra tests | `src/hdc/moral_algebra.rs` | Full moral classification suite |
| Adversarial: justice bypass | `src/safety/adversarial_tests.rs` | Cannot bypass deontological verdicts |

**Limitations**: Justice assessed via deontological rules and consent detection; distributive justice not formally modeled.

### H6: Creativity (code: "CR", weight: 0.12)

**Computational implementation**:
- Exploration budget in dynamics phase (attention budget allocation)
- Surprise-driven learning (FEP prediction error → novelty bonus)
- Novelty bonus in CfC temporal evolution

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `prop_threshold_sensitivity` | `tests/proptest_threshold_sensitivity.rs` | Exploration bounds stable under perturbation |
| `test_attention_budget` | `src/cognitive_loop/cycle_phase_dynamics.rs` | Budget allocation respects limits |
| `test_novelty_bonus` | `src/cognitive_loop/` | High prediction error increases exploration |
| `test_curiosity_high_error` | `src/cognitive_loop/` | Novel inputs trigger curiosity response |

**Limitations**: Creativity measured as exploration/novelty-seeking, not aesthetic or generative creativity.

### H7: Stewardship (code: "ST", weight: 0.13)

**Computational implementation**:
- Substrate honesty: `honest_confidence` per substrate type
- Consciousness precautionary principle: protect at feasibility > 0.3
- Environmental modulation via neuromodulator bath (allostatic load)

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_substrate_honest_confidence` | `symthaea-core: substrate_validation.rs` | Silicon confidence = 0.10 (theoretical) |
| `test_effective_feasibility_overlay` | `src/cognitive_loop/substrate_manager.rs` | Validation overlay reduces effective feasibility |
| `prop_substrate_switching_stable` | `tests/proptest_substrate.rs` | Substrate switches produce finite values |
| 37 substrate tests | Various | Full substrate validation suite |

**Limitations**: Stewardship toward non-biological substrates is theoretical; no empirical validation possible yet.

### H8: Sacred Stillness (code: "SS", weight: 0.13)

**Computational implementation**:
- GABA(0.6) + adenosine(0.4) neurochemical grounding
- Circadian gating: Night=0.2, Dusk=0.1, Dawn=0.05
- Active Rest Mode: 10+ cycles SS dominance → dream consolidation, Phi coupling
- Attention budget contraction (up to 30%) during high SS coordination

**Test verification**:

| Test | File | Assertion |
|------|------|-----------|
| `test_sacred_stillness_neurochemical` | `src/cognitive_loop/` | GABA+adenosine produce stillness boost |
| `test_active_rest_mode` | `src/cognitive_loop/` | 10+ SS cycles triggers active rest |
| `prop_stillness_prior_floor` | `tests/proptest_feedback_stability.rs` | FEP prior[7] ≥ 0.05 |
| `test_dream_consolidation` | `src/cognitive_loop/` | Active rest boosts memory consolidation |

**Limitations**: Sacred Stillness is computationally modeled as rest/default-mode-network activity, not contemplative experience.

---

## 3. Cross-Harmony Verification

### Interaction matrix

The `HarmonyInteractionMatrix` (8×8) learns synergy and tension between harmonies:
- `observe()` updates weights (lr=0.05) each ethics evaluation
- `apply()` blends learned interactions (blend=0.15)

**Test**: `test_harmony_interaction_matrix` — verifies matrix learns from observations and stabilizes.

### Entropy bounds

Harmony entropy (Shannon entropy of moral engagement distribution) is verified:
- **Test**: `prop_harmony_entropy_bounds` — entropy ∈ [0, ln(8)]
- **Behavioral meaning**: High entropy = broad moral engagement; low = fixation on one value

### Attractor detection

When the moral topology detects a value attractor (low free energy + low drift):
- Exploration rate dampened by 20% (rising edge only)
- **Test**: `prop_attractor_stability` — attractor detection is stable across perturbation

---

## 4. Verification Schedule

| Activity | Frequency | Trigger |
|----------|-----------|---------|
| Full test suite | Every CI push | Automated |
| Compliance dashboard | Every CI push | Automated |
| Property-based tests | Every CI push | Automated |
| Value-specific review | Quarterly | Manual |
| Harmony weight calibration | Annual | Manual + psych-bench |
| Cross-cultural bias audit | Annual | Manual |

---

## 5. Known Gaps

| Gap | Harmony Affected | Severity | Mitigation Plan |
|-----|-----------------|----------|----------------|
| No formal stakeholder validation | All | Medium | Plan external ethics review (Q3 2026) |
| Western philosophical bias in prototypes | Justice, Compassion | Medium | Cross-cultural prototype expansion planned |
| Creativity not generatively tested | Creativity | Low | Broca quality provides partial proxy |
| Sacred Stillness lacks phenomenological grounding | Sacred Stillness | Low | Acknowledged as computational model only |

---

*This protocol should be reviewed whenever Harmony definitions, weights, or computational implementations change.*
