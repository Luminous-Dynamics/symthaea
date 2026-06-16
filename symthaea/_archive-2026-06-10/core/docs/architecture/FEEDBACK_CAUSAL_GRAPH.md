# Feedback Causal Graph

> **Generated**: 2026-02-27
> **Source**: `src/cognitive_loop/cycle.rs`, `cycle_consciousness.rs`, `cycle_late_consciousness.rs`
> **Mechanism**: `ProposalCollector` in `feedback_state.rs`

## Overview

The cognitive loop modifies four feedback variables across ~115 sites during each
cycle. Two variables (`prediction_confidence`, `fep_lr_boost`) use the
`ProposalCollector` attribution system (Phase 2.2); two (`exploration_urge`,
`adaptive_threshold_scale`) use direct arithmetic.

| Variable | Range | Proposals | Sites |
|----------|-------|-----------|-------|
| `prediction_confidence` | 0.0–1.0 | Yes (Add/Scale/Set) | ~56 |
| `fep_lr_boost` | 1.0–3.0 | Yes (Add/Scale/Set) | ~24 |
| `exploration_urge` | 0.0–1.0 | Direct arithmetic | ~28 |
| `adaptive_threshold_scale` | 0.0–∞ | Direct arithmetic | ~10 |

### Integration Strategy (ProposalCollector)

- **Set** proposals: last one wins (rare — inference mode init, temporal discontinuity)
- **Add** proposals: averaged across all sources (consensus delta)
- **Scale** proposals: geometric mean of all factors
- Clamping enforced after integration

---

## Table 1: Prediction Confidence

### cycle.rs — Main Pipeline

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `self_model_trust` | Add | Self-model accuracy > 0.7 | +0.03×(acc−0.7) | — |
| `self_model_low_acc` | Scale | Self-model accuracy < 0.3 | ×0.98 | — |
| `resonator_error_high` | Add | Resonator PE > 0.5 | −boost×0.5 | — |
| `resonator_error_low` | Add | Resonator PE < 0.2 | +(0.2−PE)×0.03 | — |
| `binding_strong` | Add | Cross-modal binding > 0.7 | +(bind−0.7)×0.1 | Tononi (2004) |
| `binding_weak` | Add | Cross-modal binding < 0.3 | −(0.3−bind)×0.15 | Tononi (2004) |
| `resonator_factor_high` | Add | Resonator "high" state | +0.03 | — |
| `resonator_recall_prime` | Add | Best match sim > 0.3 | +sim×0.02 | Tulving (1983) |
| `pred_coherence_low` | Scale | Temporal coherence < 0.5 | ×(1.0−(0.5−coh)×0.04) | — |
| `pred_coherence_high` | Add | Temporal coherence > 0.8 | +(coh−0.8)×0.02 | — |
| `mcts_effective` | Add | Plan effectiveness > 0.6 | +(eff−0.6)×0.03 | — |
| `mcts_consolidate` | Add | Plan action = "consolidate" | +weight×0.05 | — |
| `fep_accuracy_high` | Add | FEP accuracy > 0.5 | +0.01 | — |
| `causal_graph_dense` | Add | Causal edges > 5, conf > 0.5 | +(conf−0.5)×0.03 | — |
| `moral_consent_viol` | Scale | Consent violation | ×0.7 | Greene (2013) |
| `moral_harm_detect` | Scale | Harm violation | ×0.85 | Greene (2013) |
| `moral_benefit` | Scale | Positive moral alignment | ×MORAL_BENEFIT_CONFIDENCE_BOOST | — |
| `neuromod_serotonin` | Add | Serotonin bath | +5-HT.confidence_delta() | — |
| `harmonies_low_align` | Add | Harmony misalignment | −0.02 | Eight Harmonies |
| `reasoning_chain` | Add | Chain conf > 0.7, depth ≥ 3 | +(conf−0.7)×0.05 | — |
| `limit_binding` | Add | Binding is limiting component | +0.01 | — |
| `love_resonance` | Add | Harmonic love > 0.6 | +(res−0.6)×0.04 | — |
| `epistemic_reject` | Scale | Epistemic gate rejects | ×(1.0−strength×0.15) | — |
| `temporal_continuity` | Add | Continuity > 0.7 | +(cont−0.7)×0.05 | — |
| `causal_chain_detect` | Add | Temporal causal chains > 2 | +(chains.min(10)−2)×0.005 | — |
| `coherence_vel_drop` | Scale | Coherence degradation | ×(1.0−severity×0.1) | — |
| `boredom_dampen` | Scale | Boredom > 0.7 | ×(1.0−(bor−0.7)×0.15).max(0.85) | — |
| `sigma_high` | Add | Phi integration σ > 0.5 | +((σ−0.5)×0.1).min(0.05)×0.5 | — |
| `phi_validated` | Add | Phi validation > 0.7 | +σ×(val−0.7)×0.1 | — |
| `phi_unvalidated` | Scale | Phi validation < 0.3 | ×(1.0−(0.3−val)×0.05) | — |
| `eq_v2_deviation` | Add | Equation_v2 deviates | +(eq_v2×(1−wt)×0.03) | — |
| `cross_mod_agree` | Add | Module agreement > 0.8 | +(agree−0.8)×0.05 | — |
| `cross_mod_disagree` | Scale | Module agreement < 0.3 | ×(1.0−(0.3−agree)×0.1) | — |

### cycle_consciousness.rs

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `harmonies_low_align` | Add | Harmony misaligned, not approved | −0.02 | Eight Harmonies |
| `reasoning_chain` | Add | Strong chain (conf > 0.7) | +(conf−0.7)×0.03 | — |
| `temporal_continuity` | Add | Continuity > 0.7 | +(cont−0.7)×0.05 | — |
| `causal_chain_detect` | Add | Causal chains > 2 | +(chains.min(10)−2)×0.005 | — |
| `dissipative_maintain` | Add | Dissipative: Maintain | +0.005 | Prigogine |
| `dissipative_equilibrium` | Add | Dissipative: IncreaseActivity | −0.01 | Prigogine |
| `dissipative_ordered` | Add | Dissipative: IncreaseDiff. | −0.005 | Prigogine |
| `equation_v2_high` | Add | Consciousness > 0.6 | +(cons−0.6)×0.08 | — |
| `harmonic_interference` | Add | Interferences detected | −count.min(3)×0.01 | — |

### cycle_late_consciousness.rs

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `body_valence_pos` | Add | Body valence > 0.3 | +valence×0.02 | Damasio (1994) |
| `body_valence_neg` | Add | Body valence < −0.3 | +valence×0.03 | Damasio (1994) |
| `arousal_trap_escape` | Scale | Arousal trap count > 10 | ×0.9 | — |
| `narrative_self_strong` | Scale | Narrative ψ > 0.5 | ×1.02 | Gallagher (2000) |
| `narrative_self_weak` | Scale | Narrative ψ < 0.2 | ×0.95 | Gallagher (2000) |
| `gwt_broadcast` | Add | GWT broadcast active | +GWT_BROADCAST_CONFIDENCE_BOOST | Baars (1988) |
| `quantum_coherence_high` | Add | Quantum coherence > 0.6 | +(qc−0.6)×0.05 | — |
| `quantum_decoherence` | Scale | Quantum coherence < 0.2 | ×0.98 | — |
| `temporal_discontinuity` | Scale | Temporal discontinuity | ×0.8 | — |
| `enacted_meaning_neg` | Scale | Enacted meaning < −0.5 | ×(1.0+val×0.1) | — |

---

## Table 2: Learning Rate (fep_lr_boost)

### cycle.rs

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `goal_priority` | Scale | Goal priority > 0.5 | ×(1.0+(pri−0.5)×0.1) | — |
| `wm_stiff` | Add | WM stiffness > 0.5 | +(stiff−0.5)×0.05 | — |
| `wm_spongy` | Scale | WM stiffness < 0.2 | ×(1.0−(0.2−stiff)×0.15) | — |
| `mcts_exploit` | Scale | Plan action = "exploit" | ×(1.0−weight×0.1), floor 1.0 | — |
| `fep_complexity` | Scale | FEP complexity > 1.0 | ×(1.0−(comp−1.0).min(0.5)×0.1) | Occam's razor |
| `neuromod_dopamine` | Scale | Dopamine bath | ×DA.learning_rate_factor() | — |
| `fep_surprise` | Add | Surprise detected | +(fep/SCALE).clamp(0.1,0.5) | — |
| `fep_decay` | Scale | Not surprised | ×FEP_LR_DECAY | — |
| `gaba_inhibition` | Scale | GABA < 0.95 | ×gaba_inhibition | — |
| `coherence_degraded` | Scale | Coherence degradation | ×1.3 | — |
| `reflection_decrease` | Scale | Reflection: decrease LR | ×0.9 | — |
| `reflection_increase` | Scale | Reflection: increase LR | ×1.1 | — |
| `guide_pragmatic` | Scale | Pragmatic question | ×1.02 | — |
| `glutamate_fatigue` | Scale | Glutamate fatigue < 1.0 | ×fatigue_factor() | Circadian |
| `limit_efficacy` | Scale | Efficacy limiting | ×1.05 | — |

### cycle_consciousness.rs

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `dissipative_coherence` | Scale | Dissipative: IncreaseCoherence | ×1.05 | Prigogine |
| `dissipative_integration` | Scale | Dissipative: IncreaseIntegration | ×1.03 | Prigogine |

### cycle_late_consciousness.rs

| Label | Method | Condition | Effect | Citation |
|-------|--------|-----------|--------|----------|
| `affective_arousal_suppress` | Scale | Arousal > 0.7 | ×(1.0−(aro−0.7)×0.25).min(0.08) | Yerkes-Dodson |
| `arousal_trap_recovery` | Scale | Arousal trap recovery | ×(1.0−intensity×0.1) | — |
| `low_arousal_consolidate` | Scale | Arousal < 0.3 | ×(1.0+(0.3−aro)×0.3).min(0.05) | — |
| `hierarchical_free_energy` | Scale | High HFE | ×(1.0+(hfe×0.02).min(0.1)) | Friston (2008) |
| `temporal_discontinuity` | Set | Temporal discontinuity | =1.0 (hard reset) | — |
| `persistent_discontinuity` | Scale | Persistent discontinuity | ×1.5 | — |
| `low_entropy_consolidate` | Scale | Thermo entropy < 0.3 | ×(1.0+(0.3−ent)×0.08).min(0.08) | — |
| `allostatic_overload` | Scale | Allostatic load > 0.7 | ×(1.0−(load−0.7)×0.5) | McEwen (2004) |

---

## Table 3: Exploration Urge (direct arithmetic)

| File | Condition | Effect | Citation |
|------|-----------|--------|----------|
| cycle.rs | Startup warmup | ×warmup_progress | — |
| cycle.rs | High quantum coherence | +(coh−thresh)×boost | Orch OR |
| cycle.rs | High resonator PE (0.5–1.0) | +(pe−0.5)×0.08 | — |
| cycle.rs | Goal pursuit success | +(priority×0.03) | — |
| cycle.rs | Conceptual confusion | +0.08 | — |
| cycle.rs | Poor plan effectiveness | +(0.3−eff)×0.02 | — |
| cycle.rs | MCTS explore action | +(weight×0.08) | — |
| cycle.rs | FEP surprise > threshold | +((surp−thresh)×0.1).min(0.05) | — |
| cycle.rs | High FEP pragmatic (>0.7) | ×(1.0−(prag−0.7)×0.3) | — |
| cycle.rs | Low FEP pragmatic (<0.3) | +((0.3−prag)×0.15).min(0.05) | — |
| cycle.rs | Sparse causal graph | +0.02 | — |
| cycle.rs | Moral concern | ×MORAL_CONCERN_EXPLORATION_DAMPEN | Greene (2013) |
| cycle.rs | Harm violation | ×0.4 | Greene (2013) |
| cycle.rs | High PFE (>0.5) | +((pfe−0.5)×0.2).min(0.1) | Friston (2010) |
| cycle.rs | Low PFE (<0.2) | ×(1.0−(0.2−pfe)×0.15).min(0.05) | — |
| cycle.rs | Neuromod bath | +NE.exploration_delta() | — |
| cycle.rs | D2 flexibility | ×(0.5+(urge−0.5)×flex) | — |
| cycle.rs | High NE phase (>0.3) | +(phase−0.3)×0.15 | — |
| cycle.rs | GABA inhibition | ×gaba_inhibition | — |
| cycle.rs | High epistemic value | +0.1 | — |
| cycle.rs | Reflection increase | +0.12 | — |
| cycle.rs | Reflection decrease | ×0.75 | — |
| cycle.rs | Epistemic question | +0.03 | — |
| cycle.rs | Low epistemic confidence | ×(1.0+caution) | — |
| cycle.rs | Low cross-mod agreement | +(0.3−agree)×0.15 | — |
| cycle.rs | Low unified quality | ×0.9 | — |
| cycle_consciousness.rs | Dissipative: IncreaseActivity | +(inc−0.5)×0.15 | Prigogine |
| cycle_consciousness.rs | Dissipative: IncreaseCoherence | ×0.9 | Prigogine |
| cycle_consciousness.rs | Dissipative: IncreaseDiff. | +0.02 | Prigogine |
| cycle_consciousness.rs | Low eq_v2 consciousness | +0.02 | — |
| cycle_late_consciousness.rs | Prefrontal veto | =0.0 | Miller & Cohen (2001) |
| cycle_late_consciousness.rs | Dual veto freeze | =0.3 | — |
| cycle_late_consciousness.rs | Arousal trap recovery | +((count−5)/5)×0.025 | — |
| cycle_late_consciousness.rs | Arousal trap escape | =1.0 | — |
| cycle_late_consciousness.rs | Low attention focus | +(0.3−focus)×0.06 | — |
| cycle_late_consciousness.rs | High emotional suppression | ×0.85 | — |
| cycle_late_consciousness.rs | Persistent discontinuity | ×0.7 | — |
| cycle_late_consciousness.rs | High thermo entropy | +(ent−0.7)×0.1 | — |
| cycle_late_consciousness.rs | High sensorimotor surprise | +(surp×0.1).min(0.15) | — |
| cycle_late_consciousness.rs | Low embodied agency | ×(1.0−(0.3−agency)×0.1) | — |

---

## Table 4: Adaptive Threshold Scale (direct arithmetic)

| File | Condition | Effect | Citation |
|------|-----------|--------|----------|
| cycle.rs | Strong binding (>0.7) | ×(1.0−(bind−0.7)×0.3) | Tononi (2004) |
| cycle.rs | Weak binding (<0.3) | ×(1.0+(0.3−bind)×0.2) | Tononi (2004) |
| cycle.rs | Neuromod ACh threshold | ×ACh.threshold_factor() | Yu & Dayan (2005) |
| cycle.rs | Low epistemic confidence | ×(1.0+(0.4−conf)×0.3) | — |
| cycle.rs | High epistemic confidence | ×(1.0−(conf−0.8)×0.15) | — |
| cycle.rs | Coherence velocity drop | ×(1.0+severity×0.2) | — |
| cycle_late_consciousness.rs | Temporal discontinuity | ×0.8 | — |
| cycle_late_consciousness.rs | High temporal coherence | ×1.01 | — |
| cycle_late_consciousness.rs | Baseline return drift | +=(1.0−scale)×0.02 | — |

---

## Runtime Tracing

When `CognitiveLoopConfig::trace_feedback` is `true`, the `ProposalCollector`
records individual proposals which can be inspected via `dump_proposals()`. The
dumps are written to `CycleMetadata::feedback_trace_confidence` and
`CycleMetadata::feedback_trace_lr` as `Vec<(source, description)>` tuples.

This is intended for debugging and development — not production.
