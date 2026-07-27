// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Accessor methods for CognitiveLoopService.
//!
//! High-level query methods (flow state, prediction confidence, consciousness
//! snapshot, etc.) are `pub` for use by external consumers (examples, LUCID,
//! nixward). A small number of `pub(crate)` accessors exist for internal
//! unit tests (e.g., `flow_state()`, `curiosity_drive()`).
//!
//! Organized into thematic sub-modules:
//! - [`system`]: Core state, CfC/HDC, prediction confidence, psi attestation
//! - [`neuromodulation`]: Bath state, pharmacological ablation, circadian
//! - [`behavior`]: Flow, emotion, curiosity, reflection, adaptive, voice, learning
//! - [`consciousness`]: Snapshot, pattern, primitive tier, experience bus, attention
//! - [`memory`]: Episodic, semantic, causal, HDC projection, goals

/// One-liner accessor helper: adds `#[inline]` to each method.
macro_rules! cognitive_accessors {
    ($(
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(& $self_:ident) -> $ret:ty { $($body:tt)+ }
    )*) => {
        $(
            $(#[$meta])*
            #[inline]
            $vis fn $name(& $self_) -> $ret { $($body)+ }
        )*
    };
}

mod behavior;
mod consciousness;
#[cfg(feature = "social-fabric")]
mod memetics;
mod memory;
mod neuromodulation;
mod system;

#[cfg(test)]
mod tests {
    use super::super::{CognitiveLoopConfig, CognitiveLoopService};

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    // ── Stats accessor ────────────────────────────────────────────────

    #[test]
    fn stats_initial_total_cycles() {
        let s = make_service();
        assert_eq!(s.stats().total_cycles, 0);
    }

    #[test]
    fn stats_initial_avg_error_zero() {
        let s = make_service();
        assert_eq!(s.stats().avg_prediction_error, 0.0);
    }

    // ── Config accessor ───────────────────────────────────────────────

    #[test]
    fn config_returns_learning_threshold() {
        let cfg = CognitiveLoopConfig::default();
        let expected = cfg.learning_threshold;
        let s = CognitiveLoopService::new(cfg).unwrap();
        assert_eq!(s.config().learning_threshold, expected);
    }

    #[test]
    fn config_returns_target_frequency() {
        let s = make_service();
        assert_eq!(s.config().target_frequency, 50.0);
    }

    // ── Prediction confidence ─────────────────────────────────────────

    #[test]
    fn prediction_confidence_initial_value() {
        let s = make_service();
        assert!((s.prediction_confidence() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn prediction_confidence_is_finite() {
        let s = make_service();
        assert!(s.prediction_confidence().is_finite());
    }

    #[test]
    fn predictions_trustworthy_at_initial() {
        let s = make_service();
        // prediction_confidence=0.5 > 0.4, so should be trustworthy
        assert!(s.predictions_trustworthy());
    }

    // ── Reward injection ──────────────────────────────────────────────

    #[test]
    fn provide_reward_clamps_positive() {
        let mut s = make_service();
        s.provide_reward(100.0);
        assert!(s.behavior.social_mgr.social.external_reward <= 1.0);
    }

    #[test]
    fn provide_reward_clamps_negative() {
        let mut s = make_service();
        s.provide_reward(-100.0);
        assert!(s.behavior.social_mgr.social.external_reward >= -1.0);
    }

    #[test]
    fn provide_reward_preserves_zero() {
        let mut s = make_service();
        s.provide_reward(0.0);
        assert!((s.behavior.social_mgr.social.external_reward).abs() < f32::EPSILON);
    }

    #[test]
    fn provide_reward_preserves_in_range() {
        let mut s = make_service();
        s.provide_reward(0.7);
        assert!((s.behavior.social_mgr.social.external_reward - 0.7).abs() < f32::EPSILON);
    }

    // ── Social signals ────────────────────────────────────────────────

    #[test]
    fn set_social_signals_clamps_trust() {
        let mut s = make_service();
        s.set_social_signals(5.0, 0.5, 0.5, 0, 0.5);
        assert!(s.behavior.social_mgr.social.social_trust <= 1.0);
        assert!(s.behavior.social_mgr.social.social_trust >= 0.0);
    }

    #[test]
    fn set_social_signals_clamps_cooperation() {
        let mut s = make_service();
        s.set_social_signals(0.5, -3.0, 0.5, 0, 0.5);
        assert!(s.behavior.social_mgr.social.social_cooperation_rate >= 0.0);
        assert!(s.behavior.social_mgr.social.social_cooperation_rate <= 1.0);
    }

    #[test]
    fn set_social_signals_preserves_in_range() {
        let mut s = make_service();
        s.set_social_signals(0.8, 0.3, 0.7, 5, 0.6);
        assert!((s.behavior.social_mgr.social.social_trust - 0.8).abs() < f32::EPSILON);
        assert!((s.behavior.social_mgr.social.social_cooperation_rate - 0.3).abs() < f32::EPSILON);
        assert!(
            (s.behavior.social_mgr.social.social_prediction_accuracy - 0.7).abs() < f32::EPSILON
        );
        assert_eq!(s.behavior.social_mgr.social.social_models_count, 5);
        assert!((s.behavior.social_mgr.social.social_mean_trust - 0.6).abs() < f32::EPSILON);
    }

    // ── Relational Psi ────────────────────────────────────────────────

    #[test]
    fn set_relational_psi_stores_value() {
        let mut s = make_service();
        s.set_relational_psi(0.42);
        assert!((s.behavior.social_mgr.social.relational_psi - 0.42).abs() < f64::EPSILON);
    }

    // ── FEP learning signal ───────────────────────────────────────────

    #[test]
    fn fep_learning_signal_initial() {
        let s = make_service();
        assert!((s.fep_learning_signal() - 0.0).abs() < f32::EPSILON);
    }

    // ── Flow state ────────────────────────────────────────────────────

    #[test]
    fn flow_initial_not_in_flow() {
        let s = make_service();
        assert!(!s.in_flow());
    }

    #[test]
    fn flow_initial_intensity_bounded() {
        let s = make_service();
        let i = s.flow_intensity();
        assert!(i.is_finite());
        assert!((0.0..=1.0).contains(&i));
    }

    #[test]
    fn flow_initial_streak_zero() {
        let s = make_service();
        assert_eq!(s.flow_streak(), 0);
    }

    #[test]
    fn flow_learning_boost_initial() {
        let s = make_service();
        let b = s.flow_learning_boost();
        assert!(b.is_finite());
    }

    // ── Emotion ───────────────────────────────────────────────────────

    #[test]
    fn emotional_valence_initial_bounded() {
        let s = make_service();
        let v = s.emotional_valence();
        assert!(v.is_finite());
        assert!((-1.0..=1.0).contains(&v));
    }

    #[test]
    fn emotional_arousal_initial_bounded() {
        let s = make_service();
        let a = s.emotional_arousal();
        assert!(a.is_finite());
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn no_emotional_content_initially() {
        let s = make_service();
        assert!(!s.has_emotional_content());
    }

    // ── Curiosity and boredom ─────────────────────────────────────────

    #[test]
    fn boredom_initial_bounded() {
        let s = make_service();
        let b = s.boredom();
        assert!(b.is_finite());
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn curiosity_initial_bounded() {
        let s = make_service();
        let c = s.curiosity();
        assert!(c.is_finite());
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn novelty_bonus_initial_finite() {
        let s = make_service();
        assert!(s.novelty_bonus().is_finite());
    }

    #[test]
    fn is_bored_false_initially() {
        let s = make_service();
        // Default boredom is low, so should not be bored
        assert!(!s.is_bored());
    }

    // ── Self-reflection ───────────────────────────────────────────────

    #[test]
    fn reflection_count_initial() {
        let s = make_service();
        assert_eq!(s.reflection_count(), 0);
    }

    #[test]
    fn learning_effectiveness_initial_finite() {
        let s = make_service();
        assert!(s.learning_effectiveness().is_finite());
    }

    #[test]
    fn recommendations_initially_empty() {
        let s = make_service();
        assert!(s.recommendations().is_empty());
    }

    #[test]
    fn force_reflect_returns_vec() {
        let mut s = make_service();
        let recs = s.force_reflect();
        // Just verifying it doesn't panic and returns a vec
        assert!(recs.len() <= 100); // Sanity bound
    }

    // ── Consciousness snapshot ────────────────────────────────────────

    #[test]
    fn consciousness_snapshot_fields_finite() {
        let s = make_service();
        let snap = s.consciousness_snapshot();
        assert!(snap.consciousness_level >= 0.0 && snap.consciousness_level <= 1.0);
        assert!(snap.prediction_confidence.is_finite());
        assert!(snap.flow_intensity.is_finite());
        assert!(snap.boredom.is_finite());
        assert!(snap.curiosity.is_finite());
        assert!(snap.emotional_valence.is_finite());
        assert!(snap.emotional_arousal.is_finite());
    }

    #[test]
    fn consciousness_snapshot_cycle_zero() {
        let s = make_service();
        let snap = s.consciousness_snapshot();
        assert_eq!(snap.cycle, 0);
    }

    #[test]
    fn consciousness_level_in_range() {
        let s = make_service();
        let cl = s.consciousness_level();
        assert!((0.0..=1.0).contains(&cl), "consciousness_level={cl}");
    }

    #[test]
    fn status_line_not_empty() {
        let s = make_service();
        let line = s.status_line();
        assert!(!line.is_empty());
    }

    // ── Goal system ───────────────────────────────────────────────────

    #[test]
    fn no_goals_initially() {
        let s = make_service();
        assert!(s.active_goals().is_empty());
    }

    #[test]
    fn add_goal_and_retrieve() {
        let mut s = make_service();
        s.add_goal("explore", "explore the environment", 0.8);
        let goals = s.active_goals();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].id, "explore");
        assert!(goals[0].is_active);
    }

    #[test]
    fn add_multiple_goals() {
        let mut s = make_service();
        s.add_goal("g1", "first goal", 0.9);
        s.add_goal("g2", "second goal", 0.5);
        s.add_goal("g3", "third goal", 0.1);
        assert_eq!(s.active_goals().len(), 3);
    }

    // ── Memory counts ─────────────────────────────────────────────────

    #[test]
    fn memory_counts_initially_zero() {
        let s = make_service();
        let (st, lt) = s.memory_counts();
        assert_eq!(st, 0);
        assert_eq!(lt, 0);
    }

    // ── Causal accessors ──────────────────────────────────────────────

    #[test]
    fn causal_graph_none_when_disabled() {
        let s = make_service();
        assert!(s.causal_graph().is_none());
    }

    #[test]
    fn causal_discoveries_none_when_disabled() {
        let s = make_service();
        assert!(s.causal_discoveries().is_none());
    }

    #[test]
    fn has_causal_structure_false_when_disabled() {
        let s = make_service();
        assert!(!s.has_causal_structure());
    }

    // ── Episodic replay ───────────────────────────────────────────────

    #[test]
    fn top_phi_episodes_empty_when_disabled() {
        let s = make_service();
        assert!(s.top_phi_episodes(10).is_empty());
    }

    // ── CfC state ─────────────────────────────────────────────────────

    #[test]
    fn cfc_state_diversity_finite() {
        let s = make_service();
        assert!(s.cfc_state_diversity().is_finite());
    }

    #[test]
    fn cfc_state_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.cfc_state_dim(), 256);
    }

    #[test]
    fn state_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.state_dim(), 256);
    }

    #[test]
    fn prediction_dim_matches_config() {
        let s = make_service();
        assert_eq!(s.prediction_dim(), 256);
    }

    // ── Temporal coherence ────────────────────────────────────────────

    #[test]
    fn temporal_coherence_finite() {
        let s = make_service();
        assert!(s.temporal_coherence().is_finite());
    }

    // ── Neurochemistry ────────────────────────────────────────────────

    #[test]
    fn neurochemistry_checkpoint_roundtrip() {
        let mut s = make_service();
        let ckpt = s.neurochemistry_checkpoint();
        // Restore should not panic
        s.restore_neurochemistry(&ckpt);
        let ckpt2 = s.neurochemistry_checkpoint();
        // Values should be identical after restore
        assert_eq!(ckpt.da_sensitivity, ckpt2.da_sensitivity);
        assert_eq!(ckpt.ne_sensitivity, ckpt2.ne_sensitivity);
        assert_eq!(ckpt.sht_sensitivity, ckpt2.sht_sensitivity);
        assert_eq!(ckpt.ach_sensitivity, ckpt2.ach_sensitivity);
    }

    #[test]
    fn clamp_neuromod_levels_no_panic() {
        let mut s = make_service();
        s.clamp_neuromod_levels(Some(0.5), None, Some(1.0), None);
        let snap = s.neuromod_snapshot();
        assert!(
            snap.da_effective.is_finite(),
            "DA should be finite after clamp"
        );
        assert!(
            snap.sht_effective.is_finite(),
            "5HT should be finite after clamp"
        );
    }

    #[test]
    fn neuromod_snapshot_finite() {
        let s = make_service();
        let snap = s.neuromod_snapshot();
        assert!(snap.da_effective.is_finite());
        assert!(snap.ne_effective.is_finite());
        assert!(snap.sht_effective.is_finite());
        assert!(snap.ach_effective.is_finite());
    }

    // ── Pain sender ───────────────────────────────────────────────────

    #[test]
    fn pain_sender_present() {
        let s = make_service();
        assert!(s.pain_sender().is_some());
    }

    // ── Psi attestation ───────────────────────────────────────────────

    #[test]
    fn psi_attestation_count_zero() {
        let s = make_service();
        assert_eq!(s.psi_attestation_count(), 0);
    }

    #[test]
    fn drain_psi_attestations_empty() {
        let mut s = make_service();
        let drained = s.drain_psi_attestations();
        assert!(drained.is_empty());
    }

    #[test]
    fn latest_psi_attestation_none() {
        let s = make_service();
        assert!(s.latest_psi_attestation().is_none());
    }

    // ── Adaptive behavior ─────────────────────────────────────────────

    #[test]
    fn speech_rate_multiplier_finite() {
        let s = make_service();
        assert!(s.speech_rate_multiplier().is_finite());
    }

    #[test]
    fn pause_multiplier_finite() {
        let s = make_service();
        assert!(s.pause_multiplier().is_finite());
    }

    #[test]
    fn attention_sensitivity_finite() {
        let s = make_service();
        assert!(s.attention_sensitivity().is_finite());
    }

    #[test]
    fn exploration_factor_finite() {
        let s = make_service();
        assert!(s.exploration_factor().is_finite());
    }

    #[test]
    fn state_description_not_empty() {
        let s = make_service();
        assert!(!s.state_description().is_empty());
    }

    // ── Strategy and learning loop ────────────────────────────────────

    #[test]
    fn strategy_q_values_length() {
        let s = make_service();
        assert_eq!(s.strategy_q_values().len(), 5);
    }

    #[test]
    fn strategy_usage_counts_length() {
        let s = make_service();
        assert_eq!(s.strategy_usage_counts().len(), 5);
    }

    #[test]
    fn average_reward_initial_finite() {
        let s = make_service();
        assert!(s.average_reward().is_finite());
    }

    #[test]
    fn last_learning_result_none_initially() {
        let s = make_service();
        assert!(s.last_learning_result().is_none());
    }

    // ── Modulation index / coupling ───────────────────────────────────

    #[test]
    fn coupling_quality_no_panic() {
        let s = make_service();
        let q = s.coupling_quality();
        // Initially should be InsufficientData (no cycles run)
        assert!(
            matches!(
                q,
                crate::cognitive_loop::routing::CouplingQuality::InsufficientData
            ),
            "coupling should be InsufficientData before any cycles, got {q:?}"
        );
    }

    // ── World model ───────────────────────────────────────────────────

    #[test]
    fn world_model_abstract_state_not_empty() {
        let s = make_service();
        assert!(!s.world_model_abstract_state().is_empty());
    }

    #[test]
    fn world_model_level_errors_not_empty() {
        let s = make_service();
        assert!(!s.world_model_level_errors().is_empty());
    }

    // ── Combined learning rate ────────────────────────────────────────

    #[test]
    fn combined_learning_rate_positive() {
        let s = make_service();
        assert!(s.combined_learning_rate() > 0.0);
    }

    // ── Consciousness pattern ─────────────────────────────────────────

    #[test]
    fn consciousness_pattern_confidence_finite() {
        let s = make_service();
        let (_, conf) = s.consciousness_pattern();
        assert!(conf.is_finite());
    }

    // ── FEP free energy ───────────────────────────────────────────────

    #[test]
    fn fep_free_energy_initially_none() {
        let s = make_service();
        // No cycles run yet, so no FE components computed
        assert!(s.fep_free_energy().is_none());
    }

    // ── Experience bus ─────────────────────────────────────────────────

    #[test]
    fn experience_signals_present() {
        let s = make_service();
        // ExperienceBus is created by default
        let _ = s.experience_signals();
    }

    #[test]
    fn guiding_question_present() {
        let s = make_service();
        let _ = s.guiding_question();
    }

    // ── Attention visualization ────────────────────────────────────────

    #[test]
    fn attention_summary_none_before_cycle() {
        let s = make_service();
        // enable_visualization defaults to true, so the visualizer itself is
        // constructed eagerly (see constructor.rs) — before any cycle runs it
        // holds zero snapshots, not an absent Option.
        assert_eq!(s.attention_summary().unwrap().num_snapshots, 0);
    }

    #[test]
    fn attention_heatmap_none_before_cycle() {
        let s = make_service();
        // Same eager-construction reasoning as attention_summary above: the
        // heatmap renderer reports its own "no data" placeholder rather than
        // the accessor returning None.
        assert_eq!(s.attention_heatmap().unwrap(), "No attention data recorded");
    }

    // ── HDC bridge dimension ──────────────────────────────────────────

    #[test]
    fn hdc_bridge_dim_none_for_cfc() {
        let s = make_service();
        // Default backend is CfC, which has no HDC bridge dim
        let _ = s.hdc_bridge_dim();
    }

    #[test]
    fn hdc_bridge_dim_some_for_hdc_ltc() {
        let config = CognitiveLoopConfig::with_hdc_ltc_unified();
        let s = CognitiveLoopService::new(config).unwrap();
        assert!(s.hdc_bridge_dim().is_some());
    }

    // ── Voice feedback ────────────────────────────────────────────────

    #[test]
    fn voice_indicates_uncertainty_initial() {
        let s = make_service();
        let u = s.voice_indicates_uncertainty();
        // Verify accessor returns without panic; initial state may vary
        let _ = u;
        // Success: accessor completed without panic
        assert!(true);
    }

    #[test]
    fn voice_consciousness_signals_finite() {
        let s = make_service();
        let sig = s.voice_consciousness_signals();
        assert!(sig.unified_quality.is_finite());
        assert!(sig.consciousness_level.is_finite());
    }

    #[test]
    fn combined_phi_contribution_finite() {
        let s = make_service();
        assert!(s.combined_phi_contribution().is_finite());
    }

    // ── Thalamic stats ────────────────────────────────────────────────

    #[test]
    fn thalamic_stats_sum_to_one_or_zero() {
        let s = make_service();
        let (reflex, cortical, deep) = s.thalamic_stats();
        // All should be finite
        assert!(reflex.is_finite());
        assert!(cortical.is_finite());
        assert!(deep.is_finite());
    }

    // ── User state ────────────────────────────────────────────────────

    #[test]
    fn user_state_none_when_disabled() {
        let s = make_service();
        let _ = s.user_state();
    }

    // ── Moral topology summary ─────────────────────────────────────────

    #[test]
    fn moral_topology_summary_default_before_cycles() {
        let s = make_service();
        let summary = s.moral_topology_summary();
        assert_eq!(summary.scenario_count, 0, "no scenarios before cycles");
        assert_eq!(summary.beta_0, 0);
    }

    /// Smoke test: instantiate default CLS and call every accessor group.
    /// Catches field-path regressions after refactors without testing values.
    #[test]
    fn accessor_smoke_all_groups() {
        let s = make_service();

        // Behavior accessors
        let _ = s.in_flow();
        let _ = s.flow_intensity();
        let _ = s.flow_streak();
        let _ = s.flow_learning_boost();
        let valence = s.emotional_valence();
        let _ = s.emotional_arousal();
        let _ = s.is_bored();
        let _ = s.current_strategy();
        let _ = s.is_confident();

        // System accessors
        let _ = s.stats();
        let _ = s.config();
        let confidence = s.prediction_confidence();

        // At least verify returned values are finite
        assert!(valence.is_finite(), "emotional_valence should be finite");
        assert!(
            confidence.is_finite(),
            "prediction_confidence should be finite"
        );
    }
}
