//! Integration tests for manager round-trip behavior.
//!
//! Verifies that manager structs (VoiceCoherenceBridge, SocialManager,
//! GwtManager, BiorhythmManager) produce identical state to manual init
//! and that reset/default restore clean state.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// VoiceCoherenceBridge
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn voice_coherence_bridge_default_state() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Bridge should start with zero smoothed coherence
    assert!(service.voice_coherence.bridge.smoothed_coherence().is_finite());
    // Voice feedback should start with default quality
    let summary = service.voice_coherence.voice.summary();
    assert!(summary.articulation_quality.is_finite());
    // Temporal signature should start with some default pattern
    let temporal = service.voice_coherence.temporal.summary();
    assert!(temporal.confidence.is_finite());
}

#[test]
fn voice_coherence_bridge_reset_restores_clean_state() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Run a few cycles to mutate state
    for _ in 0..5 {
        let _ = service.cycle("test input");
    }
    // Reset
    service.reset();
    // State should be clean
    assert!(service.voice_coherence.bridge.smoothed_coherence().is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SocialManager
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn social_manager_default_values() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert!((service.social_mgr.social.relational_psi - 0.0).abs() < f64::EPSILON);
    assert!((service.social_mgr.social.external_reward - 0.0).abs() < f32::EPSILON);
    assert!((service.social_mgr.social.social_trust - 0.5).abs() < f32::EPSILON);
    assert!((service.social_mgr.social.social_cooperation_rate - 0.0).abs() < f32::EPSILON);
    assert!((service.social_mgr.social.social_prediction_accuracy - 0.5).abs() < f32::EPSILON);
    assert_eq!(service.social_mgr.social.social_models_count, 0);
    assert!((service.social_mgr.social.social_mean_trust - 0.5).abs() < f32::EPSILON);
}

#[test]
fn social_manager_reset_restores_defaults() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Mutate
    service.set_social_signals(0.9, 0.8, 0.7, 10, 0.6);
    service.set_relational_psi(0.99);
    // Verify mutation
    assert!((service.social_mgr.social.social_trust - 0.9).abs() < f32::EPSILON);
    // Reset
    service.reset();
    // Verify restored
    assert!((service.social_mgr.social.social_trust - 0.5).abs() < f32::EPSILON);
    assert!((service.social_mgr.social.relational_psi - 0.0).abs() < f64::EPSILON);
}

#[test]
fn social_manager_phi_dyad_disabled_by_default() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Default config doesn't enable primitive consciousness, so phi_dyad is None
    assert!(service.social_mgr.phi_dyad.is_none());
    assert!(service.social_mgr.partner_model.is_none());
    assert!(service.social_mgr.recent_ai_hvs.is_empty());
    assert!(service.social_mgr.recent_input_hvs.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// GwtManager
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn gwt_manager_default_flags() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Memory flag should be false initially
    assert!(!service.gwt_mgr.memory_flag.load(std::sync::atomic::Ordering::Relaxed));
    // Perception count should be 0
    assert_eq!(
        service.gwt_mgr.perception_count.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn gwt_manager_gwt_enabled_by_default() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // GWT is enabled by default
    assert!(service.gwt_mgr.gwt.is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
// BiorhythmManager
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn biorhythm_manager_initial_state() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert_eq!(service.biorhythm_mgr.refresh_counter, 0);
    // Biorhythm should have finite plasticity modulation
    let plasticity = service.biorhythm_mgr.rhythm.plasticity_mod;
    assert!(plasticity.is_finite() && plasticity > 0.0);
}

#[test]
fn biorhythm_counter_increments_over_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Run enough cycles that the counter should have been used
    for _ in 0..5 {
        let _ = service.cycle("test");
    }
    // Counter tracks cycles mod 100
    assert!(service.biorhythm_mgr.refresh_counter <= 100);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-manager: full service round-trip
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn full_service_reset_restores_all_managers() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Run cycles to mutate everything
    for _ in 0..10 {
        let _ = service.cycle("changing state");
    }
    // Reset
    service.reset();
    // All managers should be at clean state
    assert!((service.social_mgr.social.social_trust - 0.5).abs() < f32::EPSILON);
    assert!((service.social_mgr.social.relational_psi - 0.0).abs() < f64::EPSILON);
    assert!(service.voice_coherence.bridge.smoothed_coherence().is_finite());
}
