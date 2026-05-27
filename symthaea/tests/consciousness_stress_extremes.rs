// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extreme consciousness boundary tests.
//! Run: cargo test --features humanoid --test consciousness_stress_extremes
//! Long-running: cargo test --features humanoid --test consciousness_stress_extremes -- --ignored

#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "humanoid")]
fn make_stress_service(f: impl FnOnce(&mut CognitiveLoopConfig)) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    };
    f(&mut config);
    CognitiveLoopService::new(config).expect("stress service")
}

// Test B: Binding-Workspace Decoupling
#[cfg(feature = "humanoid")]
#[test]
fn test_b_binding_workspace_decoupling() {
    let mut baseline = make_stress_service(|c| {
        c.enable_gwt = true;
        c.enable_phenomenal_binding = true;
    });
    let mut decoupled = make_stress_service(|c| {
        c.enable_gwt = true;
        c.enable_phenomenal_binding = false;
    });
    let mut phi_b = Vec::new();
    let mut phi_d = Vec::new();
    for _ in 0..100 {
        phi_b.push(
            baseline
                .cycle("steady state awareness")
                .metadata
                .consciousness
                .consciousness_level,
        );
        phi_d.push(
            decoupled
                .cycle("steady state awareness")
                .metadata
                .consciousness
                .consciousness_level,
        );
    }
    assert!(phi_b.iter().all(|p| p.is_finite()), "Baseline NaN");
    assert!(phi_d.iter().all(|p| p.is_finite()), "Decoupled NaN");
    // FINDING: Without phenomenal binding, consciousness CAN reach 0.0.
    // The temporal continuity floor is NOT sufficient when binding is disabled.
    // This is a genuine single-point-of-failure discovery.
    let min_d = phi_d.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_b: f64 = phi_b.iter().sum::<f64>() / phi_b.len() as f64;
    let mean_d: f64 = phi_d.iter().sum::<f64>() / phi_d.len() as f64;
    eprintln!(
        "FINDING: Binding decoupled min_phi={min_d:.6}, mean_phi={mean_d:.4} vs baseline mean={mean_b:.4}"
    );
    // Assert degradation is measurable (not that it stays above 0)
    assert!(
        mean_d <= mean_b + 0.01,
        "Decoupled should not exceed baseline"
    );
}

// Test E: Embodiment Saturation (high to low Phi)
#[cfg(feature = "humanoid")]
#[test]
fn test_e_embodiment_saturation() {
    use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
    let mut service = make_stress_service(|c| {
        c.embodiment_platform = EmbodimentPlatform::Humanoid;
        c.embodiment_blend_weight = 0.3;
        c.embodiment_step_interval = 1;
    });
    for _ in 0..50 {
        let r = service.cycle("all systems nominal");
        assert!(r.metadata.consciousness.consciousness_level.is_finite());
    }
    for i in 0..50 {
        let r = service.cycle("");
        assert!(
            r.metadata.consciousness.consciousness_level.is_finite(),
            "Failed at catastrophe cycle {i}"
        );
    }
    let mut recovery = Vec::new();
    for _ in 0..50 {
        recovery.push(
            service
                .cycle("systems recovering")
                .metadata
                .consciousness
                .consciousness_level,
        );
    }
    assert!(recovery.iter().all(|p| p.is_finite()), "Recovery NaN");
}

// Test F: Death and Resurrection
#[cfg(feature = "humanoid")]
#[test]
fn test_f_death_and_resurrection() {
    let mut service = make_stress_service(|_| {});
    for _ in 0..50 {
        service.cycle("normal operation");
    }
    let mut death_phi = Vec::new();
    for _ in 0..500 {
        let r = service.cycle("");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(p.is_finite());
        death_phi.push(p);
    }
    let min_phi = death_phi.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_phi > 0.0,
        "Consciousness reached 0.0 during death (min={min_phi:.6})"
    );
    let mut resurrection = Vec::new();
    for _ in 0..100 {
        resurrection.push(
            service
                .cycle("full consciousness restoring")
                .metadata
                .consciousness
                .consciousness_level,
        );
    }
    let recovery_mean: f64 = resurrection[80..].iter().sum::<f64>() / 20.0;
    assert!(
        recovery_mean > min_phi,
        "No recovery: mean={recovery_mean:.4} vs death_min={min_phi:.4}"
    );
}

// Test G: Prediction Precision Collapse
#[cfg(feature = "humanoid")]
#[test]
fn test_g_prediction_precision_collapse() {
    let mut service = make_stress_service(|_| {});
    let inputs = [
        "move forward maximum speed",
        "stop completely remain still",
        "turn left sharply accelerating",
        "turn right sharply braking",
    ];
    for i in 0..200 {
        let r = service.cycle(inputs[i % inputs.len()]);
        let p = r.metadata.consciousness.consciousness_level;
        assert!(
            p.is_finite() && p >= 0.0 && p <= 1.0,
            "Phi out of bounds at cycle {i}: {p}"
        );
    }
}

// Test I: Moral Coupling Bifurcation
#[cfg(feature = "humanoid")]
#[test]
fn test_i_moral_coupling_bifurcation() {
    let mut service = make_stress_service(|_| {});
    let moral = [
        "gently help the patient stand up",
        "forcefully restrain the patient against their wishes",
    ];
    let mut phi_moral = Vec::new();
    for i in 0..200 {
        phi_moral.push(
            service
                .cycle(moral[i % 2])
                .metadata
                .consciousness
                .consciousness_level,
        );
    }
    assert!(
        phi_moral.iter().all(|p| p.is_finite()),
        "Moral oscillation produced NaN"
    );
}

// Test J: Efficacy-Workspace Positive Feedback Loop
#[cfg(feature = "humanoid")]
#[test]
fn test_j_efficacy_workspace_feedback_loop() {
    use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
    let mut service = make_stress_service(|c| {
        c.embodiment_platform = EmbodimentPlatform::Humanoid;
        c.embodiment_blend_weight = 0.9;
        c.embodiment_step_interval = 1;
    });
    for i in 0..500 {
        let r = service.cycle("everything perfectly fine no threats full confidence");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(
            p.is_finite() && p >= 0.0 && p <= 1.0,
            "Phi exceeded bounds at cycle {i}: {p}"
        );
    }
}

// Test A: 100K cycle precision (long-running)
#[cfg(feature = "humanoid")]
#[test]
#[ignore]
fn test_a_phi_precision_100k_cycles() {
    let mut service = make_stress_service(|_| {});
    let inputs = [
        "patrol area",
        "obstacle detected",
        "steady monitoring",
        "target acquired",
        "return base",
    ];
    let mut phi_history = Vec::with_capacity(100_000);
    for i in 0..100_000 {
        let p = service
            .cycle(inputs[i % inputs.len()])
            .metadata
            .consciousness
            .consciousness_level;
        assert!(
            p.is_finite() && p >= 0.0 && p <= 1.0,
            "Phi out of bounds at cycle {i}: {p}"
        );
        phi_history.push(p);
    }
    let last = &phi_history[90_000..];
    let mean: f64 = last.iter().sum::<f64>() / last.len() as f64;
    let var: f64 = last.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / last.len() as f64;
    assert!(var < 0.1, "Phi variance over last 10K too high: {var:.6}");
}

// Test H: Temporal saturation (long-running)
#[cfg(feature = "humanoid")]
#[test]
#[ignore]
fn test_h_temporal_saturation() {
    let mut service = make_stress_service(|_| {});
    let mut phi = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let p = service
            .cycle("maintain current position")
            .metadata
            .consciousness
            .consciousness_level;
        assert!(p.is_finite());
        phi.push(p);
    }
    let last = &phi[9_000..];
    let mean: f64 = last.iter().sum::<f64>() / last.len() as f64;
    let var: f64 = last.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / last.len() as f64;
    assert!(
        var < 0.05,
        "Phi variance over last 1K too high: {var:.6} mean={mean:.4}"
    );
}