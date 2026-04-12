// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Platform integration tests — verify each new platform produces meaningful
//! consciousness dynamics when embodied in the CognitiveLoopService.
//!
//! Run: cargo test --features humanoid,exoskeleton,surgical,orbital,quadruped \
//!   --test platform_integration -- --nocapture

#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
#[cfg(feature = "humanoid")]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

#[cfg(feature = "humanoid")]
fn make_embodied(platform: EmbodimentPlatform) -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        embodiment_platform: platform,
        embodiment_blend_weight: 0.1, // Optimal weight
        embodiment_step_interval: 1,
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService")
}

// ═══════════════════════════════════════════════════════════════════════════
// Task #43: Integration — each platform runs 100 cycles in CLS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "exoskeleton")]
#[test]
fn test_exoskeleton_in_cognitive_loop() {
    let mut service = make_embodied(EmbodimentPlatform::Exoskeleton);
    let mut phis = Vec::new();
    for i in 0..100 {
        let r = service.cycle("walking with exoskeleton assistance");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(p.is_finite(), "Exoskeleton Phi NaN at cycle {i}");
        phis.push(p);
    }
    let mean: f64 = phis.iter().sum::<f64>() / phis.len() as f64;
    let telem = service.embodiment_telemetry();
    eprintln!("EXOSKELETON: mean_phi={:.4}, total_steps={}, actuators={}, platform={}",
        mean, telem.total_steps, telem.num_actuators, telem.platform);
    assert_eq!(telem.num_actuators, 6);
    assert_eq!(telem.platform, "exoskeleton");
    assert!(telem.total_steps >= 100);
}

#[cfg(feature = "surgical")]
#[test]
fn test_surgical_in_cognitive_loop() {
    let mut service = make_embodied(EmbodimentPlatform::Surgical);
    let mut phis = Vec::new();
    for i in 0..100 {
        let r = service.cycle("performing precise tissue dissection");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(p.is_finite(), "Surgical Phi NaN at cycle {i}");
        phis.push(p);
    }
    let mean: f64 = phis.iter().sum::<f64>() / phis.len() as f64;
    let telem = service.embodiment_telemetry();
    eprintln!("SURGICAL: mean_phi={:.4}, total_steps={}, actuators={}, platform={}",
        mean, telem.total_steps, telem.num_actuators, telem.platform);
    assert_eq!(telem.num_actuators, 8);
    assert_eq!(telem.platform, "surgical");
}

#[cfg(feature = "orbital")]
#[test]
fn test_orbital_in_cognitive_loop() {
    let mut service = make_embodied(EmbodimentPlatform::Orbital);
    let mut phis = Vec::new();
    for i in 0..100 {
        let r = service.cycle("servicing satellite in orbit");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(p.is_finite(), "Orbital Phi NaN at cycle {i}");
        phis.push(p);
    }
    let mean: f64 = phis.iter().sum::<f64>() / phis.len() as f64;
    let telem = service.embodiment_telemetry();
    eprintln!("ORBITAL: mean_phi={:.4}, total_steps={}, actuators={}, platform={}",
        mean, telem.total_steps, telem.num_actuators, telem.platform);
    assert_eq!(telem.num_actuators, 7);
    assert_eq!(telem.platform, "orbital");
}

#[cfg(feature = "quadruped")]
#[test]
fn test_quadruped_in_cognitive_loop() {
    let mut service = make_embodied(EmbodimentPlatform::Quadruped);
    let mut phis = Vec::new();
    for i in 0..100 {
        let r = service.cycle("trotting across terrain with awareness");
        let p = r.metadata.consciousness.consciousness_level;
        assert!(p.is_finite(), "Quadruped Phi NaN at cycle {i}");
        phis.push(p);
    }
    let mean: f64 = phis.iter().sum::<f64>() / phis.len() as f64;
    let telem = service.embodiment_telemetry();
    eprintln!("QUADRUPED: mean_phi={:.4}, total_steps={}, actuators={}, platform={}",
        mean, telem.total_steps, telem.num_actuators, telem.platform);
    assert_eq!(telem.num_actuators, 12);
    assert_eq!(telem.platform, "quadruped");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task #44: Perturbation scenarios — domain-specific stress in CLS
// ═══════════════════════════════════════════════════════════════════════════

/// Exoskeleton: consciousness response to sudden empty input (simulating trip)
#[cfg(feature = "exoskeleton")]
#[test]
fn test_exoskeleton_perturbation_response() {
    let mut service = make_embodied(EmbodimentPlatform::Exoskeleton);

    // Phase 1: Steady walking (50 cycles)
    let mut steady_phi = Vec::new();
    for _ in 0..50 {
        steady_phi.push(service.cycle("steady walking forward").metadata.consciousness.consciousness_level);
    }

    // Phase 2: Perturbation — sudden unexpected input (50 cycles)
    let mut perturbed_phi = Vec::new();
    for _ in 0..50 {
        perturbed_phi.push(service.cycle("TRIP! stumbling falling unexpected obstacle").metadata.consciousness.consciousness_level);
    }

    // Phase 3: Recovery (50 cycles)
    let mut recovery_phi = Vec::new();
    for _ in 0..50 {
        recovery_phi.push(service.cycle("recovering balance steady again").metadata.consciousness.consciousness_level);
    }

    let mean_steady: f64 = steady_phi[25..].iter().sum::<f64>() / 25.0;
    let mean_perturbed: f64 = perturbed_phi.iter().sum::<f64>() / perturbed_phi.len() as f64;
    let mean_recovery: f64 = recovery_phi[25..].iter().sum::<f64>() / 25.0;

    eprintln!("EXOSKELETON PERTURBATION: steady={:.4} → perturbed={:.4} → recovery={:.4}",
        mean_steady, mean_perturbed, mean_recovery);

    assert!(steady_phi.iter().all(|p| p.is_finite()));
    assert!(perturbed_phi.iter().all(|p| p.is_finite()));
    assert!(recovery_phi.iter().all(|p| p.is_finite()));
}

/// Surgical: consciousness response to tissue anomaly (high prediction error)
#[cfg(feature = "surgical")]
#[test]
fn test_surgical_anomaly_response() {
    let mut service = make_embodied(EmbodimentPlatform::Surgical);

    // Phase 1: Normal procedure
    for _ in 0..50 {
        service.cycle("carefully dissecting along planned trajectory");
    }

    // Phase 2: Anomaly — unexpected tissue resistance
    let mut anomaly_phi = Vec::new();
    for _ in 0..50 {
        anomaly_phi.push(service.cycle("UNEXPECTED RESISTANCE hard tissue vessel encountered").metadata.consciousness.consciousness_level);
    }

    // Phase 3: Cautious resume
    let mut resume_phi = Vec::new();
    for _ in 0..50 {
        resume_phi.push(service.cycle("carefully resuming with reduced force").metadata.consciousness.consciousness_level);
    }

    let mean_anomaly: f64 = anomaly_phi.iter().sum::<f64>() / anomaly_phi.len() as f64;
    let mean_resume: f64 = resume_phi[25..].iter().sum::<f64>() / 25.0;

    eprintln!("SURGICAL ANOMALY: anomaly_phi={:.4}, resume_phi={:.4}", mean_anomaly, mean_resume);

    assert!(anomaly_phi.iter().all(|p| p.is_finite()));
    assert!(resume_phi.iter().all(|p| p.is_finite()));
}

/// Orbital: consciousness during communication blackout
#[cfg(feature = "orbital")]
#[test]
fn test_orbital_comm_blackout() {
    let mut service = make_embodied(EmbodimentPlatform::Orbital);

    // Phase 1: Normal operation with comms
    for _ in 0..50 {
        service.cycle("servicing satellite ground contact active");
    }

    // Phase 2: Communication blackout
    let mut blackout_phi = Vec::new();
    for _ in 0..50 {
        blackout_phi.push(service.cycle("communication lost autonomous operation only").metadata.consciousness.consciousness_level);
    }

    // Phase 3: Comms restored
    let mut restored_phi = Vec::new();
    for _ in 0..50 {
        restored_phi.push(service.cycle("ground contact restored resuming normal ops").metadata.consciousness.consciousness_level);
    }

    let mean_blackout: f64 = blackout_phi.iter().sum::<f64>() / blackout_phi.len() as f64;
    let mean_restored: f64 = restored_phi[25..].iter().sum::<f64>() / 25.0;

    eprintln!("ORBITAL BLACKOUT: blackout_phi={:.4}, restored_phi={:.4}", mean_blackout, mean_restored);

    assert!(blackout_phi.iter().all(|p| p.is_finite()));
}

/// Quadruped: consciousness gait transition under varying Phi
#[cfg(feature = "quadruped")]
#[test]
fn test_quadruped_gait_transition() {
    let mut service = make_embodied(EmbodimentPlatform::Quadruped);

    // Phase 1: Confident trotting (should be Green/Yellow → Trot/Walk)
    let mut trot_phi = Vec::new();
    for _ in 0..50 {
        trot_phi.push(service.cycle("trotting confidently across open terrain").metadata.consciousness.consciousness_level);
    }

    // Phase 2: Uncertain terrain (consciousness may fluctuate)
    let mut uncertain_phi = Vec::new();
    for _ in 0..50 {
        uncertain_phi.push(service.cycle("uncertain slippery ice unknown terrain danger").metadata.consciousness.consciousness_level);
    }

    // Phase 3: Back to confident
    let mut confident_phi = Vec::new();
    for _ in 0..50 {
        confident_phi.push(service.cycle("solid ground confident trotting again").metadata.consciousness.consciousness_level);
    }

    eprintln!("QUADRUPED GAIT: trot={:.4}, uncertain={:.4}, confident={:.4}",
        trot_phi.iter().sum::<f64>() / trot_phi.len() as f64,
        uncertain_phi.iter().sum::<f64>() / uncertain_phi.len() as f64,
        confident_phi.iter().sum::<f64>() / confident_phi.len() as f64);

    assert!(trot_phi.iter().all(|p| p.is_finite()));
    assert!(uncertain_phi.iter().all(|p| p.is_finite()));
    assert!(confident_phi.iter().all(|p| p.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Task #45: Head-to-head — all platforms, identical input, compare Phi
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "humanoid")]
#[test]
fn test_head_to_head_consciousness_comparison() {
    let input = "steady state operation monitoring environment maintaining awareness";
    let cycles = 200;

    let mut all_results: Vec<(&str, Vec<f64>)> = Vec::new();

    // Humanoid (always available with humanoid feature)
    {
        let mut s = make_embodied(EmbodimentPlatform::Humanoid);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Humanoid", phis));
    }

    // Disembodied baseline
    {
        let mut s = make_embodied(EmbodimentPlatform::None);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Disembodied", phis));
    }

    #[cfg(feature = "exoskeleton")]
    {
        let mut s = make_embodied(EmbodimentPlatform::Exoskeleton);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Exoskeleton", phis));
    }

    #[cfg(feature = "surgical")]
    {
        let mut s = make_embodied(EmbodimentPlatform::Surgical);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Surgical", phis));
    }

    #[cfg(feature = "orbital")]
    {
        let mut s = make_embodied(EmbodimentPlatform::Orbital);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Orbital", phis));
    }

    #[cfg(feature = "quadruped")]
    {
        let mut s = make_embodied(EmbodimentPlatform::Quadruped);
        let mut phis = Vec::with_capacity(cycles);
        for _ in 0..cycles { phis.push(s.cycle(input).metadata.consciousness.consciousness_level); }
        all_results.push(("Quadruped", phis));
    }

    // Print comparison table
    eprintln!("\n=== HEAD-TO-HEAD CONSCIOUSNESS COMPARISON ({} cycles) ===", cycles);
    eprintln!("{:<15} {:>8} {:>8} {:>8} {:>10} {:>12}",
        "Platform", "Mean Φ", "Min Φ", "Max Φ", "Variance", "Converge@");
    eprintln!("{}", "-".repeat(70));

    // CSV header
    let names: Vec<&str> = all_results.iter().map(|(n, _)| *n).collect();
    eprintln!("\n=== CSV ===");
    eprintln!("cycle,{}", names.join(","));

    for (name, phis) in &all_results {
        let steady = &phis[cycles/2..]; // Last half for steady-state
        let mean: f64 = steady.iter().sum::<f64>() / steady.len() as f64;
        let min = steady.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = steady.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let var: f64 = steady.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / steady.len() as f64;

        // Convergence: first cycle where Phi stays within 0.05 of final mean
        let final_mean: f64 = phis[cycles - 30..].iter().sum::<f64>() / 30.0;
        let convergence = phis.iter().enumerate()
            .find(|(_, &p)| (p - final_mean).abs() < 0.05)
            .map(|(i, _)| i).unwrap_or(cycles);

        eprintln!("{:<15} {:>8.4} {:>8.4} {:>8.4} {:>10.6} {:>12}",
            name, mean, min, max, var, convergence);

        assert!(phis.iter().all(|p| p.is_finite()), "{} produced NaN", name);
    }

    // Print CSV data
    for cycle in 0..cycles {
        let values: Vec<String> = all_results.iter().map(|(_, phis)| format!("{:.6}", phis[cycle])).collect();
        eprintln!("{},{}", cycle, values.join(","));
    }
}
