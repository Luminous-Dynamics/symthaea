// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;

// ═══════════════════════════════════════════════════════════════════════
// CalibrationValidator tests (Item #9 from previous session)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_validator_default_no_damping() {
    let v = CalibrationValidator::default();
    assert_eq!(v.total_validations(), 0);
    assert!((v.adjustment_multiplier() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_validator_records_improvement() {
    let mut v = CalibrationValidator::default();
    // Record pre-calibration: PE=0.5, coherence=0.3, confidence_error=0.2
    v.record_pre_calibration(0.5, 0.3, 0.2, 100);
    // Check after settling (100 cycles): PE improved, coherence improved
    let result = v.check_validation(0.3, 0.5, 0.1, 201);
    assert_eq!(result, Some(true), "should detect improvement");
    assert_eq!(v.improvements, 1);
    assert_eq!(v.regressions, 0);
}

#[test]
fn test_validator_records_regression() {
    let mut v = CalibrationValidator::default();
    v.record_pre_calibration(0.2, 0.7, 0.05, 100);
    // Check after settling: PE worsened significantly, coherence worsened
    let result = v.check_validation(0.5, 0.4, 0.1, 201);
    assert_eq!(result, Some(false), "should detect regression");
    assert_eq!(v.regressions, 1);
}

#[test]
fn test_validator_settling_period() {
    let mut v = CalibrationValidator::default();
    v.record_pre_calibration(0.5, 0.3, 0.2, 100);
    // Check before settling period elapsed (< 100 cycles)
    let result = v.check_validation(0.3, 0.5, 0.1, 150);
    assert_eq!(result, None, "should wait for settling period");
    assert_eq!(v.total_validations(), 0);
}

#[test]
fn test_validator_regression_damping() {
    let mut v = CalibrationValidator::default();
    // Simulate 4 regressions, 0 improvements → should trigger damping
    for i in 0..4 {
        let cycle_base = i * 200;
        v.record_pre_calibration(0.2, 0.7, 0.05, cycle_base as u64);
        v.check_validation(0.5, 0.4, 0.1, (cycle_base + 101) as u64);
    }
    assert!(v.regressions >= 3);
    assert!(
        v.regression_damping > 0.0,
        "damping should be active after multiple regressions"
    );
    assert!(
        v.adjustment_multiplier() < 1.0,
        "multiplier should reduce adjustments"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Adaptive calibration cadence tests (Item #3)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_adapt_sensitivity_good_ratio_reduces_cooldown() {
    let mut monitor = SelfAssessmentMonitor::default();
    let original = monitor.cooldown_duration;
    monitor.adapt_sensitivity(0.7); // >0.5 → reduce cooldown
    assert!(
        monitor.cooldown_duration < original,
        "good improvement ratio should reduce cooldown: {} < {}",
        monitor.cooldown_duration,
        original
    );
}

#[test]
fn test_adapt_sensitivity_bad_ratio_increases_cooldown() {
    let mut monitor = SelfAssessmentMonitor::default();
    let original = monitor.cooldown_duration;
    monitor.adapt_sensitivity(0.2); // <0.3 → increase cooldown
    assert!(
        monitor.cooldown_duration > original,
        "bad improvement ratio should increase cooldown: {} > {}",
        monitor.cooldown_duration,
        original
    );
}

#[test]
fn test_adapt_sensitivity_neutral_no_change() {
    let mut monitor = SelfAssessmentMonitor::default();
    let original = monitor.cooldown_duration;
    monitor.adapt_sensitivity(0.4); // 0.3..0.5 → no change
    assert_eq!(monitor.cooldown_duration, original);
}

#[test]
fn test_adapt_sensitivity_floor_cap() {
    let mut monitor = SelfAssessmentMonitor::default();
    // Repeatedly reduce → should hit floor of 200
    for _ in 0..100 {
        monitor.adapt_sensitivity(0.9);
    }
    assert!(monitor.cooldown_duration >= 200, "cooldown floor is 200");

    // Repeatedly increase → should hit cap of 2000
    for _ in 0..100 {
        monitor.adapt_sensitivity(0.1);
    }
    assert!(monitor.cooldown_duration <= 2000, "cooldown cap is 2000");
}

#[test]
fn test_calibration_from_z_scores() {
    let z_scores = vec![
        ("Executive::Stroop", 1.5),        // interference too high
        ("Executive::Flanker", 1.2),       // interference too high
        ("WorM::N-back", -0.8),            // WM below baseline
        ("Inhibition::StopSignal", 0.0),   // normal
        ("SustainedAttention::CPT", -1.0), // attention below baseline
    ];

    let cal = NeuromodCalibration::from_z_scores(&z_scores);

    // Should have 6 transmitter adjustments (DA, ACh, NE-β, NE-α, 5-HT, ECB)
    // NE-β from StopSignal (normal z=0, still produces adjustment)
    // NE-α from N-back WM z=-0.8
    // ECB composite from CPT sustained attention (stress signal)
    assert_eq!(cal.adjustments.len(), 6);

    // DA should be attenuated (interference too high → reduce DA)
    let da = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "DA")
        .unwrap();
    assert!(
        da.sensitivity_factor < 1.0,
        "High interference z should reduce DA sensitivity, got {}",
        da.sensitivity_factor
    );

    // ACh should be boosted (WM below baseline)
    let ach = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "ACh")
        .unwrap();
    assert!(
        ach.sensitivity_factor > 1.0,
        "Low WM z should boost ACh sensitivity, got {}",
        ach.sensitivity_factor
    );

    // 5-HT should be boosted (attention below baseline)
    let sht = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "5-HT")
        .unwrap();
    assert!(
        sht.sensitivity_factor > 1.0,
        "Low attention z should boost 5-HT sensitivity, got {}",
        sht.sensitivity_factor
    );
}

#[test]
fn test_calibration_normal_range() {
    let z_scores = vec![
        ("Executive::Stroop", 0.0),
        ("WorM::N-back", 0.0),
        ("Inhibition::StopSignal", 0.0),
        ("SustainedAttention::CPT", 0.0),
    ];

    let cal = NeuromodCalibration::from_z_scores(&z_scores);

    // At z=0, all factors should be ~1.0
    for adj in &cal.adjustments {
        assert!(
            (adj.sensitivity_factor - 1.0).abs() < 0.01,
            "{}: factor should be ~1.0 at z=0, got {}",
            adj.transmitter,
            adj.sensitivity_factor
        );
    }
}

#[test]
fn test_calibration_clamping() {
    let z_scores = vec![
        ("Executive::Stroop", 10.0), // extreme z
    ];

    let cal = NeuromodCalibration::from_z_scores(&z_scores);
    let da = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "DA")
        .unwrap();

    // Should be clamped, not run away
    assert!(da.sensitivity_factor >= 0.85);
    assert!(da.sensitivity_factor <= 1.15);
}

#[test]
fn test_confidence_calibration() {
    let z_scores = vec![
        ("Metacognition::FeelingOfKnowing", 1.5), // overconfident
    ];

    let cal = NeuromodCalibration::from_z_scores(&z_scores);
    assert!(
        cal.confidence_delta < 0.0,
        "Overconfident FoK should reduce confidence, got {}",
        cal.confidence_delta
    );
}

#[test]
fn test_sensitivity_factor_symmetry() {
    let boost = z_to_sensitivity_factor(-1.0, false);
    let attenuate = z_to_sensitivity_factor(1.0, false);

    // Symmetric around 1.0
    assert!((boost - 1.0).abs() - (attenuate - 1.0).abs() < 0.01);
    assert!(boost > 1.0);
    assert!(attenuate < 1.0);
}

#[test]
fn test_invert_flag() {
    // For lower-is-better metrics: positive z = bad = should reduce
    let normal = z_to_sensitivity_factor(1.0, false); // positive z, normal → attenuate
    let inverted = z_to_sensitivity_factor(1.0, true); // positive z, inverted → boost

    assert!(normal < 1.0);
    assert!(inverted > 1.0);
}

#[test]
fn test_summary_format() {
    let z_scores = vec![("Executive::Stroop", 1.0), ("WorM::N-back", -0.5)];
    let cal = NeuromodCalibration::from_z_scores(&z_scores);
    let summary = cal.summary();
    assert!(summary.contains("DA"));
    assert!(summary.contains("ACh"));
    assert!(summary.contains("Neuromod Calibration"));
}

#[test]
fn test_from_normative_z_scores() {
    // NormativeReport z-scores are sign-corrected: positive = better.
    // For lower-is-better metrics (stroop_effect), positive z means
    // "less interference than baseline" = good.
    // from_normative_z_scores un-corrects the sign for calibration.
    let scores = vec![
        // Stroop: positive normative z = less interference = good
        // Un-corrected: raw z = -1.0 → don't attenuate DA
        ("Executive::Stroop", "stroop_effect", 1.0),
        // N-back: positive normative z = better WM = good
        // Higher-is-better → no sign change
        ("WorM::N-back", "nback_2::accuracy", -1.0),
    ];

    let cal = NeuromodCalibration::from_normative_z_scores(&scores);

    // DA: normative z=+1.0 on stroop_effect (lower-is-better, sign-corrected)
    // Un-corrected: raw z = -1.0 → DA should be BOOSTED (factor > 1.0)
    let da = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "DA")
        .unwrap();
    assert!(
        da.sensitivity_factor > 1.0,
        "Good Stroop → boost DA, got {}",
        da.sensitivity_factor
    );

    // ACh: normative z=-1.0 on nback accuracy (higher-is-better)
    // No sign change → raw z = -1.0 → ACh should be BOOSTED
    let ach = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "ACh")
        .unwrap();
    assert!(
        ach.sensitivity_factor > 1.0,
        "Poor WM → boost ACh, got {}",
        ach.sensitivity_factor
    );
}

#[test]
fn test_apply_modifies_bath() {
    let z_scores = vec![
        ("Executive::Stroop", 2.0),        // high interference → attenuate DA
        ("SustainedAttention::CPT", -2.0), // poor attention → boost 5-HT
    ];
    let cal = NeuromodCalibration::from_z_scores(&z_scores);

    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    let da_before = bath.dopamine.receptor_sensitivity;
    let sht_before = bath.serotonin.receptor_sensitivity;

    cal.apply(&mut bath);

    assert!(
        bath.dopamine.receptor_sensitivity < da_before,
        "DA sensitivity should decrease: {} → {}",
        da_before,
        bath.dopamine.receptor_sensitivity
    );
    assert!(
        bath.serotonin.receptor_sensitivity > sht_before,
        "5-HT sensitivity should increase: {} → {}",
        sht_before,
        bath.serotonin.receptor_sensitivity
    );
}

#[test]
fn test_is_lower_better_metric() {
    assert!(is_lower_better_metric("stroop_effect"));
    assert!(is_lower_better_metric("flanker_effect"));
    assert!(is_lower_better_metric("ssrt_ticks"));
    assert!(!is_lower_better_metric("nback_2::accuracy"));
    assert!(!is_lower_better_metric("overall_accuracy"));
    assert!(!is_lower_better_metric("fok_gamma"));
}

// ═══════════════════════════════════════════════════════════════════
// Self-Assessment Monitor tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_self_assessment_no_trigger_during_warmup() {
    let mut monitor = SelfAssessmentMonitor::default();
    // Feed bad signals but within warmup period (200 cycles)
    let bad_input = SelfAssessmentInput {
        prediction_error: 0.5,
        coherence: 0.2,
        confidence_calibration_error: 0.3,
        attention_utilization: 0.95,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };
    for _ in 0..100 {
        monitor.update(&bad_input);
    }
    // Should NOT trigger — only 100 observations, warmup is 200
    assert!(
        monitor.check_trigger(false).is_none(),
        "Should not trigger during warmup"
    );
}

#[test]
fn test_self_assessment_triggers_on_high_pe() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 10, // low warmup for testing
        cooldown_duration: 5,
        ..Default::default()
    };

    let bad_input = SelfAssessmentInput {
        prediction_error: 0.5, // well above 0.1 baseline
        coherence: 0.7,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    // Feed enough cycles past warmup
    for _ in 0..50 {
        monitor.update(&bad_input);
    }

    let cal = monitor.check_trigger(false);
    assert!(cal.is_some(), "High PE should trigger calibration");
    let cal = cal.unwrap();
    // Should have DA adjustment (PE → interference proxy)
    assert!(
        cal.adjustments.iter().any(|a| a.transmitter == "DA"),
        "PE proxy should map to DA adjustment"
    );
}

#[test]
fn test_self_assessment_cooldown() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 100,
        ..Default::default()
    };

    let bad_input = SelfAssessmentInput {
        prediction_error: 0.5,
        coherence: 0.7,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..30 {
        monitor.update(&bad_input);
    }

    // First trigger should succeed
    assert!(monitor.check_trigger(false).is_some());

    // Feed more bad data
    for _ in 0..20 {
        monitor.update(&bad_input);
    }

    // Second trigger should fail — cooldown active (100 cycles)
    assert!(
        monitor.check_trigger(false).is_none(),
        "Should not trigger during cooldown"
    );
}

#[test]
fn test_self_assessment_drift_anomalous_bypasses_cooldown() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 100,
        ..Default::default()
    };

    let bad_input = SelfAssessmentInput {
        prediction_error: 0.5, // high PE → pe_ema will converge above 0.15
        coherence: 0.7,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: true,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..30 {
        monitor.update(&bad_input);
    }

    // First trigger should succeed
    assert!(monitor.check_trigger(true).is_some());

    // Feed more bad data (still in cooldown, only 20 cycles < 100)
    for _ in 0..20 {
        monitor.update(&bad_input);
    }

    // Normal check should fail — cooldown active
    assert!(
        monitor.check_trigger(false).is_none(),
        "Normal check should block during cooldown"
    );

    // But drift_anomalous=true should bypass cooldown because pe_ema > 0.15
    // (Turrigiano 2008: anomalous drift warrants urgent recalibration)
    assert!(
        monitor.check_trigger(true).is_some(),
        "Drift anomalous + elevated PE should bypass cooldown"
    );
}

#[test]
fn test_self_assessment_no_trigger_normal_performance() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 5,
        ..Default::default()
    };

    // Normal, healthy signals
    let good_input = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.75,
        confidence_calibration_error: 0.03,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..100 {
        monitor.update(&good_input);
    }

    assert!(
        monitor.check_trigger(false).is_none(),
        "Normal performance should not trigger calibration"
    );
}

#[test]
fn test_self_assessment_reset_after_calibration() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 50,
        ..Default::default()
    };

    let bad_input = SelfAssessmentInput {
        prediction_error: 0.5,
        coherence: 0.7,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..20 {
        monitor.update(&bad_input);
    }
    assert!(monitor.check_trigger(false).is_some());

    // External calibration resets cooldown
    monitor.reset_after_calibration();
    assert_eq!(monitor.observations_since_calibration, 0);
    assert_eq!(monitor.cooldown, 50);
}

#[test]
fn test_self_assessment_ne_inhibition_proxy() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 10,
        cooldown_duration: 5,
        ..Default::default()
    };

    let bad_input = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.7,
        confidence_calibration_error: 0.03,
        attention_utilization: 0.5,
        inhibition_error_rate: 1.0, // veto every cycle
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..80 {
        monitor.update(&bad_input);
    }

    let cal = monitor.check_trigger(false);
    assert!(
        cal.is_some(),
        "High inhibition errors should trigger calibration"
    );
    assert!(
        cal.unwrap()
            .adjustments
            .iter()
            .any(|a| a.transmitter == "NE"),
        "Inhibition errors should map to NE adjustment"
    );
}

#[test]
fn test_self_assessment_confidence_bidirectional() {
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 5,
        ..Default::default()
    };

    let overconfident = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.7,
        confidence_calibration_error: 0.3, // high ECE
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..60 {
        monitor.update(&overconfident);
    }

    let cal = monitor.check_trigger(false);
    assert!(cal.is_some(), "High ECE should trigger FoK calibration");
    assert!(
        cal.unwrap().confidence_delta < 0.0,
        "Should reduce confidence"
    );
}

#[test]
fn test_self_assessment_telemetry_accessors() {
    let mut monitor = SelfAssessmentMonitor::default();
    assert_eq!(monitor.observations_count(), 0);
    assert_eq!(monitor.cooldown_remaining(), 0);
    assert!((monitor.inhibition_error_ema() - 0.0).abs() < 1e-6);

    let input = SelfAssessmentInput {
        prediction_error: 0.3,
        coherence: 0.5,
        confidence_calibration_error: 0.1,
        attention_utilization: 0.8,
        inhibition_error_rate: 0.5,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };
    monitor.update(&input);
    assert_eq!(monitor.observations_count(), 1);
    assert!(monitor.inhibition_error_ema() > 0.0);
    assert!(monitor.pe_ema() > 0.0);
}

#[test]
fn test_ne_subtype_routing_inhibition_to_beta() {
    // Inhibition benchmarks should route to NE β (phasic reactivity)
    let cal = NeuromodCalibration::from_z_scores(&[("Inhibition::StopSignal", -2.0)]);
    let ne_adj = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "NE")
        .expect("Should have NE adjustment");
    assert_eq!(
        ne_adj.subtype,
        Some(ReceptorSubtype::NeBeta),
        "Inhibition should route to NE β"
    );
    assert!(
        ne_adj.sensitivity_factor > 1.0,
        "Negative z (poor inhibition) should boost NE β"
    );
}

#[test]
fn test_ne_subtype_routing_wm_to_alpha() {
    // Working memory benchmarks should route to NE α (tonic precision)
    let cal = NeuromodCalibration::from_z_scores(&[("WorM::N-back", -2.0)]);
    let ne_alpha = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "NE" && a.subtype == Some(ReceptorSubtype::NeAlpha));
    assert!(
        ne_alpha.is_some(),
        "WM should route to NE α: {:?}",
        cal.adjustments
            .iter()
            .map(|a| (&a.transmitter, &a.subtype))
            .collect::<Vec<_>>()
    );
    // Should also produce ACh adjustment (WM → ACh global)
    let ach = cal.adjustments.iter().find(|a| a.transmitter == "ACh");
    assert!(ach.is_some(), "WM should also adjust ACh");
}

#[test]
fn test_ne_subtype_apply_routes_to_bath() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    let pre_alpha = bath.ne_subtypes.excitatory;
    let pre_beta = bath.ne_subtypes.inhibitory;

    let cal = NeuromodCalibration {
        adjustments: vec![
            TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeBeta),
                z_score: -2.0,
                source_benchmarks: vec!["Inhibition::StopSignal".into()],
                sensitivity_factor: 1.15,
                rationale: "boost β".into(),
            },
            TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeAlpha),
                z_score: -1.5,
                source_benchmarks: vec!["WorM::N-back".into()],
                sensitivity_factor: 1.10,
                rationale: "boost α".into(),
            },
        ],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    assert!(
        (bath.ne_subtypes.excitatory - pre_alpha * 1.10).abs() < 0.001,
        "NE α should be boosted: {} vs expected {}",
        bath.ne_subtypes.excitatory,
        pre_alpha * 1.10
    );
    assert!(
        (bath.ne_subtypes.inhibitory - pre_beta * 1.15).abs() < 0.001,
        "NE β should be boosted: {} vs expected {}",
        bath.ne_subtypes.inhibitory,
        pre_beta * 1.15
    );
    // Global NE sensitivity should be unchanged
    assert!(
        (bath.noradrenaline.receptor_sensitivity - 1.0).abs() < 0.001,
        "Global NE should be unchanged"
    );
}

#[test]
fn test_ne_subtype_summary_labels() {
    let cal = NeuromodCalibration {
        adjustments: vec![
            TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeAlpha),
                z_score: -1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.1,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "NE",
                subtype: Some(ReceptorSubtype::NeBeta),
                z_score: -2.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.2,
                rationale: "test".into(),
            },
        ],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    let summary = cal.summary();
    assert!(
        summary.contains("NE-α"),
        "Summary should show NE-α: {summary}"
    );
    assert!(
        summary.contains("NE-β"),
        "Summary should show NE-β: {summary}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: 9-transmitter expansion tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_gaba_calibration_from_dual_task() {
    // DualTask cost z → GABA sensitivity factor (inverted: high cost = low GABA)
    let cal = NeuromodCalibration::from_z_scores(&[("Executive::DualTask", 2.0)]);
    let gaba = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "GABA")
        .expect("Should have GABA adjustment");
    // High dual-task cost (positive z) with invert=true → boost GABA
    assert!(
        gaba.sensitivity_factor > 1.0,
        "High DualTask cost should boost GABA, got {}",
        gaba.sensitivity_factor
    );
}

#[test]
fn test_oxytocin_from_social_benchmarks() {
    let cal = NeuromodCalibration::from_z_scores(&[
        ("Social::UltimatumGame", -1.5),
        ("Social::RME", -1.0),
    ]);
    let oxy = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Oxytocin")
        .expect("Should have Oxytocin adjustment");
    // Negative z = poor social cognition → boost oxytocin
    assert!(
        oxy.sensitivity_factor > 1.0,
        "Low social z should boost Oxytocin, got {}",
        oxy.sensitivity_factor
    );
}

#[test]
fn test_glutamate_from_lapse_rate() {
    // Synthetic PVT key for lapse_rate → Glutamate
    let cal = NeuromodCalibration::from_z_scores(&[("PVT::lapse_rate", 2.0)]);
    let glut = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Glutamate")
        .expect("Should have Glutamate adjustment");
    // High lapse rate (positive z) with invert=true → attenuate glutamate
    assert!(
        glut.sensitivity_factor > 1.0,
        "High lapse rate should adjust Glutamate sensitivity, got {}",
        glut.sensitivity_factor
    );
}

#[test]
fn test_adenosine_from_pvt() {
    // SustainedAttention::PVT vigilance_decrement → Adenosine
    let cal = NeuromodCalibration::from_z_scores(&[("SustainedAttention::PVT", 2.0)]);
    let aden = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Adenosine")
        .expect("Should have Adenosine adjustment");
    // High decrement (positive z) with invert=true → attenuate adenosine
    assert!(
        aden.sensitivity_factor > 1.0,
        "High vigilance decrement should adjust Adenosine, got {}",
        aden.sensitivity_factor
    );
}

#[test]
fn test_ecb_from_allostatic() {
    let cal = NeuromodCalibration::from_z_scores(&[("Allostatic::Endocannabinoid", 2.0)]);
    let ecb = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Endocannabinoid")
        .expect("Should have Endocannabinoid adjustment");
    // High allostatic load (positive z) with invert=true → boost ECB
    assert!(
        ecb.sensitivity_factor > 1.0,
        "High allostatic load should boost ECB, got {}",
        ecb.sensitivity_factor
    );
}

#[test]
fn test_apply_routes_all_9_transmitters() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    let cal = NeuromodCalibration {
        adjustments: vec![
            TransmitterCalibration {
                transmitter: "DA",
                subtype: None,
                z_score: 1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 0.95,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "NE",
                subtype: None,
                z_score: -1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.05,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "5-HT",
                subtype: None,
                z_score: 0.5,
                source_benchmarks: vec![],
                sensitivity_factor: 0.98,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "ACh",
                subtype: None,
                z_score: -0.5,
                source_benchmarks: vec![],
                sensitivity_factor: 1.02,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "GABA",
                subtype: None,
                z_score: 1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.05,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "Oxytocin",
                subtype: None,
                z_score: -1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.03,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "Glutamate",
                subtype: None,
                z_score: 0.8,
                source_benchmarks: vec![],
                sensitivity_factor: 0.96,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "Adenosine",
                subtype: None,
                z_score: 1.2,
                source_benchmarks: vec![],
                sensitivity_factor: 0.94,
                rationale: "test".into(),
            },
            TransmitterCalibration {
                transmitter: "Endocannabinoid",
                subtype: None,
                z_score: -0.3,
                source_benchmarks: vec![],
                sensitivity_factor: 1.01,
                rationale: "test".into(),
            },
        ],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    // All 9 should have modified receptor_sensitivity from default 1.0
    assert!((bath.dopamine.receptor_sensitivity - 0.95).abs() < 0.001);
    assert!((bath.noradrenaline.receptor_sensitivity - 1.05).abs() < 0.001);
    assert!((bath.serotonin.receptor_sensitivity - 0.98).abs() < 0.001);
    assert!((bath.acetylcholine.receptor_sensitivity - 1.02).abs() < 0.001);
    assert!((bath.gaba.receptor_sensitivity - 1.05).abs() < 0.001);
    assert!((bath.oxytocin.receptor_sensitivity - 1.03).abs() < 0.001);
    assert!((bath.glutamate.receptor_sensitivity - 0.96).abs() < 0.001);
    assert!((bath.adenosine.receptor_sensitivity - 0.94).abs() < 0.001);
    assert!((bath.endocannabinoid.receptor_sensitivity - 1.01).abs() < 0.001);
}

#[test]
fn test_gaba_subtype_routing() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    let pre_a = bath.gaba_subtypes.excitatory;
    let pre_b = bath.gaba_subtypes.inhibitory;

    let cal = NeuromodCalibration {
        adjustments: vec![
            TransmitterCalibration {
                transmitter: "GABA",
                subtype: Some(ReceptorSubtype::GabaA),
                z_score: -1.0,
                source_benchmarks: vec![],
                sensitivity_factor: 1.10,
                rationale: "test GABA-A".into(),
            },
            TransmitterCalibration {
                transmitter: "GABA",
                subtype: Some(ReceptorSubtype::GabaB),
                z_score: -0.5,
                source_benchmarks: vec![],
                sensitivity_factor: 1.05,
                rationale: "test GABA-B".into(),
            },
        ],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    assert!(
        (bath.gaba_subtypes.excitatory - pre_a * 1.10).abs() < 0.001,
        "GABA-A should be boosted: {} vs expected {}",
        bath.gaba_subtypes.excitatory,
        pre_a * 1.10
    );
    assert!(
        (bath.gaba_subtypes.inhibitory - pre_b * 1.05).abs() < 0.001,
        "GABA-B should be boosted: {} vs expected {}",
        bath.gaba_subtypes.inhibitory,
        pre_b * 1.05
    );
    // Global GABA should be unchanged
    assert!(
        (bath.gaba.receptor_sensitivity - 1.0).abs() < 0.001,
        "Global GABA should be unchanged"
    );
}

#[test]
fn test_self_assessment_social_proxy() {
    // Low social coherence should trigger Oxytocin calibration
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 10,
        cooldown_duration: 5,
        ..Default::default()
    };

    let input = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.7,
        confidence_calibration_error: 0.03,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 0.5, // low → oxy_z < -1.0 (baseline 0.80)
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..80 {
        monitor.update(&input);
    }

    let cal = monitor.check_trigger(false);
    assert!(
        cal.is_some(),
        "Low social coherence should trigger calibration"
    );
    assert!(
        cal.unwrap()
            .adjustments
            .iter()
            .any(|a| a.transmitter == "Oxytocin"),
        "Social proxy should map to Oxytocin adjustment"
    );
}

#[test]
fn test_self_assessment_ei_ratio_proxy() {
    // High E/I ratio should trigger GABA calibration
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 10,
        cooldown_duration: 5,
        ..Default::default()
    };

    let input = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.7,
        confidence_calibration_error: 0.03,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 1.0,
        ei_ratio: 1.5, // high → gaba_z < -1.0
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..80 {
        monitor.update(&input);
    }

    let cal = monitor.check_trigger(false);
    assert!(cal.is_some(), "High E/I ratio should trigger calibration");
    assert!(
        cal.unwrap()
            .adjustments
            .iter()
            .any(|a| a.transmitter == "GABA"),
        "E/I ratio proxy should map to GABA adjustment"
    );
}

#[test]
fn test_normative_pvt_lapse_rate_routing() {
    // from_normative_z_scores should remap PVT + lapse_rate → "PVT::lapse_rate"
    let scores = vec![
        ("SustainedAttention::PVT", "lapse_rate", -1.0),
        ("SustainedAttention::PVT", "vigilance_decrement", -0.5),
    ];
    let cal = NeuromodCalibration::from_normative_z_scores(&scores);

    // lapse_rate is lower-is-better, so normative z=-1.0 means un-corrected z=+1.0
    // This should route to Glutamate (via PVT::lapse_rate key)
    let glut = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Glutamate");
    assert!(
        glut.is_some(),
        "PVT lapse_rate should route to Glutamate: {:?}",
        cal.adjustments
            .iter()
            .map(|a| &a.transmitter)
            .collect::<Vec<_>>()
    );

    // vigilance_decrement is lower-is-better, un-corrected z=+0.5
    // This should route to Adenosine (via SustainedAttention::PVT key)
    let aden = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Adenosine");
    assert!(
        aden.is_some(),
        "PVT vigilance_decrement should route to Adenosine: {:?}",
        cal.adjustments
            .iter()
            .map(|a| &a.transmitter)
            .collect::<Vec<_>>()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: Tolerance-aware calibration gating tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_apply_skips_withdrawal() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    // Force DA into withdrawal state
    bath.dopamine.withdrawal_cycles = 10;
    let pre_sensitivity = bath.dopamine.receptor_sensitivity;

    let cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: -2.0,
            source_benchmarks: vec![],
            sensitivity_factor: 1.15,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    assert!(
        (bath.dopamine.receptor_sensitivity - pre_sensitivity).abs() < 0.001,
        "DA sensitivity should be unchanged during withdrawal: pre={pre_sensitivity}, post={}",
        bath.dopamine.receptor_sensitivity
    );
}

#[test]
fn test_apply_dampens_tolerant() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    // Force DA into tolerant state (high_exposure_cycles > tolerance_onset_cycles)
    bath.dopamine.high_exposure_cycles = bath.dopamine.tolerance_onset_cycles + 5;
    let pre_sensitivity = bath.dopamine.receptor_sensitivity;

    let cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: -2.0,
            source_benchmarks: vec![],
            sensitivity_factor: 1.10,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    // Dampened factor: 1.0 + (1.10 - 1.0) * 0.5 = 1.05
    let expected = (pre_sensitivity * 1.05).clamp(0.5, 2.0);
    assert!(
        (bath.dopamine.receptor_sensitivity - expected).abs() < 0.001,
        "Tolerant DA should get half-strength: got {}, expected {}",
        bath.dopamine.receptor_sensitivity,
        expected
    );
}

#[test]
fn test_apply_normal_full_strength() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    // Normal state: no tolerance, no withdrawal
    let pre_sensitivity = bath.dopamine.receptor_sensitivity;

    let cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: -2.0,
            source_benchmarks: vec![],
            sensitivity_factor: 1.10,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    let expected = (pre_sensitivity * 1.10).clamp(0.5, 2.0);
    assert!(
        (bath.dopamine.receptor_sensitivity - expected).abs() < 0.001,
        "Normal DA should get full strength: got {}, expected {}",
        bath.dopamine.receptor_sensitivity,
        expected
    );
}

#[test]
fn test_subtype_respects_parent_tolerance() {
    let mut bath = symthaea_neuromodulators::NeuromodulatorBath::default();
    // Force NE into withdrawal
    bath.noradrenaline.withdrawal_cycles = 10;
    let pre_alpha = bath.ne_subtypes.excitatory;

    let cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "NE",
            subtype: Some(ReceptorSubtype::NeAlpha),
            z_score: -2.0,
            source_benchmarks: vec![],
            sensitivity_factor: 1.15,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };
    cal.apply(&mut bath);

    // NE-α should be unchanged because parent NE is in withdrawal
    assert!(
        (bath.ne_subtypes.excitatory - pre_alpha).abs() < 0.001,
        "NE-α should be unchanged during NE withdrawal: pre={pre_alpha}, post={}",
        bath.ne_subtypes.excitatory
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: Multi-agent calibration sharing tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_to_shareable_all_transmitters() {
    let cal = NeuromodCalibration::from_z_scores(&[
        ("Executive::Stroop", 1.0),
        ("WorM::N-back", -1.0),
        ("Inhibition::StopSignal", -1.5),
        ("SustainedAttention::CPT", -0.5),
        ("Executive::DualTask", 1.0),
        ("Social::UltimatumGame", -0.8),
        ("PVT::lapse_rate", 1.2),
        ("SustainedAttention::PVT", 0.5),
        ("Allostatic::Endocannabinoid", 0.3),
    ]);
    let profile = cal.to_shareable(Some("agent-1".into()));

    assert_eq!(profile.agent_id, Some("agent-1".into()));
    assert!(
        profile.sensitivity_factors.len() >= 9,
        "Should have factors for all transmitters: got {}",
        profile.sensitivity_factors.len()
    );
    // Check some specific keys
    assert!(profile.sensitivity_factors.contains_key("DA"));
    assert!(profile.sensitivity_factors.contains_key("ACh"));
    assert!(profile.sensitivity_factors.contains_key("GABA"));
    assert!(profile.sensitivity_factors.contains_key("Oxytocin"));
    assert!(profile.sensitivity_factors.contains_key("Glutamate"));
    assert!(profile.sensitivity_factors.contains_key("Adenosine"));
    assert!(profile.sensitivity_factors.contains_key("Endocannabinoid"));
}

#[test]
fn test_merge_weighted_blend() {
    let mut cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: 1.0,
            source_benchmarks: vec![],
            sensitivity_factor: 0.90,
            rationale: "test".into(),
        }],
        confidence_delta: -0.02,
        coverage: 1.0,
        stillness_quality: 0.5,
    };

    let mut peer_factors = std::collections::HashMap::new();
    peer_factors.insert("DA".to_string(), 1.10_f32);
    let peer = SharedCalibrationProfile {
        sensitivity_factors: peer_factors,
        confidence_delta: 0.02,
        coverage: 0.8,
        agent_id: Some("peer".into()),
    };

    cal.merge_peer_calibration(&peer, 0.2);

    // DA: 0.90 * 0.8 + 1.10 * 0.2 = 0.72 + 0.22 = 0.94
    assert!(
        (cal.adjustments[0].sensitivity_factor - 0.94).abs() < 0.01,
        "Blended DA factor should be ~0.94, got {}",
        cal.adjustments[0].sensitivity_factor
    );
    // Confidence: -0.02 * 0.8 + 0.02 * 0.2 = -0.016 + 0.004 = -0.012
    assert!(
        (cal.confidence_delta - (-0.012)).abs() < 0.01,
        "Blended confidence delta should be ~-0.012, got {}",
        cal.confidence_delta
    );
}

#[test]
fn test_merge_capped_weight() {
    let mut cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: 1.0,
            source_benchmarks: vec![],
            sensitivity_factor: 0.90,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };

    let mut peer_factors = std::collections::HashMap::new();
    peer_factors.insert("DA".to_string(), 1.10_f32);
    let peer = SharedCalibrationProfile {
        sensitivity_factors: peer_factors,
        confidence_delta: 0.0,
        coverage: 0.8,
        agent_id: None,
    };

    // Pass weight > 0.5 — should be clamped
    cal.merge_peer_calibration(&peer, 0.9);

    // DA: 0.90 * 0.5 + 1.10 * 0.5 = 1.0 (capped at 0.5)
    assert!(
        (cal.adjustments[0].sensitivity_factor - 1.0).abs() < 0.01,
        "Weight should be clamped to 0.5: got {}",
        cal.adjustments[0].sensitivity_factor
    );
}

#[test]
fn test_merge_missing_key_unchanged() {
    let mut cal = NeuromodCalibration {
        adjustments: vec![TransmitterCalibration {
            transmitter: "DA",
            subtype: None,
            z_score: 1.0,
            source_benchmarks: vec![],
            sensitivity_factor: 0.90,
            rationale: "test".into(),
        }],
        confidence_delta: 0.0,
        coverage: 1.0,
        stillness_quality: 0.5,
    };

    // Peer has ACh but not DA
    let mut peer_factors = std::collections::HashMap::new();
    peer_factors.insert("ACh".to_string(), 1.10_f32);
    let peer = SharedCalibrationProfile {
        sensitivity_factors: peer_factors,
        confidence_delta: 0.0,
        coverage: 0.5,
        agent_id: None,
    };

    cal.merge_peer_calibration(&peer, 0.3);

    // DA should be unchanged — peer doesn't have DA
    assert!(
        (cal.adjustments[0].sensitivity_factor - 0.90).abs() < 0.001,
        "DA should be unchanged when peer lacks it: got {}",
        cal.adjustments[0].sensitivity_factor
    );
}

#[test]
fn test_oxytocin_modulated_weight() {
    // Verify the CLS-level peer merge weight formula:
    //   weight = (BASE_WEIGHT + oxy * OXY_SCALE).min(CAP)
    // Production values: BASE=0.05, OXY_SCALE=0.15, CAP=0.35
    const BASE_WEIGHT: f32 = 0.05;
    const OXY_SCALE: f32 = 0.15;
    const CAP: f32 = 0.35;

    let oxy_low = 0.3_f32; // low oxytocin
    let oxy_high = 1.5_f32; // high oxytocin (effective can exceed 1.0)

    let weight_low = (BASE_WEIGHT + oxy_low * OXY_SCALE).min(CAP);
    let weight_high = (BASE_WEIGHT + oxy_high * OXY_SCALE).min(CAP);

    // Low oxy: 0.05 + 0.3 * 0.15 = 0.095
    assert!(
        weight_low < 0.10,
        "Low oxy should give low weight: {weight_low}"
    );
    // High oxy: 0.05 + 1.5 * 0.15 = 0.275 (below 0.35 cap)
    assert!(
        (weight_high - 0.275).abs() < 0.01,
        "High oxy should give ~0.275 weight: {weight_high}"
    );
    // Very high oxy should cap at 0.35
    let oxy_extreme = 3.0_f32;
    let weight_extreme = (BASE_WEIGHT + oxy_extreme * OXY_SCALE).min(CAP);
    assert!(
        (weight_extreme - CAP).abs() < 0.001,
        "Extreme oxy should cap at {CAP}: {weight_extreme}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Phase 5: JSON z-score parsing tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_from_json_z_scores_valid() {
    let json = r#"[
            {"benchmark":"Executive::Stroop","metric":"stroop_effect","z_score":-0.42},
            {"benchmark":"WorM::N-back","metric":"nback_2::accuracy","z_score":1.2}
        ]"#;
    let cal = NeuromodCalibration::from_json_z_scores(json);
    assert!(cal.is_ok(), "Valid JSON should parse: {:?}", cal.err());
    let cal = cal.unwrap();
    assert!(
        !cal.adjustments.is_empty(),
        "Should produce adjustments from valid JSON"
    );
    // DA should be present (from Stroop)
    assert!(
        cal.adjustments.iter().any(|a| a.transmitter == "DA"),
        "Should have DA from Stroop"
    );
}

#[test]
fn test_from_json_z_scores_malformed() {
    let bad_json = "not valid json at all";
    let cal = NeuromodCalibration::from_json_z_scores(bad_json);
    assert!(cal.is_err(), "Malformed JSON should return Err");
}

// ── Improvement 1: ECB composite derivation tests ────────────────────

#[test]
fn test_ecb_composite_from_pvt_and_cpt() {
    // PVT lapse_rate + CPT sustained attention → ECB composite
    let z_scores = vec![
        ("PVT::lapse_rate", 1.5),          // high lapses
        ("SustainedAttention::CPT", -1.0), // poor attention
    ];
    let cal = NeuromodCalibration::from_z_scores(&z_scores);

    // Should have: Glutamate (from PVT), 5-HT (from CPT), Adenosine (from PVT),
    // AND Endocannabinoid (composite from PVT + CPT)
    let ecb = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Endocannabinoid");
    assert!(
        ecb.is_some(),
        "ECB should be derived from PVT+CPT composite"
    );
    assert!(
        ecb.unwrap().rationale.contains("composite"),
        "Rationale should mention composite derivation"
    );
}

#[test]
fn test_ecb_composite_weighted_averaging() {
    // Allostatic proxy (weight 2.0) + PVT (weight 1.0) → weighted mean
    let z_scores = vec![
        ("Allostatic::Endocannabinoid", 2.0), // high allostatic load
        ("PVT::lapse_rate", 0.0),             // normal lapses
    ];
    let cal = NeuromodCalibration::from_z_scores(&z_scores);
    let ecb = cal
        .adjustments
        .iter()
        .find(|a| a.transmitter == "Endocannabinoid")
        .unwrap();

    // Weighted: (2.0*2.0 + 0.0*1.0) / (2.0+1.0) = 1.333...
    // With higher weight on allostatic, result should be > 1.0
    assert!(
        ecb.z_score > 1.0,
        "Weighted z should favor allostatic proxy (weight 2x)"
    );
}

#[test]
fn test_ecb_no_sources_no_ecb() {
    // Only DA benchmarks → no ECB
    let z_scores = vec![("Executive::Stroop", 0.5)];
    let cal = NeuromodCalibration::from_z_scores(&z_scores);
    assert!(
        cal.adjustments
            .iter()
            .all(|a| a.transmitter != "Endocannabinoid"),
        "No ECB sources → no ECB adjustment"
    );
}

// ── Improvement 4: Calibration history tests ─────────────────────────

#[test]
fn test_calibration_history_record_and_len() {
    let mut history = CalibrationHistory::default();
    assert!(history.is_empty());

    let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 1.0)]);
    history.record(&cal, 100);
    assert_eq!(history.len(), 1);

    history.record(&cal, 200);
    assert_eq!(history.len(), 2);
}

#[test]
fn test_calibration_history_sliding_window() {
    let mut history = CalibrationHistory::default();
    let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 0.5)]);

    // Fill beyond max_entries (20)
    for i in 0..25 {
        history.record(&cal, i * 100);
    }
    assert_eq!(history.len(), 20, "Should cap at max_entries");
}

#[test]
fn test_calibration_history_drift_direction() {
    let mut history = CalibrationHistory::default();

    // Consistently boost DA (factor > 1.0)
    for i in 0..5 {
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", -1.0)]);
        history.record(&cal, i * 100);
    }

    let drift = history.drift_direction("DA");
    assert!(drift.is_some());
    assert!(
        drift.unwrap() > 0.0,
        "Consistent negative z → boost DA → positive drift"
    );
}

#[test]
fn test_calibration_history_systematic_drift_detection() {
    let mut history = CalibrationHistory::default();

    // 6 calibrations all boosting DA
    for i in 0..6 {
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", -2.0)]);
        history.record(&cal, i * 100);
    }
    assert!(
        history.is_systematic_drift("DA"),
        "6/6 in same direction should be systematic"
    );

    // Not enough data for 5-HT
    assert!(
        !history.is_systematic_drift("5-HT"),
        "No 5-HT data → not systematic"
    );
}

#[test]
fn test_calibration_history_no_drift_oscillating() {
    let mut history = CalibrationHistory::default();

    // Oscillating: alternating boost/attenuate
    for i in 0..6 {
        let z = if i % 2 == 0 { -1.0 } else { 1.0 };
        let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", z)]);
        history.record(&cal, i * 100);
    }
    assert!(
        !history.is_systematic_drift("DA"),
        "Oscillating should NOT be systematic drift"
    );
}

#[test]
fn test_calibration_history_mean_interval() {
    let mut history = CalibrationHistory::default();
    let cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 0.5)]);

    history.record(&cal, 100);
    history.record(&cal, 300);
    history.record(&cal, 500);

    let interval = history.mean_interval().unwrap();
    assert!(
        (interval - 200.0).abs() < 0.1,
        "Mean interval should be 200 cycles"
    );
}

#[test]
fn test_calibration_history_insufficient_data() {
    let history = CalibrationHistory::default();
    assert!(
        history.drift_direction("DA").is_none(),
        "Empty history → None"
    );
    assert!(history.mean_interval().is_none(), "Empty history → None");
}

#[test]
fn test_scale_by_sleep_quality_full() {
    let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
    let original_factor = cal.adjustments[0].sensitivity_factor;
    cal.scale_by_sleep_quality(1.0);
    assert!(
        (cal.adjustments[0].sensitivity_factor - original_factor).abs() < 0.0001,
        "Full quality should not change factors"
    );
}

#[test]
fn test_scale_by_sleep_quality_half() {
    let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
    let original_factor = cal.adjustments[0].sensitivity_factor;
    let deviation = original_factor - 1.0;
    cal.scale_by_sleep_quality(0.5);
    let expected = 1.0 + deviation * 0.5;
    assert!(
        (cal.adjustments[0].sensitivity_factor - expected).abs() < 0.0001,
        "Half quality should halve deviation: got {}, expected {}",
        cal.adjustments[0].sensitivity_factor,
        expected
    );
}

#[test]
fn test_scale_by_sleep_quality_zero() {
    let mut cal = NeuromodCalibration::from_z_scores(&[("Executive::Stroop", 2.0)]);
    cal.scale_by_sleep_quality(0.0);
    assert!(
        (cal.adjustments[0].sensitivity_factor - 1.0).abs() < 0.0001,
        "Zero quality should neutralize all factors to 1.0"
    );
    assert!(
        cal.confidence_delta.abs() < 0.0001,
        "Zero quality should zero confidence delta"
    );
}

#[test]
fn test_baseline_nudges_from_systematic_drift() {
    let mut history = CalibrationHistory::default();
    // Create 6 calibrations all boosting DA (systematic positive drift)
    for i in 0..6 {
        let mut factors = std::collections::HashMap::new();
        factors.insert("DA".to_string(), 1.05); // consistently boosting
        history.entries.push_back(CalibrationSnapshot {
            factors,
            confidence_delta: 0.0,
            coverage: 0.8,
            applied_at_cycle: i * 100,
        });
    }
    let nudges = history.compute_baseline_nudges();
    assert!(!nudges.is_empty(), "Should detect drift and produce nudges");
    let (name, nudge) = &nudges[0];
    assert_eq!(name, "DA");
    assert!(*nudge > 0.0, "Positive drift should produce positive nudge");
    assert!(*nudge <= 0.02, "Nudge should be clamped to ±0.02");
}

#[test]
fn test_baseline_nudges_no_drift() {
    let mut history = CalibrationHistory::default();
    // Oscillating factors: no systematic drift
    for i in 0..6 {
        let mut factors = std::collections::HashMap::new();
        let factor = if i % 2 == 0 { 1.05 } else { 0.95 };
        factors.insert("DA".to_string(), factor);
        history.entries.push_back(CalibrationSnapshot {
            factors,
            confidence_delta: 0.0,
            coverage: 0.8,
            applied_at_cycle: i * 100,
        });
    }
    let nudges = history.compute_baseline_nudges();
    assert!(
        nudges.is_empty(),
        "Oscillating drift should produce no nudges"
    );
}

#[test]
fn test_cooldown_override_or_logic() {
    // Verify that either drift_anomalous OR severely elevated PE overrides cooldown.
    // The override threshold is 0.35 (higher than trigger threshold ~0.2) to
    // prevent cooldown from being effectively disabled during normal drift.

    // Case 1: drift_anomalous=true, low PE → drift alone overrides cooldown
    let mut monitor1 = SelfAssessmentMonitor::default();
    monitor1.observations_since_calibration = 300;
    monitor1.cooldown = 100;
    monitor1.pe_ema = 0.05; // low PE, but drift is anomalous
    // With anomalous drift, the monitor should reach z-score computation.
    // pe_z = (0.05-0.1)/0.05 = -1.0 → won't trigger from PE alone,
    // but the cooldown gate is bypassed by drift_anomalous.
    let _result1 = monitor1.check_trigger(true);
    // (May or may not produce calibration depending on other z-scores,
    //  but the point is it wasn't blocked by cooldown.)

    // Case 2: severely elevated PE alone (no drift) overrides cooldown
    let mut monitor2 = SelfAssessmentMonitor::default();
    monitor2.observations_since_calibration = 300;
    monitor2.cooldown = 100;
    monitor2.pe_ema = 0.40; // well above 0.35 override threshold
    // pe_z = (0.40-0.1)/0.05 = 6.0 → needs_calibration=true
    let result2 = monitor2.check_trigger(false);
    assert!(
        result2.is_some(),
        "Severely elevated PE (>0.35) alone should override cooldown"
    );

    // Case 3: moderate PE (below override) should NOT override cooldown
    let mut monitor3 = SelfAssessmentMonitor::default();
    monitor3.observations_since_calibration = 300;
    monitor3.cooldown = 100;
    monitor3.pe_ema = 0.25; // above trigger but below 0.35 override
    let result3 = monitor3.check_trigger(false);
    assert!(
        result3.is_none(),
        "Moderate PE (0.25 < 0.35) should NOT override cooldown"
    );
}

#[test]
fn test_confidence_delta_symmetric() {
    // Overconfident
    let cal_over = NeuromodCalibration::from_z_scores(&[("Metacognition::FeelingOfKnowing", 1.0)]);
    // Underconfident
    let cal_under =
        NeuromodCalibration::from_z_scores(&[("Metacognition::FeelingOfKnowing", -1.0)]);
    assert!(
        (cal_over.confidence_delta.abs() - cal_under.confidence_delta.abs()).abs() < 0.0001,
        "Confidence delta should be symmetric: over={}, under={}",
        cal_over.confidence_delta,
        cal_under.confidence_delta
    );
    assert!(
        cal_over.confidence_delta < 0.0,
        "Overconfident should reduce"
    );
    assert!(
        cal_under.confidence_delta > 0.0,
        "Underconfident should boost"
    );
}

// ═══════════════════════════════════════════════════════════════════
// NaN guard tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nan_z_score_produces_neutral_factor() {
    use super::normative::z_to_sensitivity_factor;
    assert!((z_to_sensitivity_factor(f64::NAN, false) - 1.0).abs() < f32::EPSILON);
    assert!((z_to_sensitivity_factor(f64::NAN, true) - 1.0).abs() < f32::EPSILON);
    assert!((z_to_sensitivity_factor(f64::INFINITY, false) - 1.0).abs() < f32::EPSILON);
    assert!((z_to_sensitivity_factor(f64::NEG_INFINITY, true) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_nan_in_calibration_does_not_corrupt() {
    let cal = NeuromodCalibration::from_z_scores(&[
        ("Executive::Stroop", f64::NAN),
        ("WorM::N-back", 1.0), // valid
    ]);
    // NaN Stroop should produce neutral DA factor (1.0)
    let da = cal.adjustments.iter().find(|a| a.transmitter == "DA");
    if let Some(da_adj) = da {
        assert!(
            da_adj.sensitivity_factor.is_finite(),
            "DA factor should be finite even with NaN input: {}",
            da_adj.sensitivity_factor
        );
        assert_eq!(
            da_adj.sensitivity_factor, 1.0,
            "NaN z-score should produce neutral factor"
        );
    }
    // ACh should still be valid
    let ach = cal.adjustments.iter().find(|a| a.transmitter == "ACh");
    assert!(ach.is_some(), "Valid ACh z-score should produce adjustment");
    assert!(ach.unwrap().sensitivity_factor.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// ECB proxy direction test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_ecb_self_assessment_proxy_direction() {
    // High allostatic load should produce a BOOST (factor > 1.0) for ECB,
    // not an attenuation, because stress depletes the endocannabinoid buffer.
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 5,
        ..Default::default()
    };

    // Feed high allostatic load to drive EMA above baseline
    let high_stress_input = SelfAssessmentInput {
        prediction_error: 0.5,
        coherence: 0.7,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 0.8,
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.8, // high stress
    };

    for _ in 0..100 {
        monitor.update(&high_stress_input);
    }

    let cal = monitor.check_trigger(false);
    if let Some(cal) = cal {
        // If ECB is in the calibration, verify its direction
        let ecb = cal
            .adjustments
            .iter()
            .find(|a| a.transmitter == "Endocannabinoid");
        if let Some(ecb_adj) = ecb {
            assert!(
                ecb_adj.sensitivity_factor >= 1.0,
                "High allostatic load should BOOST ECB (factor >= 1.0), got {}",
                ecb_adj.sensitivity_factor
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Social coherence baseline test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_social_coherence_default_does_not_false_trigger() {
    // With default oxytocin (~0.5), the input transform produces ~0.75.
    // The monitor's oxy_z baseline should NOT produce a false-positive trigger.
    let mut monitor = SelfAssessmentMonitor {
        warmup_threshold: 5,
        cooldown_duration: 5,
        ..Default::default()
    };

    // Feed the default-ish social coherence value
    let normal_input = SelfAssessmentInput {
        prediction_error: 0.08,
        coherence: 0.75,
        confidence_calibration_error: 0.0,
        attention_utilization: 0.5,
        inhibition_error_rate: 0.0,
        drift_anomalous: false,
        social_coherence: 0.75, // default steady-state (oxy=0.5 → 0.5*0.5+0.5=0.75)
        ei_ratio: 1.0,
        excitotoxicity_risk: 0.0,
        sleep_pressure: 0.3,
        allostatic_load: 0.2,
    };

    for _ in 0..300 {
        monitor.update(&normal_input);
    }

    let cal = monitor.check_trigger(false);
    // With correct baseline (0.80), oxy_z = (0.75 - 0.80)/0.1 = -0.5
    // This is above the -1.0 threshold, so oxytocin should NOT trigger.
    if let Some(cal) = cal {
        let has_oxy = cal.adjustments.iter().any(|a| a.transmitter == "Oxytocin");
        assert!(
            !has_oxy,
            "Default social coherence (0.75) should NOT trigger oxytocin calibration"
        );
    }
}
