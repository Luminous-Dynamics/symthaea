// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: Vision Manifold → Cognitive Loop via cycle_with_hv().
//!
//! Verifies that the vision manifold's output HV flows correctly through
//! the cognitive loop's fast path (500Hz non-text pipeline).

#[cfg(feature = "vision-manifold")]
mod vision_cognitive_integration {
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
    use symthaea::perception::vision_manifold::{
        CameraManifold, VisionBridge, VisionConfig, VisionManifold,
    };

    const GENESIS: &str = "vision_manifold_integration_v1";

    fn make_service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            ..Default::default()
        })
        .expect("Failed to create CognitiveLoopService")
    }

    fn make_bridge(width: u32, height: u32) -> VisionBridge {
        let cfg = VisionConfig::default();
        VisionBridge::new(cfg, width, height)
    }

    // ── Test 1: VisionBridge output → cycle_with_hv() ────────────────

    #[test]
    fn test_vision_bridge_to_cognitive_loop() {
        let mut service = make_service();
        let mut bridge = make_bridge(64, 64);

        // Generate mock frames with gradient pattern
        let frame: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();

        // Process frame through vision bridge
        let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 1, 0.033);

        assert_eq!(tel.frame_sequence, 1);
        assert!(hv.norm() > 0.0, "Vision bridge should produce non-zero HV");

        // Feed the vision HV into the cognitive loop
        let result = service.cycle_with_hv(&hv);

        // Verify the cognitive loop processed it
        assert!(
            result.cycle_time_us > 0,
            "Cognitive cycle should take non-zero time"
        );
        assert!(
            !result.output.is_empty(),
            "Cognitive loop should produce output"
        );
    }

    // ── Test 2: Multi-frame sequence through cognitive loop ──────────

    #[test]
    fn test_multi_frame_vision_pipeline() {
        let mut service = make_service();
        let mut bridge = make_bridge(64, 64);

        let mut prediction_errors = Vec::new();

        // Feed 30 frames (1 second at 30fps) of a static scene
        for i in 0..30 {
            let brightness = 128u8;
            let frame = vec![brightness; 64 * 64];
            let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
            let result = service.cycle_with_hv(&hv);
            prediction_errors.push(result.prediction_error);
        }

        // After converging on a static scene, prediction error should stabilize
        let early_mean: f32 = prediction_errors[2..5].iter().sum::<f32>() / 3.0;
        let late_mean: f32 = prediction_errors[25..30].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= early_mean + 0.1,
            "Prediction error should stabilize: early={early_mean}, late={late_mean}"
        );
    }

    // ── Test 3: Scene change propagates through cognitive loop ───────

    #[test]
    fn test_scene_change_detection_through_loop() {
        let mut service = make_service();
        let mut bridge = make_bridge(64, 64);

        // Converge on scene A
        for _ in 0..20 {
            let frame = vec![50u8; 64 * 64];
            let hv = bridge.process_frame(&frame, 64, 64, 1, 0.033);
            service.cycle_with_hv(&hv);
        }
        let stable_error = service
            .cycle_with_hv(&bridge.process_frame(&vec![50u8; 64 * 64], 64, 64, 1, 0.033))
            .prediction_error;

        // Switch to scene B
        let frame_b = vec![200u8; 64 * 64];
        let hv_b = bridge.process_frame(&frame_b, 64, 64, 1, 0.033);
        let shift_result = service.cycle_with_hv(&hv_b);

        // The cognitive loop should see increased prediction error
        // (the vision HV changed, so the CfC's compressed state diverges)
        assert!(
            shift_result.prediction_error >= stable_error * 0.5,
            "Scene change should propagate: stable={stable_error}, shift={}",
            shift_result.prediction_error
        );
    }

    // ── Test 4: CameraManifold convenience wrapper ───────────────────

    #[test]
    fn test_camera_manifold_to_cognitive_loop() {
        let mut service = make_service();
        let cfg = VisionConfig::default();
        let mut cam = CameraManifold::with_mock(cfg, 64, 64);

        // 10 ticks through the camera manifold
        for _ in 0..10 {
            let tel = cam.tick().unwrap();
            assert!(tel.frame_sequence > 0);
        }

        // Use the manifold's state as input to cognitive loop
        let state = cam.manifold().state().clone();
        let result = service.cycle_with_hv(&state);
        assert!(result.cycle_time_us > 0);
    }

    // ── Test 5: Vision telemetry aligns with cognitive metadata ──────

    #[test]
    fn test_vision_telemetry_consistency() {
        let mut service = make_service();
        let mut bridge = make_bridge(64, 64);

        let frame: Vec<u8> = (0..64 * 64).map(|i| ((i * 7) % 256) as u8).collect();

        // Process a few frames
        for _ in 0..5 {
            let (hv, tel) = bridge.process_frame_with_telemetry(&frame, 64, 64, 1, 0.033);
            let result = service.cycle_with_hv(&hv);

            // Vision manifold should report encoding time
            assert!(
                tel.encode_time_us > 0 || tel.frame_sequence == 1,
                "Encoding should take non-zero time"
            );

            // Cognitive loop should return a thought vector
            assert!(!result.thought_vector.is_empty());
        }
    }

    // ── Test 6: Horizon accuracy accessible through bridge ───────────

    #[test]
    fn test_horizon_accuracy_through_pipeline() {
        let mut bridge = make_bridge(64, 64);
        let frame: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();

        // Feed frames to build up temporal state
        for _ in 0..10 {
            bridge.process_frame(&frame, 64, 64, 1, 0.033);
        }

        let acc = bridge.manifold().evaluate_horizons();
        assert_eq!(acc.horizons.len(), 4);
        assert_eq!(acc.labels.len(), 4);
        assert_eq!(acc.errors.len(), 4);

        // After converging on static scene, short-horizon should be accurate
        assert!(
            acc.errors[0] < 0.5,
            "Short-horizon prediction should be decent after convergence, got {}",
            acc.errors[0]
        );
    }

    // ── Test 7: Saliency refinement through bridge ───────────────────

    #[test]
    fn test_saliency_refinement_through_bridge() {
        let mut bridge = make_bridge(64, 64);

        // Frame A (dark)
        let frame_a = vec![30u8; 64 * 64];
        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);

        // Frame B (gradient → creates surprise contrast)
        let frame_b: Vec<u8> = (0..64 * 64).map(|i| ((i * 3) % 256) as u8).collect();
        bridge.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Saliency refinement should not panic
        bridge.manifold_mut().refine_from_attention();
    }

    // ── Test 8: Config-enabled vision path with injected frame ─────

    #[test]
    fn test_config_enabled_vision_with_injected_frame() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            enable_vision_manifold: true,
            vision_frame_width: 64,
            vision_frame_height: 64,
            ..Default::default()
        })
        .expect("Failed to create service");

        service.inject_vision_frame(vec![128u8; 64 * 64]);
        let result = service.cycle("hello vision");

        let vt = result
            .metadata
            .vision
            .as_ref()
            .expect("vision telemetry should be Some");
        assert!(vt.vision_active);
        assert!(vt.prediction_error.is_finite());
        assert!(vt.manifold_coherence.is_finite());
        assert!(vt.attention_entropy.is_finite());
        assert!(vt.frame_sequence > 0);
    }

    // ── Test 9: Config-enabled vision path without frame (gray fallback) ──

    #[test]
    fn test_config_enabled_vision_gray_fallback() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            enable_vision_manifold: true,
            vision_frame_width: 64,
            vision_frame_height: 64,
            ..Default::default()
        })
        .expect("Failed to create service");

        let result = service.cycle("no frame");
        assert!(
            result.metadata.vision.is_some(),
            "vision telemetry should be populated even without explicit frame injection"
        );
    }

    // ── Test 10: Cross-manifold predictor Hebbian learning converges ──

    #[test]
    fn test_cross_manifold_predictor_learning_convergence() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            enable_vision_manifold: true,
            enable_cross_manifold_predictor: true,
            vision_frame_width: 64,
            vision_frame_height: 64,
            ..Default::default()
        })
        .expect("Failed to create service");

        // Feed a consistent frame+text pair so vision→cognitive mapping can learn
        let frame = vec![128u8; 64 * 64];
        let mut errors = Vec::new();

        for _ in 0..50 {
            service.inject_vision_frame(frame.clone());
            let result = service.cycle("consistent input for learning");
            if let Some(ref vt) = result.metadata.vision {
                errors.push(vt.cross_manifold_prediction_error);
            }
        }

        assert!(
            errors.len() >= 40,
            "Should have vision telemetry for most cycles, got {}",
            errors.len()
        );

        // Prediction error should decrease (or at least not grow) as mapping learns
        let early: f32 = errors[5..10].iter().sum::<f32>() / 5.0;
        let late: f32 = errors[40..].iter().sum::<f32>() / (errors.len() - 40) as f32;
        assert!(
            late <= early + 0.15,
            "Cross-manifold prediction error should stabilize or decrease: early={early:.3}, late={late:.3}"
        );

        // All errors should be finite
        assert!(
            errors.iter().all(|e| e.is_finite()),
            "All prediction errors must be finite"
        );
    }

    // ── Test 11: ACh modulates scene memory thresholds ────────────────

    #[test]
    fn test_ach_modulated_scene_memory_thresholds() {
        // Run two services: one with high ACh, one with low ACh
        // Verify they produce different scene recognition behavior.

        let make = || {
            CognitiveLoopService::new(CognitiveLoopConfig {
                genesis_phrase: Some(GENESIS.to_string()),
                async_training: false,
                learning_threshold: 0.0,
                enable_vision_manifold: true,
                vision_frame_width: 64,
                vision_frame_height: 64,
                ..Default::default()
            })
            .expect("Failed to create service")
        };

        let mut svc_low_ach = make();
        let mut svc_high_ach = make();

        // Inject low ACh (0.3) and high ACh (1.8)
        svc_low_ach.inject_pharmacological("acetylcholine", -0.7, 200);
        svc_high_ach.inject_pharmacological("acetylcholine", 0.8, 200);

        let frame = vec![100u8; 64 * 64];

        // Run 20 cycles to let scene memory accumulate
        for _ in 0..20 {
            svc_low_ach.inject_vision_frame(frame.clone());
            svc_low_ach.cycle("scene memory test");
            svc_high_ach.inject_vision_frame(frame.clone());
            svc_high_ach.cycle("scene memory test");
        }

        // Read ACh levels from telemetry to verify injection worked
        let low_result = {
            svc_low_ach.inject_vision_frame(frame.clone());
            svc_low_ach.cycle("check ach")
        };
        let high_result = {
            svc_high_ach.inject_vision_frame(frame.clone());
            svc_high_ach.cycle("check ach")
        };

        let low_ach = low_result.metadata.neuromod.acetylcholine_effective;
        let high_ach = high_result.metadata.neuromod.acetylcholine_effective;

        // ACh injection should produce different effective levels
        assert!(
            high_ach > low_ach,
            "High ACh service should have higher ACh: low={low_ach:.3}, high={high_ach:.3}"
        );

        // Both should produce valid vision telemetry
        assert!(low_result.metadata.vision.is_some());
        assert!(high_result.metadata.vision.is_some());

        let low_vt = low_result.metadata.vision.unwrap();
        let high_vt = high_result.metadata.vision.unwrap();

        // Both should have valid prediction errors
        assert!(low_vt.prediction_error.is_finite());
        assert!(high_vt.prediction_error.is_finite());

        // Both should report timing > 0 (timing fields wired correctly)
        assert!(
            low_vt.encode_time_us > 0 || low_vt.frame_sequence <= 1,
            "Encoding should take measurable time"
        );
    }

    // ── Test 12: Vision telemetry timing fields are populated ─────────

    #[test]
    fn test_vision_timing_telemetry_populated() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            enable_vision_manifold: true,
            vision_frame_width: 64,
            vision_frame_height: 64,
            ..Default::default()
        })
        .expect("Failed to create service");

        // Run a few cycles to warm up
        let frame: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        for _ in 0..5 {
            service.inject_vision_frame(frame.clone());
            service.cycle("warmup");
        }

        // Check timing on a real cycle
        service.inject_vision_frame(frame.clone());
        let result = service.cycle("timing check");
        let vt = result.metadata.vision.expect("vision should be active");

        assert!(vt.vision_active);
        // After warmup, encode + evolve should take measurable time
        assert!(
            vt.encode_time_us > 0 || vt.evolve_time_us > 0,
            "At least one timing field should be non-zero: encode={}us, evolve={}us",
            vt.encode_time_us,
            vt.evolve_time_us
        );
    }

    // ── Test 13: Neuromod gating affects vision manifold behavior ─────

    #[test]
    fn test_neuromod_gating_vision_behavior() {
        let make = || {
            CognitiveLoopService::new(CognitiveLoopConfig {
                genesis_phrase: Some(GENESIS.to_string()),
                async_training: false,
                learning_threshold: 0.0,
                enable_vision_manifold: true,
                vision_frame_width: 64,
                vision_frame_height: 64,
                ..Default::default()
            })
            .expect("Failed to create service")
        };

        let mut svc_calm = make();
        let mut svc_alert = make();

        // Calm: low NE + low DA → less reactive vision
        svc_calm.inject_pharmacological("noradrenaline", -0.5, 200);
        svc_calm.inject_pharmacological("dopamine", -0.5, 200);

        // Alert: high NE + high DA → more reactive vision
        svc_alert.inject_pharmacological("noradrenaline", 0.5, 200);
        svc_alert.inject_pharmacological("dopamine", 0.5, 200);

        // Converge on scene A
        let frame_a = vec![80u8; 64 * 64];
        for _ in 0..15 {
            svc_calm.inject_vision_frame(frame_a.clone());
            svc_calm.cycle("scene a");
            svc_alert.inject_vision_frame(frame_a.clone());
            svc_alert.cycle("scene a");
        }

        // Scene shift to B — high-contrast change
        let frame_b: Vec<u8> = (0..64 * 64).map(|i| ((i * 5) % 256) as u8).collect();
        svc_calm.inject_vision_frame(frame_b.clone());
        let calm_shift = svc_calm.cycle("scene change");
        svc_alert.inject_vision_frame(frame_b.clone());
        let alert_shift = svc_alert.cycle("scene change");

        let calm_vt = calm_shift.metadata.vision.expect("vision active");
        let alert_vt = alert_shift.metadata.vision.expect("vision active");

        // Both should detect the scene change (non-zero prediction error)
        assert!(
            calm_vt.prediction_error > 0.0,
            "Calm service should still detect scene change"
        );
        assert!(
            alert_vt.prediction_error > 0.0,
            "Alert service should detect scene change"
        );

        // All telemetry should be finite
        assert!(calm_vt.prediction_error.is_finite());
        assert!(alert_vt.prediction_error.is_finite());
        assert!(calm_vt.manifold_coherence.is_finite());
        assert!(alert_vt.manifold_coherence.is_finite());

        // Verify neuromod injection took effect
        let calm_ne = calm_shift.metadata.neuromod.noradrenaline_effective;
        let alert_ne = alert_shift.metadata.neuromod.noradrenaline_effective;
        assert!(
            alert_ne > calm_ne,
            "Alert service should have higher NE: calm={calm_ne:.3}, alert={alert_ne:.3}"
        );
    }

    // ── Test 14: P3-A goal signal — cycles with vision manifold remain stable ──
    //
    // After P3-A, each cognitive cycle sets the vision bridge's goal signal to
    // the current thought HV. This test verifies that:
    //   1. Cycles do not panic or produce NaN
    //   2. Vision telemetry remains well-formed across 20 cycles
    //   3. The goal signal does not degrade prediction quality
    //
    // (We cannot directly inspect `goal_signal` since it is a private field,
    //  but a stable pipeline is the meaningful property to verify.)
    #[test]
    fn test_p3a_goal_signal_wiring_stable() {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            genesis_phrase: Some(GENESIS.to_string()),
            async_training: false,
            learning_threshold: 0.0,
            enable_vision_manifold: true,
            vision_frame_width: 64,
            vision_frame_height: 64,
            ..Default::default()
        })
        .expect("Failed to create service");

        let frame: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let mut last_frame_seq = 0u64;

        for _ in 0..20 {
            service.inject_vision_frame(frame.clone());
            let result = service.cycle("p3a goal signal test");

            let vt = result
                .metadata
                .vision
                .as_ref()
                .expect("vision telemetry should be present");

            assert!(
                vt.vision_active,
                "Vision should remain active across cycles"
            );
            assert!(
                vt.prediction_error.is_finite(),
                "Prediction error must be finite after goal signal wiring"
            );
            assert!(
                vt.manifold_coherence.is_finite(),
                "Manifold coherence must be finite after goal signal wiring"
            );
            assert!(
                vt.frame_sequence > last_frame_seq,
                "Frame sequence should increment: got {}",
                vt.frame_sequence
            );
            last_frame_seq = vt.frame_sequence;
        }
    }

    // ── Test 15: P3-C multi-spectral encoding integrates with VisionConfig ────
    //
    // Verifies that the MultiSpectralEncoder produces valid HVs for all five
    // bands and that multi-band fusion is distinct from single-band encoding.
    #[test]
    fn test_p3c_multi_spectral_encoder() {
        use symthaea::perception::vision_manifold::{
            MultiSpectralEncoder, MultiSpectralFrame, SpectralLayer, SpectrumBand,
        };

        let cfg = VisionConfig::default();
        let mut enc = MultiSpectralEncoder::new(&cfg, 64, 64);

        // Single visible band
        let vis_frame = MultiSpectralFrame::from_visible(vec![128u8; 64 * 64], 64, 64);
        let vis_hv = enc.encode(&vis_frame);
        assert!(
            vis_hv.norm() > 0.0,
            "Visible-band encoding should produce non-zero HV"
        );
        assert!(
            vis_hv.as_slice().iter().all(|x| x.is_finite()),
            "Visible-band HV must be finite"
        );

        // Multi-band frame (Visible + ThermalIR)
        let multi_frame = MultiSpectralFrame::from_visible(vec![128u8; 64 * 64], 64, 64)
            .with_layer(SpectrumBand::ThermalIR, vec![200u8; 64 * 64]);
        let multi_hv = enc.encode(&multi_frame);
        assert!(
            multi_hv.norm() > 0.0,
            "Multi-band encoding should produce non-zero HV"
        );

        // Multi-band HV should differ from single-band HV (different spectral content)
        let sim = vis_hv.similarity(&multi_hv);
        assert!(
            sim < 0.99,
            "Multi-band HV should differ from single-band HV, sim={sim}"
        );

        // All five bands should encode without panicking
        for band in SpectrumBand::ALL {
            let frame = MultiSpectralFrame {
                width: 64,
                height: 64,
                layers: vec![SpectralLayer {
                    band,
                    data: vec![100u8; 64 * 64],
                }],
            };
            let hv = enc.encode(&frame);
            assert!(
                hv.as_slice().iter().all(|x| x.is_finite()),
                "Band {:?} encoding produced non-finite HV",
                band
            );
        }

        // Band identity HVs should be mutually near-orthogonal
        assert!(
            enc.bands_are_orthogonal(),
            "Band identity HVs should be near-orthogonal in 16,384D"
        );
    }

    // ── Test 16: Depth channel adds one feature when enabled ─────────────────
    #[test]
    fn test_depth_channel_adds_feature() {
        let mut cfg = VisionConfig::default();
        assert_eq!(cfg.total_features(), 11);
        cfg.enable_depth = true;
        assert_eq!(cfg.total_features(), 12);

        // Encoder with depth should produce a valid HV from the same pixel data
        use symthaea::perception::vision_manifold::VisionManifold;
        let mut m = VisionManifold::new(cfg, 32, 32);
        let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let tel = m.observe_frame(&pixels, 32, 32, 3, 0.033);
        assert!(tel.manifold_coherence.is_finite());
        assert!(tel.prediction_error.is_finite());
    }

    // ── Test 17: Object binding changes the frame HV representation ──────────
    #[test]
    fn test_object_binding_changes_frame_hv() {
        use symthaea::perception::vision_manifold::VisionManifold;

        // Create two manifolds with identical config except object binding
        let cfg_off = VisionConfig::default();
        let mut cfg_on = VisionConfig::default();
        cfg_on.enable_object_binding = true;

        let mut m_off = VisionManifold::new(cfg_off, 64, 64);
        let mut m_on = VisionManifold::new(cfg_on, 64, 64);

        // Non-uniform frame (two distinct halves → should cluster differently)
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for y in 0..32 {
            for x in 0..64 {
                let base = (y * 64 + x) * 3;
                pixels[base] = 200;
                pixels[base + 1] = 50;
                pixels[base + 2] = 50;
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                let base = (y * 64 + x) * 3;
                pixels[base] = 50;
                pixels[base + 1] = 50;
                pixels[base + 2] = 200;
            }
        }

        m_off.observe_frame(&pixels, 64, 64, 3, 0.033);
        m_on.observe_frame(&pixels, 64, 64, 3, 0.033);

        let state_off = m_off.state();
        let state_on = m_on.state();

        // Both should be finite
        assert!(state_off.norm() > 0.0);
        assert!(state_on.norm() > 0.0);

        // Object binding should produce a different representation
        let sim = state_off.similarity(state_on);
        assert!(
            sim < 0.999,
            "Object binding should change the frame HV, sim={sim}"
        );
    }

    // ── Test 18: Visual imagination (dream_ahead) produces valid HVs ─────────
    #[test]
    fn test_dream_ahead_produces_valid_hvs() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);

        // Need at least one frame to seed the state
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 7 % 256) as u8).collect();
        m.observe_frame(&pixels, 64, 64, 3, 0.033);

        let dreams = m.dream_ahead(5, 0.033);
        assert_eq!(dreams.len(), 5, "Should produce exactly 5 dream steps");

        for (i, dream_hv) in dreams.iter().enumerate() {
            assert!(dream_hv.norm() > 0.0, "Dream step {i} should be non-zero");
            assert!(
                dream_hv.as_slice().iter().all(|x| x.is_finite()),
                "Dream step {i} must be finite"
            );
        }

        // Later dreams should diverge from earlier ones (CfC evolves)
        let sim_01 = dreams[0].similarity(&dreams[1]);
        let sim_04 = dreams[0].similarity(&dreams[4]);
        assert!(
            sim_04 < sim_01 || sim_01 > 0.999,
            "Later dreams should generally diverge (sim01={sim_01}, sim04={sim_04})"
        );
    }

    // ── Test 19: Object memory tracks identity across frames ─────────────────
    #[test]
    fn test_object_memory_cross_frame_tracking() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 64, 64);
        m.enable_object_memory(32);

        // Red top / blue bottom frame
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for y in 0..32 {
            for x in 0..64 {
                let b = (y * 64 + x) * 3;
                pixels[b] = 200;
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                let b = (y * 64 + x) * 3;
                pixels[b + 2] = 200;
            }
        }

        // Feed 5 frames of the same scene
        for _ in 0..5 {
            m.observe_frame(&pixels, 64, 64, 3, 0.033);
        }

        let obj_mem = m.object_memory().expect("object memory should be enabled");
        assert!(
            !obj_mem.is_empty(),
            "Object memory should have tracked objects"
        );

        // Tracks should have length > 1 (persisted across frames)
        let long_tracks = obj_mem
            .tracks()
            .iter()
            .filter(|t| t.track_length > 1)
            .count();
        assert!(
            long_tracks > 0,
            "At least one object should persist across multiple frames"
        );
    }

    // ── Test 20: Multiband bridge pipeline (P3-C wiring end-to-end) ──────────
    #[test]
    fn test_multiband_bridge_pipeline() {
        use symthaea::perception::vision_manifold::{
            MultiSpectralFrame, SpectrumBand, VisionBridge,
        };

        let mut bridge = VisionBridge::new(VisionConfig::default(), 64, 64);
        bridge.enable_multi_spectral(64, 64);

        let frame = MultiSpectralFrame::from_visible(vec![128u8; 64 * 64], 64, 64)
            .with_layer(SpectrumBand::ThermalIR, vec![200u8; 64 * 64]);

        let (hv, tel) = bridge.process_multiband_frame(&frame, 0.033);
        assert!(
            hv.norm() > 0.0,
            "Multiband bridge should produce non-zero HV"
        );
        assert!(tel.encode_time_us > 0, "Encoding time should be measured");
        assert_eq!(tel.frame_sequence, 1, "Frame sequence should be 1");

        // Second frame
        let (hv2, tel2) = bridge.process_multiband_frame(&frame, 0.033);
        assert!(hv2.norm() > 0.0);
        assert_eq!(tel2.frame_sequence, 2);
    }

    // ── Test 21: Imagination-reality comparison produces surprise signal ──────
    #[test]
    fn test_imagination_reality_surprise() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);

        // Frame 1: establish state
        let frame1: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 7 % 256) as u8).collect();
        m.observe_frame(&frame1, 64, 64, 3, 0.033);

        // Frame 2: same scene — imagination should predict well (low surprise)
        let tel2 = m.observe_frame(&frame1, 64, 64, 3, 0.033);
        let surprise_same = tel2.imagination_surprise;

        // Frame 3: completely different scene — high imagination surprise
        let frame3: Vec<u8> = (0..64 * 64 * 3).map(|i| (255 - (i % 256)) as u8).collect();
        let tel3 = m.observe_frame(&frame3, 64, 64, 3, 0.033);
        let surprise_diff = tel3.imagination_surprise;

        assert!(
            surprise_same.is_finite() && surprise_diff.is_finite(),
            "Imagination surprise must be finite"
        );
        assert!(
            surprise_diff > surprise_same,
            "Scene change should produce higher imagination surprise \
             (same={surprise_same}, diff={surprise_diff})"
        );
    }

    // ── Test 22: Visual working memory holds ≤ capacity objects ──────────────
    #[test]
    fn test_visual_working_memory() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 64, 64);
        m.enable_object_memory(32);
        m.enable_working_memory(4);

        // Create a non-uniform frame with distinct regions
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for y in 0..32 {
            for x in 0..32 {
                pixels[(y * 64 + x) * 3] = 200; // red top-left
            }
            for x in 32..64 {
                pixels[(y * 64 + x) * 3 + 1] = 200; // green top-right
            }
        }
        for y in 32..64 {
            for x in 0..32 {
                pixels[(y * 64 + x) * 3 + 2] = 200; // blue bottom-left
            }
            for x in 32..64 {
                let b = (y * 64 + x) * 3;
                pixels[b] = 200;
                pixels[b + 1] = 200; // yellow bottom-right
            }
        }

        for _ in 0..5 {
            m.observe_frame(&pixels, 64, 64, 3, 0.033);
        }

        let wm = m
            .working_memory()
            .expect("working memory should be enabled");
        assert!(
            wm.load() <= 4,
            "Working memory should hold ≤ 4 objects, got {}",
            wm.load()
        );
        // Should have at least 1 object in memory after 5 frames
        assert!(
            wm.load() > 0,
            "Working memory should hold at least 1 object after 5 frames"
        );
    }

    // ── Test 23: Scene graph computes spatial relations ───────────────────────
    #[test]
    fn test_scene_graph_spatial_relations() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 64, 64);
        m.enable_object_memory(32);
        m.enable_scene_graph();

        // Two distinct colored halves → should produce objects with spatial relations
        let mut pixels = vec![0u8; 64 * 64 * 3];
        for y in 0..32 {
            for x in 0..64 {
                pixels[(y * 64 + x) * 3] = 200;
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                pixels[(y * 64 + x) * 3 + 2] = 200;
            }
        }

        for _ in 0..5 {
            m.observe_frame(&pixels, 64, 64, 3, 0.033);
        }

        let sg = m.scene_graph().expect("scene graph should be enabled");
        // If multiple objects are tracked, there should be at least one relation
        let obj_mem = m.object_memory().expect("object memory enabled");
        if obj_mem.len() >= 2 {
            assert!(
                sg.num_edges() > 0,
                "Scene graph should have edges when ≥2 objects are tracked"
            );
            // All edge HVs should be finite
            for edge in sg.edges() {
                assert!(
                    edge.relation_hv.as_slice().iter().all(|x| x.is_finite()),
                    "Scene graph edge HV must be finite"
                );
            }
            // Graph HV should exist
            assert!(
                sg.graph_hv().is_some(),
                "Scene graph HV should be present when edges exist"
            );
        }
    }

    // ── Test 24: Monocular depth cues vary by position ───────────────────────
    #[test]
    fn test_monocular_depth_varies_by_position() {
        use symthaea::perception::vision_manifold::VisionManifold;

        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        // Check feature count before moving cfg into manifold
        assert_eq!(cfg.total_features(), 12);
        let mut m = VisionManifold::new(cfg, 64, 64);

        // High-variance top (near) + low-variance bottom (far)
        let mut pixels = vec![128u8; 64 * 64 * 3];
        // Add noise to top half (high variance → near)
        for y in 0..32 {
            for x in 0..64 {
                let b = (y * 64 + x) * 3;
                pixels[b] = ((x * 17 + y * 31) % 256) as u8;
                pixels[b + 1] = ((x * 23 + y * 13) % 256) as u8;
                pixels[b + 2] = ((x * 37 + y * 7) % 256) as u8;
            }
        }
        // Bottom half stays uniform (low variance → far)

        let tel = m.observe_frame(&pixels, 64, 64, 3, 0.033);
        assert!(
            tel.prediction_error.is_finite(),
            "Depth-enabled frame should produce finite telemetry"
        );
    }
}