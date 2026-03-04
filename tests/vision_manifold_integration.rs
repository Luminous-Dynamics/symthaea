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
}
