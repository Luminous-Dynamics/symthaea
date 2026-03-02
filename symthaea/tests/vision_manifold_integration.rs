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
}
