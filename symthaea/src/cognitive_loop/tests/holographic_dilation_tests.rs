// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(all(test, feature = "vision-manifold"))]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    #[test]
    fn test_surprise_triggers_holographic_dilation_and_thermo_cost() {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        let _ = fmt().with_env_filter(EnvFilter::new("warn")).try_init();

        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;
        let mut service = CognitiveLoopService::new(config).unwrap();

        let initial_dim = service
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .unwrap()
            .manifold()
            .hdc_dim();

        // === Phase 1: Stabilize + pass dilation cooldown ===
        let frame_a = vec![64u8; 64 * 64 * 3];
        for _ in 0..12 {
            service.sensorimotor.vision_sensory.vision_frame_buffer = Some(frame_a.clone());
            service.cycle("stabilize");
        }

        // === Phase 2: Trigger massive surprise (different frame) ===
        let frame_b = vec![200u8; 64 * 64 * 3];
        service.sensorimotor.vision_sensory.vision_frame_buffer = Some(frame_b);
        let result = service.cycle("trigger dilation + geodesic");

        let post_dim = service
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .unwrap()
            .manifold()
            .hdc_dim();
        let post_thermo = result.metadata.temporal.thermodynamic_load;

        println!("Initial dim: {initial_dim}, Post dim: {post_dim}");
        println!("Initial thermo: 0.0, Post thermo: {post_thermo}");

        assert_eq!(post_dim, 65536, "Should have dilated to Ultra");
        assert!(
            post_thermo > 0.05,
            "Thermodynamic load should have increased (got {post_thermo})"
        );
    }

    #[test]
    fn test_request_geodesic_triggers_mental_simulation() {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        let _ = fmt().with_env_filter(EnvFilter::new("warn")).try_init();

        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;
        let mut service = CognitiveLoopService::new(config).unwrap();

        // Ensure we have a manifold state to start from
        {
            let bridge = service
                .sensorimotor
                .vision_sensory
                .vision_bridge
                .as_mut()
                .unwrap();
            let frame = vec![128u8; 64 * 64 * 3];
            bridge
                .manifold_mut()
                .observe_frame(&frame, 64, 64, 3, 0.033);
        }

        // 3. Run enough cycles to ensure managers (VisionManager @ 11, Multimodal @ 31) run
        // We look for the cycle where the mental simulation actually triggered.
        let mut mental_movie = None;
        let mut last_result = None;
        let frame_a = vec![64u8; 64 * 64 * 3];
        let frame_b = vec![200u8; 64 * 64 * 3];

        for i in 0..33 {
            // Alternate frames to keep surprise/free-energy high
            let frame = if i % 2 == 0 {
                frame_a.clone()
            } else {
                frame_b.clone()
            };
            service.sensorimotor.vision_sensory.vision_frame_buffer = Some(frame);

            // Force free energy into metrics JUST in case managers need a specific range
            // (0.2 < F <= 0.5 triggers REQUEST_GEODESIC in VisionManager)
            service.carryover.quality.last_vision_free_energy = 0.4;

            let res = service.cycle("trigger geodesic");
            if res.mental_movie.is_some() {
                mental_movie = res.mental_movie.clone();
            }
            last_result = Some(res);
        }
        let result = last_result.unwrap();

        // 4. Verify that a geodesic path was generated in telemetry at some point
        let bridge = service
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_ref()
            .unwrap();
        let telemetry = bridge.manifold().telemetry();

        // The manifold telemetry persists the LAST geodesic path generated
        assert!(
            !telemetry.last_geodesic_path.is_empty(),
            "Mental simulation path should be persisted in manifold telemetry"
        );
        assert_eq!(
            telemetry.last_geodesic_length, 8,
            "Default geodesic length should be 8"
        );

        // 5. Verify the mental movie was captured in the result when it triggered
        let movie = mental_movie.expect("Mental movie should have been captured during the run");
        assert_eq!(movie.frames.len(), 8);
        assert_eq!(movie.width, 64);
        assert_eq!(movie.height, 64);
        assert!(!movie.frames[0].is_empty());
    }

    #[test]
    fn test_imagine_future_proactive_simulation() {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        let _ = fmt().with_env_filter(EnvFilter::new("warn")).try_init();

        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;
        let mut service = CognitiveLoopService::new(config).unwrap();

        // Initialize manifold state
        {
            let bridge = service
                .sensorimotor
                .vision_sensory
                .vision_bridge
                .as_mut()
                .unwrap();
            let frame = vec![128u8; 64 * 64 * 3];
            bridge
                .manifold_mut()
                .observe_frame(&frame, 64, 64, 3, 0.033);
        }

        let initial_thermo = service.thermodynamic_load();

        // Proactive imagination: 16 steps
        let movie = service
            .imagine_future(16)
            .expect("Imagine future should succeed");

        assert_eq!(movie.frames.len(), 16);
        assert_eq!(movie.path_length, 16);
        assert!(
            service.thermodynamic_load() > initial_thermo,
            "Thermodynamic load should increase after imagination"
        );

        // Verify frame quality (basic check)
        assert!(!movie.frames[0].is_empty());
        let expected_len = (64 * 64 * movie.channels) as usize;
        assert_eq!(movie.frames[0].len(), expected_len);
    }
}
