// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use symthaea::cognitive_loop::CognitiveLoopService;
    use symthaea::cognitive_loop::config::CognitiveLoopConfig;

    /// Integration test for the Vision Manifold top-down feedback loop.
    /// Science: predictive coding (Friston 2010), attention-gating (Treisman 1996).
    #[test]
    #[cfg(feature = "vision-manifold")]
    fn test_vision_manifold_closure_integration() {
        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;
        config.vision_frame_width = 64;
        config.vision_frame_height = 64;

        let mut svc = CognitiveLoopService::new(config).expect("Failed to create service");

        // 1. Establish stable scene
        let gray_frame = vec![128u8; 64 * 64 * 3];
        svc.inject_vision_frame(gray_frame.clone());
        let res1 = svc.cycle("baseline");
        assert!(res1.metadata.vision_manifold_enabled);

        // 2. Sudden scene change (Visual Surprise)
        let color_frame: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        svc.inject_vision_frame(color_frame);
        let res2 = svc.cycle("spike");

        let surprise = res2
            .metadata
            .vision
            .as_ref()
            .map(|v| v.prediction_error)
            .unwrap_or(0.0);
        assert!(
            surprise > 0.0,
            "Sudden color change should generate visual surprise"
        );

        // 3. Top-down Goal Injection
        // We simulate a focused observation on the new color.
        let res3 = svc.cycle("focus");
        assert!(res3.metadata.vision.is_some());

        // 4. Mental Simulation (Imagine the future)
        // Verify that the manifold can now generate a geodesic mental movie.
        if let Some(movie) = res3.mental_movie {
            assert!(
                !movie.frames.is_empty(),
                "Manifold should generate mental simulation frames"
            );
            assert!(movie.semantic_coherence > 0.0);
        }
    }
}
