// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live-fire test for the phone vision pump (P1.1,
//! VISION_PROJECTION_REVIEW_2026-07-15.md): real Pixel screen pixels →
//! `PhoneCaptureHandle` → perception phase → the loop's own VisionBridge.
//!
//! Requires a USB-attached, ADB-authorized Android device; the serial comes
//! from the `PHONE_SERIAL` env var (falls back to the first `adb devices`
//! entry). Ignored by default — run with:
//!
//! ```text
//! cargo test -p symthaea --lib --features phone phone_frames -- --ignored --nocapture
//! ```

#[cfg(all(test, feature = "phone"))]
mod tests {
    use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn detect_serial() -> Option<String> {
        if let Ok(s) = std::env::var("PHONE_SERIAL") {
            if !s.trim().is_empty() {
                return Some(s.trim().to_string());
            }
        }
        let out = std::process::Command::new("adb")
            .arg("devices")
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .skip(1)
            .find_map(|l| {
                let mut parts = l.split_whitespace();
                match (parts.next(), parts.next()) {
                    (Some(serial), Some("device")) => Some(serial.to_string()),
                    _ => None,
                }
            })
    }

    #[test]
    #[ignore = "live-fire: requires a USB-attached, ADB-authorized Android device"]
    fn phone_frames_reach_the_loop() {
        let serial = detect_serial()
            .expect("no ADB device found — attach/authorize a phone or set PHONE_SERIAL");

        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;
        // Scope: this test proves the pump path, not dilation. Auto-dilation
        // on novel real-world frames would walk straight into the P0.5
        // balloon (dilation has its own suite).
        config.enable_vision_auto_dilation = false;
        let mut service = CognitiveLoopService::new(config).unwrap();

        service
            .start_phone_capture(&serial)
            .expect("phone capture failed to start (probe screenshot)");
        assert!(service.phone_capture_active());

        // Under consume-per-cycle retina semantics, vision telemetry appears
        // ONLY on cycles that actually perceived a frame — so telemetry
        // presence is direct proof a real phone frame entered the
        // VisionBridge. Capture cadence is ~500ms; pace cycles at 200ms and
        // give it up to 25 cycles (~5s) to see several frames.
        let mut frames_perceived = 0u32;
        for i in 0..25 {
            let result = service.cycle("looking at the phone screen");
            if result.metadata.vision.is_some() {
                frames_perceived += 1;
            }
            if frames_perceived >= 3 {
                println!("3 real phone frames perceived by cycle {i} — pump verified");
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        assert!(
            frames_perceived >= 3,
            "expected at least 3 cycles to perceive real phone frames, got {frames_perceived}"
        );

        service.stop_phone_capture();
        assert!(!service.phone_capture_active());
    }
}
