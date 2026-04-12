// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PhoneBridge — EmbodimentBridge implementation for ADB-controlled phone.
//!
//! Closes the perception-action loop:
//! 1. Capture screenshot → VisionManifold (P1-P8) → scene understanding
//! 2. Working memory selects highest-saliency attended object
//! 3. Phi-gated action dispatch → ADB tap/swipe/type
//! 4. Next screenshot → prediction error → active inference learning

use image::GenericImageView;
use symthaea_core::embodiment::{
    EmbodimentPlatform, EmbodimentResult, EmbodimentTelemetry, MotorSafetyLevel,
    GROUNDING_TEMPORAL,
};
use symthaea_core::hdc::ContinuousHV;
use symthaea_vision_manifold::{VisionConfig, VisionManifold};

use crate::adb::AdbDevice;
use crate::actions::PhoneAction;

/// Phone screen embodiment bridge.
///
/// Wraps ADB device control + VisionManifold perception into a single
/// `EmbodimentBridge`-compatible interface. The cognitive loop sees the
/// phone as just another body to inhabit.
pub struct PhoneBridge {
    adb: AdbDevice,
    vision: VisionManifold,
    /// Native screen resolution (from screenshot).
    screen_w: u32,
    screen_h: u32,
    /// Vision manifold target resolution.
    target_w: u32,
    target_h: u32,
    /// Current safety level (from Phi + overrides).
    current_safety: MotorSafetyLevel,
    safety_override: Option<MotorSafetyLevel>,
    /// Last perception HV.
    last_perception: Option<ContinuousHV>,
    /// Last executed action.
    last_action: Option<PhoneAction>,
    /// Last proposed action (for confirmation mode).
    proposed_action: Option<PhoneAction>,
    /// Total embodiment steps.
    total_steps: usize,
    /// Last prediction error.
    last_prediction_error: f32,
    /// Whether confirmation mode is active (default: true).
    confirmation_mode: bool,
}

impl PhoneBridge {
    /// Create a new phone bridge for the given ADB device.
    ///
    /// # Arguments
    /// * `serial` — ADB device serial (e.g., "41201FDJG000UM")
    /// * `screen_w`, `screen_h` — Native screen resolution
    pub fn new(serial: impl Into<String>, screen_w: u32, screen_h: u32) -> Self {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_object_binding = true;
        cfg.enable_temporal_binding = true;

        let target_w = 64;
        let target_h = 64;
        let mut vision = VisionManifold::new(cfg, target_w, target_h);
        vision.enable_object_memory(16);
        vision.enable_working_memory(4);
        vision.enable_scene_graph();
        vision.enable_scene_memory(16);

        Self {
            adb: AdbDevice::new(serial),
            vision,
            screen_w,
            screen_h,
            target_w,
            target_h,
            current_safety: MotorSafetyLevel::Red, // Start at Red until first Phi assessment
            safety_override: None,
            last_perception: None,
            last_action: None,
            proposed_action: None,
            total_steps: 0,
            last_prediction_error: 0.0,
            confirmation_mode: true,
        }
    }

    /// Capture a screenshot and process through the vision manifold.
    ///
    /// Returns the vision telemetry from the observation.
    pub fn capture_and_observe(
        &mut self,
        dt: f32,
    ) -> Result<symthaea_vision_manifold::VisionTelemetry, String> {
        let png_bytes = self.adb.screenshot()?;
        let img = image::load_from_memory(&png_bytes)
            .map_err(|e| format!("Image decode failed: {e}"))?;

        // Update native screen dimensions from actual capture
        let (w, h) = img.dimensions();
        self.screen_w = w;
        self.screen_h = h;

        // Resize to vision target
        let resized = img.resize_exact(
            self.target_w,
            self.target_h,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let pixels: Vec<u8> = rgb.into_raw();

        let tel = self.vision.observe_frame(&pixels, self.target_w, self.target_h, 3, dt);
        Ok(tel)
    }

    /// Map a vision grid position to phone screen coordinates.
    ///
    /// Converts from the 8×8 grid (at 64×64) to native screen pixels.
    pub fn grid_to_screen(&self, grid_row: usize, grid_col: usize) -> (u32, u32) {
        let patch_size = 8; // VisionConfig default
        // Center of the patch
        let vision_x = (grid_col * patch_size + patch_size / 2) as f32;
        let vision_y = (grid_row * patch_size + patch_size / 2) as f32;
        // Scale to screen
        let screen_x = (vision_x * self.screen_w as f32 / self.target_w as f32) as u32;
        let screen_y = (vision_y * self.screen_h as f32 / self.target_h as f32) as u32;
        (screen_x, screen_y)
    }

    /// Propose an action based on current visual state and consciousness level.
    ///
    /// In confirmation mode, this stores the proposal without executing.
    /// Call `confirm_and_execute()` to actually dispatch it.
    pub fn propose_action(&mut self, phi: f64) -> PhoneAction {
        // Update safety level
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };

        // At Red: no action
        if self.current_safety >= MotorSafetyLevel::Red {
            return PhoneAction::NoOp;
        }

        // At Orange: screenshot only
        if self.current_safety >= MotorSafetyLevel::Orange {
            return PhoneAction::Screenshot;
        }

        // Find the highest-saliency working memory object
        let action = if let Some(wm) = self.vision.working_memory() {
            if let Some(best) = wm.slots().iter().max_by(|a, b| {
                a.saliency.partial_cmp(&b.saliency).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let (screen_x, screen_y) = self.grid_to_screen(best.centroid_row, best.centroid_col);
                // Only tap if Phi allows it
                if phi >= (PhoneAction::Tap { x: 0, y: 0 }).required_phi() {
                    PhoneAction::Tap { x: screen_x, y: screen_y }
                } else {
                    PhoneAction::Screenshot
                }
            } else {
                PhoneAction::Screenshot
            }
        } else {
            PhoneAction::Screenshot
        };

        self.proposed_action = Some(action.clone());
        action
    }

    /// Execute the proposed action (after user confirmation).
    pub fn confirm_and_execute(&mut self) -> Result<(), String> {
        let action = self.proposed_action.take()
            .ok_or("No action proposed")?;
        self.execute_action(&action)
    }

    /// Execute an action directly (bypasses confirmation mode).
    pub fn execute_action(&mut self, action: &PhoneAction) -> Result<(), String> {
        match action {
            PhoneAction::NoOp => Ok(()),
            PhoneAction::Screenshot => {
                // Screenshot is already captured by capture_and_observe
                Ok(())
            }
            PhoneAction::Back => self.adb.back(),
            PhoneAction::Home => self.adb.home(),
            PhoneAction::OpenUrl { url } => self.adb.open_url(url),
            PhoneAction::Swipe { x1, y1, x2, y2, duration_ms } => {
                self.adb.swipe(*x1, *y1, *x2, *y2, *duration_ms)
            }
            PhoneAction::Tap { x, y } => self.adb.tap(*x, *y),
            PhoneAction::Type { text } => self.adb.input_text(text),
        }?;
        self.last_action = Some(action.clone());
        Ok(())
    }

    /// Access the underlying vision manifold.
    pub fn vision(&self) -> &VisionManifold {
        &self.vision
    }

    /// Access the ADB device.
    pub fn adb(&self) -> &AdbDevice {
        &self.adb
    }

    /// Whether confirmation mode is active.
    pub fn confirmation_mode(&self) -> bool {
        self.confirmation_mode
    }

    /// Set confirmation mode (default: true).
    pub fn set_confirmation_mode(&mut self, mode: bool) {
        self.confirmation_mode = mode;
    }

    /// Last proposed action (for display in confirmation UI).
    pub fn proposed(&self) -> Option<&PhoneAction> {
        self.proposed_action.as_ref()
    }

    /// Scene description for display.
    pub fn describe_scene(&self) -> Vec<(String, String, String)> {
        self.vision.describe_scene()
    }

    /// Working memory contents.
    pub fn working_memory_summary(&self) -> Vec<(u64, f32, u32, u32)> {
        self.vision
            .working_memory()
            .map_or(Vec::new(), |wm| {
                wm.slots()
                    .iter()
                    .map(|s| {
                        let (sx, sy) = self.grid_to_screen(s.centroid_row, s.centroid_col);
                        (s.track_id, s.saliency, sx, sy)
                    })
                    .collect()
            })
    }
}

// EmbodimentBridge implementation — allows PhoneBridge to be used
// as a drop-in embodiment platform in the cognitive loop.
//
// NOTE: This requires adding `Phone` to the `EmbodimentPlatform` enum
// and the feature flag `phone` to the cfg gate in sensorimotor_execution.rs.
// For now, this is used standalone via the example binary.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_to_screen_mapping() {
        let phone = PhoneBridge::new("test", 1008, 2244);
        // Center of grid cell (0,0) → top-left area of screen
        let (x, y) = phone.grid_to_screen(0, 0);
        assert!(x < 100, "Top-left should be in first 100 pixels, got {x}");
        assert!(y < 200, "Top-left should be in first 200 pixels, got {y}");

        // Center of grid cell (7,7) → bottom-right area of screen
        let (x, y) = phone.grid_to_screen(7, 7);
        assert!(x > 800, "Bottom-right x should be >800, got {x}");
        assert!(y > 1800, "Bottom-right y should be >1800, got {y}");
    }

    #[test]
    fn test_safety_starts_at_red() {
        let phone = PhoneBridge::new("test", 1008, 2244);
        assert_eq!(phone.current_safety, MotorSafetyLevel::Red);
    }

    #[test]
    fn test_phi_gates_actions() {
        let mut phone = PhoneBridge::new("test", 1008, 2244);
        // Very low Phi → NoOp
        let action = phone.propose_action(0.05);
        assert!(matches!(action, PhoneAction::NoOp));

        // Medium Phi → Screenshot
        let action = phone.propose_action(0.25);
        assert!(matches!(action, PhoneAction::Screenshot));
    }

    #[test]
    fn test_confirmation_mode_default_on() {
        let phone = PhoneBridge::new("test", 1008, 2244);
        assert!(phone.confirmation_mode());
    }
}
