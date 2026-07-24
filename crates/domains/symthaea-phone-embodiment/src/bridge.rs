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
    EmbodimentPlatform, EmbodimentResult, EmbodimentTelemetry, GROUNDING_TEMPORAL, MotorSafetyLevel,
};
use symthaea_core::hdc::ContinuousHV;
use symthaea_vision_manifold::{VisionConfig, VisionManifold};

use crate::actions::PhoneAction;
use crate::adb::AdbDevice;

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
    /// Last template match similarity (from goal-directed search).
    last_match_similarity: Option<f32>,
}

impl PhoneBridge {
    /// Create a new phone bridge for the given ADB device.
    ///
    /// # Arguments
    /// * `serial` — ADB device serial (e.g., "41201FDJG000UM")
    /// * `screen_w`, `screen_h` — Native screen resolution
    pub fn new(serial: impl Into<String>, screen_w: u32, screen_h: u32) -> Self {
        Self::with_resolution(serial, screen_w, screen_h, 64, 64)
    }

    /// Create a phone bridge with custom vision resolution.
    ///
    /// Higher resolution (e.g., 128×128) improves icon discrimination
    /// at the cost of ~4× processing time. 64×64 fits 20Hz budget;
    /// 128×128 runs at ~10 Hz but can distinguish individual app icons.
    pub fn with_resolution(
        serial: impl Into<String>,
        screen_w: u32,
        screen_h: u32,
        target_w: u32,
        target_h: u32,
    ) -> Self {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_object_binding = true;
        cfg.enable_temporal_binding = true;
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
            last_match_similarity: None,
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
        let img =
            image::load_from_memory(&png_bytes).map_err(|e| format!("Image decode failed: {e}"))?;

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

        Ok(self.observe_pixels(&pixels, dt))
    }

    /// Observe an externally supplied RGB frame (already at the vision target
    /// resolution). Both screenshot capture paths route through this; callers
    /// that source frames elsewhere (e.g., scrcpy/RDP streams) can use it
    /// directly.
    ///
    /// Updates the perception HV and prediction error reported by
    /// `encode_perception()` / `step_embodiment()` — these fields were
    /// previously never written, leaving the cognitive loop blind (constant
    /// zero HV, constant 0.0 PE).
    pub fn observe_pixels(
        &mut self,
        pixels: &[u8],
        dt: f32,
    ) -> symthaea_vision_manifold::VisionTelemetry {
        let tel = self
            .vision
            .observe_frame(pixels, self.target_w, self.target_h, 3, dt);
        self.last_perception = Some(self.vision.state().clone());
        self.last_prediction_error = tel.prediction_error;
        tel
    }

    /// Capture a screenshot, observe through the vision manifold, AND return the
    /// full-resolution RGBA frame for downstream consumers (e.g., SomaRdpServer).
    ///
    /// Unlike `capture_and_observe`, this keeps the native-resolution pixels alive
    /// so they can be fed into a remote-desktop codec. The vision manifold still
    /// runs on the downsampled target resolution — this method does both in one pass.
    pub fn capture_and_observe_rgba(
        &mut self,
        dt: f32,
    ) -> Result<(symthaea_vision_manifold::VisionTelemetry, Vec<u8>, u32, u32), String> {
        let png_bytes = self.adb.screenshot()?;
        let img =
            image::load_from_memory(&png_bytes).map_err(|e| format!("Image decode failed: {e}"))?;

        let (w, h) = img.dimensions();
        self.screen_w = w;
        self.screen_h = h;

        // Downsample copy for vision manifold (RGB).
        let resized = img.resize_exact(
            self.target_w,
            self.target_h,
            image::imageops::FilterType::Triangle,
        );
        let rgb_pixels: Vec<u8> = resized.to_rgb8().into_raw();
        let tel = self.observe_pixels(&rgb_pixels, dt);

        // Native-resolution RGBA copy for RDP codec.
        let rgba_pixels: Vec<u8> = img.to_rgba8().into_raw();
        Ok((tel, rgba_pixels, w, h))
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
                a.saliency
                    .partial_cmp(&b.saliency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let (screen_x, screen_y) =
                    self.grid_to_screen(best.centroid_row, best.centroid_col);
                // Only tap if Phi allows it
                if phi >= (PhoneAction::Tap { x: 0, y: 0 }).required_phi() {
                    PhoneAction::Tap {
                        x: screen_x,
                        y: screen_y,
                    }
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

    /// Goal-directed action proposal: select action to minimize distance to goal.
    ///
    /// Instead of tapping the most salient object (exploration), this finds the
    /// working memory object most similar to the goal HV and taps it (exploitation).
    /// If no match exceeds the threshold, falls back to exploration (swipe to
    /// reveal new content).
    ///
    /// # Arguments
    /// * `phi` — consciousness level for safety gating
    /// * `goal_hv` — target visual template (from `learn_template` or `load_template`)
    /// * `threshold` — minimum cosine similarity to count as a match (0.3–0.8)
    pub fn propose_goal_action(
        &mut self,
        phi: f64,
        goal_hv: &ContinuousHV,
        threshold: f32,
    ) -> PhoneAction {
        // Update safety level
        let phi_level = MotorSafetyLevel::from_phi(phi);
        self.current_safety = match self.safety_override {
            Some(override_level) => phi_level.max(override_level),
            None => phi_level,
        };

        if self.current_safety >= MotorSafetyLevel::Red {
            return PhoneAction::NoOp;
        }
        if self.current_safety >= MotorSafetyLevel::Orange {
            return PhoneAction::Screenshot;
        }

        // Strategy 1: Search all patches for best match to goal
        if let Some(match_result) = self.find_on_screen(goal_hv, threshold) {
            let (sx, sy, sim) = match_result;
            if phi >= (PhoneAction::Tap { x: 0, y: 0 }).required_phi() {
                let action = PhoneAction::Tap { x: sx, y: sy };
                self.proposed_action = Some(action.clone());
                self.last_match_similarity = Some(sim);
                return action;
            }
        }

        // Strategy 2: No match found — explore by swiping down
        if phi
            >= (PhoneAction::Swipe {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0,
                duration_ms: 0,
            })
            .required_phi()
        {
            let mid_x = self.screen_w / 2;
            let action = PhoneAction::Swipe {
                x1: mid_x,
                y1: self.screen_h * 2 / 3,
                x2: mid_x,
                y2: self.screen_h / 3,
                duration_ms: 300,
            };
            self.proposed_action = Some(action.clone());
            self.last_match_similarity = None;
            return action;
        }

        PhoneAction::Screenshot
    }

    /// Search the current frame's patches for the best match to a template HV.
    ///
    /// Uses **appearance-only matching**: unbinds the position component from
    /// each patch HV before comparing, so the template matches by visual content
    /// regardless of where it appears on screen. This is critical because the
    /// template was encoded from a crop (with its own position binding) that
    /// doesn't correspond to the icon's actual screen position.
    ///
    /// Returns `(screen_x, screen_y, similarity)` if a match exceeds the threshold.
    pub fn find_on_screen(
        &self,
        template_hv: &ContinuousHV,
        threshold: f32,
    ) -> Option<(u32, u32, f32)> {
        let patch_hvs = self.vision.last_patch_hvs();
        if patch_hvs.is_empty() {
            return None;
        }

        let patch_size = 8usize;
        let grid_cols = self.target_w as usize / patch_size;
        let encoder = self.vision.encoder();

        let mut best_sim = threshold;
        let mut best_pos: Option<(usize, usize)> = None;

        for (idx, patch_hv) in patch_hvs.iter().enumerate() {
            let row = idx / grid_cols.max(1);
            let col = idx % grid_cols.max(1);

            // Unbind position to get pure appearance HV
            let appearance = encoder.unbind_position(patch_hv, row, col);
            let sim = template_hv.similarity(&appearance);

            if sim > best_sim {
                best_sim = sim;
                best_pos = Some((row, col));
            }
        }

        best_pos.map(|(row, col)| {
            let (sx, sy) = self.grid_to_screen(row, col);
            (sx, sy, best_sim)
        })
    }

    /// Learn a visual template from an image file (PNG/JPG).
    ///
    /// Loads the image, resizes to the vision target resolution, encodes
    /// through the patch encoder, then **unbinds all position components**
    /// to produce a pure appearance template. This template will match
    /// the same visual content regardless of screen position.
    pub fn learn_template_from_file(&self, path: &std::path::Path) -> Result<ContinuousHV, String> {
        let img = image::open(path).map_err(|e| format!("Load failed: {e}"))?;
        let resized = img.resize_exact(
            self.target_w,
            self.target_h,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let pixels: Vec<u8> = rgb.into_raw();

        // Encode through a temporary encoder (same config as manifold)
        let cfg = self.vision.config().clone();
        let mut encoder =
            symthaea_vision_manifold::PatchHdcEncoder::new(&cfg, self.target_w, self.target_h);
        let (_, patch_hvs) = encoder.encode_frame(&pixels, self.target_w, self.target_h, 3);

        // Unbind position from each patch to get appearance-only HVs,
        // then bundle them into a single appearance template.
        let patch_size = 8usize;
        let grid_cols = self.target_w as usize / patch_size;
        let appearance_hvs: Vec<ContinuousHV> = patch_hvs
            .iter()
            .enumerate()
            .map(|(idx, phv)| {
                let row = idx / grid_cols.max(1);
                let col = idx % grid_cols.max(1);
                encoder.unbind_position(phv, row, col)
            })
            .collect();

        if appearance_hvs.is_empty() {
            return Err("No patches encoded".into());
        }

        let refs: Vec<&ContinuousHV> = appearance_hvs.iter().collect();
        Ok(ContinuousHV::bundle(&refs).normalize())
    }

    /// Last match similarity from `propose_goal_action` (if a match was found).
    pub fn last_match_similarity(&self) -> Option<f32> {
        self.last_match_similarity
    }

    /// Detect whether the last action caused a significant state transition.
    ///
    /// Uses imagination surprise and prediction error from the vision manifold
    /// to determine if the screen changed meaningfully. This is the active
    /// inference success signal: high surprise = reality changed.
    ///
    /// Returns `(is_transition, confidence)`:
    /// - `is_transition`: true if PE > threshold (screen changed significantly)
    /// - `confidence`: how confident we are (higher PE = more confident)
    ///
    /// Thresholds calibrated from empirical observation:
    /// - App opening: PE ≈ 0.10–0.20, motion ≈ 0.20–0.40
    /// - No change: PE ≈ 0.00–0.05
    pub fn detect_state_transition(&self) -> (bool, f32) {
        let tel = self.vision.telemetry();
        let pe = tel.prediction_error;
        let motion = tel.motion_surprise;
        let img_surp = tel.imagination_surprise;

        // Combine signals: PE is primary, motion and imagination surprise are secondary
        let combined = pe * 0.5 + motion * 0.3 + img_surp * 0.2;
        let threshold = 0.06; // Calibrated from YouTube opening: PE=0.133

        (combined > threshold, combined.min(1.0))
    }

    /// Learn a template from the current screen by extracting a specific patch.
    ///
    /// Instead of encoding a cropped image file (which has resolution mismatch),
    /// this extracts a single patch HV from the **live screen** at the given
    /// grid position. The template is already at the correct patch resolution.
    ///
    /// # Arguments
    /// * `grid_row`, `grid_col` — Position on the vision grid where the target
    ///   object is currently visible. Use the output of a previous observation
    ///   to identify which patch to learn from.
    pub fn learn_template_from_screen(
        &self,
        grid_row: usize,
        grid_col: usize,
    ) -> Option<ContinuousHV> {
        let patch_hvs = self.vision.last_patch_hvs();
        let patch_size = 8usize;
        let grid_cols = self.target_w as usize / patch_size;
        let idx = grid_row * grid_cols + grid_col;

        patch_hvs.get(idx).map(|phv| {
            self.vision
                .encoder()
                .unbind_position(phv, grid_row, grid_col)
        })
    }

    /// Save a learned template HV to disk for future sessions.
    ///
    /// Templates are stored as JSON files containing the raw f32 vector.
    /// Path: `data/phone-templates/{name}.json`
    pub fn save_template(&self, name: &str, hv: &ContinuousHV) -> Result<(), String> {
        let dir = std::path::Path::new("data/phone-templates");
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
        let path = dir.join(format!("{name}.json"));
        let values = hv.as_slice();
        let json = serde_json::to_string(values).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, json).map_err(|e| format!("write: {e}"))?;
        Ok(())
    }

    /// Load a previously saved template HV from disk.
    pub fn load_template(&self, name: &str) -> Result<ContinuousHV, String> {
        let path = std::path::Path::new("data/phone-templates").join(format!("{name}.json"));
        let json = std::fs::read_to_string(&path).map_err(|e| format!("read: {e}"))?;
        let values: Vec<f32> =
            serde_json::from_str(&json).map_err(|e| format!("deserialize: {e}"))?;
        Ok(ContinuousHV::from_vec(values))
    }

    /// Learn a template from a screen region by bundling multiple patches.
    ///
    /// Bundles appearance HVs from a rectangular region of the grid.
    pub fn learn_template_from_region(
        &self,
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
    ) -> Option<ContinuousHV> {
        let patch_hvs = self.vision.last_patch_hvs();
        let patch_size = 8usize;
        let grid_cols = self.target_w as usize / patch_size;
        let encoder = self.vision.encoder();

        let mut appearances: Vec<ContinuousHV> = Vec::new();
        for r in row_start..row_start + rows {
            for c in col_start..col_start + cols {
                let idx = r * grid_cols + c;
                if let Some(phv) = patch_hvs.get(idx) {
                    appearances.push(encoder.unbind_position(phv, r, c));
                }
            }
        }

        if appearances.is_empty() {
            return None;
        }
        let refs: Vec<&ContinuousHV> = appearances.iter().collect();
        Some(ContinuousHV::bundle(&refs).normalize())
    }

    /// Execute the proposed action (after user confirmation).
    pub fn confirm_and_execute(&mut self) -> Result<(), String> {
        let action = self.proposed_action.take().ok_or("No action proposed")?;
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
            PhoneAction::Swipe {
                x1,
                y1,
                x2,
                y2,
                duration_ms,
            } => self.adb.swipe(*x1, *y1, *x2, *y2, *duration_ms),
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

    /// Override the bridge's idea of the device's native screen size.
    /// Used by [`crate::streaming_bridge::StreamingPhoneBridge`] when the
    /// scrcpy stream's video header reports different dimensions than the
    /// caller initially supplied (rotation, encoder downscale).
    pub fn set_screen_dimensions(&mut self, width: u32, height: u32) {
        self.screen_w = width;
        self.screen_h = height;
    }

    /// Vision manifold's downsampled target dimensions (e.g. 64×64).
    pub fn vision_target_dims(&self) -> (u32, u32) {
        (self.target_w, self.target_h)
    }

    /// Mutable access to the vision manifold (for dream_replay, reset, etc.).
    pub fn vision_mut(&mut self) -> &mut VisionManifold {
        &mut self.vision
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
        self.vision.working_memory().map_or(Vec::new(), |wm| {
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
// ── EmbodimentBridge Implementation ──────────────────────────────────

impl PhoneBridge {
    /// Step the embodiment: capture screen → propose action → execute if not confirmation mode.
    ///
    /// This is the EmbodimentBridge-compatible step that maps thought HV to phone actions.
    /// The thought HV is used as a goal template for visual search when in Green safety.
    pub fn step_embodiment(
        &mut self,
        _thought_hv: &symthaea_core::hdc::ContinuousHV,
        dt: f32,
        phi: f64,
    ) -> symthaea_core::embodiment::EmbodimentResult {
        use symthaea_core::embodiment::*;

        // 1. Capture and observe
        let capture_ok = self.capture_and_observe(dt).is_ok();

        // 2. Propose action based on consciousness level
        let action = self.propose_action(phi);

        // 3. Execute if not in confirmation mode and capture succeeded
        let mut success = capture_ok;
        if capture_ok && !self.confirmation_mode && !matches!(action, PhoneAction::NoOp) {
            if self.execute_action(&action).is_err() {
                success = false;
            }
        }

        // 4. Compute prediction error from vision manifold
        let pe = self.last_prediction_error;
        self.total_steps += 1;

        EmbodimentResult {
            num_actuators: 5, // tap_x, tap_y, swipe_dx, swipe_dy, keyboard
            control_effort: if matches!(action, PhoneAction::NoOp | PhoneAction::Screenshot) {
                0.0
            } else {
                0.5
            },
            success,
            prediction_error: pe,
            safety_level: self.current_safety,
            epistemic_grounding: GROUNDING_SENSORIMOTOR,
            observation_confidence: grounding_from_prediction_error(pe),
        }
    }

    /// Encode current screen perception as a 16,384D ContinuousHV.
    ///
    /// Uses the last perception HV from the vision manifold observation,
    /// or returns a zero HV if no observation has been made yet.
    pub fn encode_perception_hv(&mut self) -> symthaea_core::hdc::ContinuousHV {
        self.last_perception.clone().unwrap_or_else(|| {
            symthaea_core::hdc::ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION)
        })
    }
}

impl symthaea_core::embodiment::EmbodimentBridge for PhoneBridge {
    fn step(
        &mut self,
        thought_hv: &symthaea_core::hdc::ContinuousHV,
        dt: f32,
        phi: f64,
    ) -> symthaea_core::embodiment::EmbodimentResult {
        self.step_embodiment(thought_hv, dt, phi)
    }

    fn encode_perception(&mut self) -> symthaea_core::hdc::ContinuousHV {
        self.encode_perception_hv()
    }

    fn reset(&mut self) {
        self.current_safety = MotorSafetyLevel::Red;
        self.safety_override = None;
        self.last_perception = None;
        self.last_action = None;
        self.proposed_action = None;
        self.total_steps = 0;
        self.last_prediction_error = 0.0;
    }

    fn safety_level(&self) -> MotorSafetyLevel {
        self.current_safety
    }

    fn set_safety_override(&mut self, level: MotorSafetyLevel) {
        self.safety_override = Some(level);
    }

    fn clear_safety_override(&mut self) {
        self.safety_override = None;
    }

    fn platform(&self) -> symthaea_core::embodiment::EmbodimentPlatform {
        symthaea_core::embodiment::EmbodimentPlatform::Phone
    }

    fn num_actuators(&self) -> usize {
        5
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn telemetry(&self) -> symthaea_core::embodiment::EmbodimentTelemetry {
        use symthaea_core::embodiment::*;
        EmbodimentTelemetry {
            total_steps: self.total_steps as u64,
            control_effort: 0.0,
            prediction_error: self.last_prediction_error,
            safety_level: self.current_safety,
            platform: "phone".to_string(),
            num_actuators: 5,
            epistemic_grounding: grounding_label(GROUNDING_SENSORIMOTOR).to_string(),
            observation_confidence: grounding_from_prediction_error(self.last_prediction_error),
            platform_specific: Vec::new(),
        }
    }
}

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
    fn test_embodiment_bridge_step_no_device() {
        // step() fails gracefully when no ADB device is connected
        let mut phone = PhoneBridge::new("nonexistent", 1008, 2244);
        let hv = symthaea_core::hdc::ContinuousHV::random(symthaea_core::hdc::HDC_DIMENSION, 42);
        let result = phone.step_embodiment(&hv, 0.033, 0.7);
        // Should not panic — failure is reported via success=false
        assert!(!result.success);
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

    #[test]
    fn test_observation_populates_perception_and_pe() {
        // Regression: last_perception/last_prediction_error were never
        // written by the capture path, so encode_perception() returned a
        // constant zero HV and PE stayed 0.0 — the loop was blind. Inject
        // frames directly (no device needed) and assert both fields go live.
        let mut phone = PhoneBridge::new("test", 1008, 2244);

        // Frame 1: flat gray
        let frame_a = vec![128u8; 64 * 64 * 3];
        phone.observe_pixels(&frame_a, 0.033);

        use symthaea_core::embodiment::EmbodimentBridge;
        let p = phone.encode_perception();
        assert!(
            p.as_slice().iter().any(|v| *v != 0.0),
            "perception HV must be non-zero after an observation"
        );

        // Frame 2: strongly different content — PE must be finite, >= 0
        let mut frame_b = vec![0u8; 64 * 64 * 3];
        for (i, px) in frame_b.iter_mut().enumerate() {
            *px = ((i * 7) % 256) as u8;
        }
        phone.observe_pixels(&frame_b, 0.033);
        assert!(
            phone.last_prediction_error.is_finite() && phone.last_prediction_error >= 0.0,
            "prediction error must be a live, finite signal, got {}",
            phone.last_prediction_error
        );
    }
}
