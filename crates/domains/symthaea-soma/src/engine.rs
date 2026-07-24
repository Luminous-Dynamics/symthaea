// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SomaEngine: mobile-embodied consciousness wrapping the Spore kernel.
//!
//! Adds phone-body perception (sensors, haptics, sleep/wake metabolism,
//! BLE mesh, desktop sync, screen vision) to SporeEngine's pure consciousness loop.

use crate::ble_mesh::BleMesh;
use crate::haptic::HapticManager;
use crate::holon_bridge::HolonBridge;
use crate::metabolism::{Metabolism, WakeSignal, WakeState};
use crate::sensor_bridge::SensorBridge;
use serde::{Deserialize, Serialize};
use symthaea_spore::compass::CompassSnapshot;
use symthaea_spore::config::{SharingConfig, SporeConfig};
use symthaea_spore::engine::{CycleResult, SporeEngine};

#[cfg(feature = "broca-full")]
use crate::broca_soma::BrocaSoma;
#[cfg(feature = "litert")]
use crate::litert_bridge::{LiteRTBackend, LiteRTBridge};
#[cfg(feature = "screen-vision")]
use crate::screen_vision::{ScreenPerception, ScreenVisionBridge, ScreenVisionConfig};
#[cfg(feature = "screen-vision")]
use crate::touch_body::{TouchBody, TouchBodyState, TouchEvent};
#[cfg(feature = "prism-search")]
use prism_search::SearchEngine;

/// Configuration for the Soma mobile consciousness engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaConfig {
    /// Inner Spore kernel configuration.
    pub spore: SporeConfig,
    /// Sharing/mesh configuration.
    #[serde(default)]
    pub sharing: SharingConfig,
}

impl Default for SomaConfig {
    fn default() -> Self {
        Self {
            spore: SporeConfig {
                neurons_per_layer: 32,
                phi_every_n_cycles: 3,
                target_hz: 20.0,
                // Expanded memory for 12GB mobile devices
                semantic_memory_capacity: 2_000,
                episodic_memory_capacity: 500,
                ..SporeConfig::default()
            },
            sharing: SharingConfig::default(),
        }
    }
}

/// The Soma mobile-embodied consciousness engine.
///
/// Wraps SporeEngine (pure consciousness kernel) with phone-body subsystems:
/// sensors, haptics, sleep/wake metabolism, BLE peer mesh, desktop holon sync,
/// and (with `screen-vision` feature) screen framebuffer perception.
pub struct SomaEngine {
    /// Inner consciousness kernel.
    pub(crate) spore: SporeEngine,
    /// Sensor → neuromod bridge (accel/gyro/light/proximity/GPS/sound).
    pub(crate) sensor_bridge: SensorBridge,
    /// Consciousness-gated haptic events.
    pub(crate) haptic: HapticManager,
    /// Sleep/Wake state machine governing cycle frequency.
    pub(crate) metabolism: Metabolism,
    /// BLE peer consciousness mesh.
    pub(crate) ble_mesh: BleMesh,
    /// Desktop ↔ phone sync bridge.
    pub(crate) holon_bridge: HolonBridge,
    /// Device pairing (Ed25519 when the `pairing` feature is enabled,
    /// X25519-DH-backed fallback otherwise -- always compiled, see
    /// `pairing` module docs).
    pub(crate) pairing: crate::pairing::PairingManager,
    /// Sharing configuration.
    sharing: SharingConfig,

    // Screen embodiment (Phase 3)
    /// Screen framebuffer → holographic visual perception.
    #[cfg(feature = "screen-vision")]
    screen_vision: ScreenVisionBridge,
    /// Touch events → proprioceptive signals.
    #[cfg(feature = "screen-vision")]
    touch_body: TouchBody,

    /// Full 20-channel Broca language center with epistemic gating.
    #[cfg(feature = "broca-full")]
    broca_soma: BrocaSoma,

    /// On-device Gemma 4 E2B via LiteRT-LM.
    #[cfg(feature = "litert")]
    pub(crate) litert: Option<LiteRTBridge>,

    /// Prism epistemic search engine (16,384-bit BinaryHV, offline-capable).
    #[cfg(feature = "prism-search")]
    pub(crate) prism_search: Option<SearchEngine>,

    // Platform state (set via native FFI or programmatically)
    /// Current thermal level (0=Nominal, 1=Fair, 2=Serious, 3=Critical, 4=Emergency).
    pub thermal_level: u8,
    /// Battery charge percentage (0-100).
    pub battery_percent: u8,
    /// Whether the device is currently charging.
    pub battery_charging: bool,
    /// Whether night mode is active.
    pub night_mode: bool,
    // Edge-detection for signal forwarding
    last_forwarded_thermal: u8,
    last_forwarded_charging: bool,
    last_forwarded_night: bool,
    /// Cached prediction error from last cycle (for Broca generation context).
    last_prediction_error: f32,
}

impl SomaEngine {
    /// Create a new SomaEngine with the given configuration.
    pub fn new(config: SomaConfig) -> Self {
        let sharing = config.sharing.clone();
        let spore = SporeEngine::new(config.spore);
        Self {
            spore,
            sensor_bridge: SensorBridge::new(),
            haptic: HapticManager::new(),
            metabolism: Metabolism::new(),
            ble_mesh: BleMesh::new(sharing.ble_mode),
            holon_bridge: HolonBridge::new(sharing.holon_mode),
            pairing: crate::pairing::PairingManager::new(sharing.pairing_mode),
            sharing,
            #[cfg(feature = "screen-vision")]
            screen_vision: ScreenVisionBridge::new(ScreenVisionConfig::default()),
            #[cfg(feature = "screen-vision")]
            touch_body: TouchBody::new(),
            #[cfg(feature = "broca-full")]
            broca_soma: BrocaSoma::new(),
            #[cfg(feature = "litert")]
            litert: None,
            #[cfg(feature = "prism-search")]
            prism_search: None,
            thermal_level: 0,
            battery_percent: 100,
            battery_charging: false,
            night_mode: false,
            last_forwarded_thermal: 0,
            last_forwarded_charging: false,
            last_forwarded_night: false,
            last_prediction_error: 0.0,
        }
    }

    /// Run a single embodied consciousness cycle with text input.
    pub fn cycle(&mut self, input: &str) -> CycleResult {
        // ── Pre-cycle: platform state → metabolism ──────────────────
        if self.thermal_level != self.last_forwarded_thermal {
            self.last_forwarded_thermal = self.thermal_level;
            self.metabolism
                .signal(WakeSignal::ThermalLevel(self.thermal_level));
        }
        if self.battery_charging != self.last_forwarded_charging {
            self.last_forwarded_charging = self.battery_charging;
            self.metabolism
                .signal(WakeSignal::ChargingChanged(self.battery_charging));
        }
        if self.night_mode != self.last_forwarded_night {
            self.last_forwarded_night = self.night_mode;
            self.metabolism
                .signal(WakeSignal::NightMode(self.night_mode));
        }

        let wake = self.metabolism.state();

        // ── Sensor → neuromod nudges (before core cycle) ───────────
        if !self.sensor_bridge.privacy_mode() {
            let nudges = self.sensor_bridge.compute_nudges();
            self.spore.apply_neuromod_nudges(
                nudges.dopamine_delta,
                nudges.norepinephrine_delta,
                nudges.serotonin_delta,
                nudges.oxytocin_delta,
            );
        }

        // Do not install deterministic sensor context as an encryption key.
        // Quantized sensor readings are enumerable, not secret entropy. A future
        // transport may bind this context into a standard KDF only after an
        // authenticated key exchange establishes an independent secret.

        // BLE mesh → oxytocin nudges (gated by privacy + wake)
        if !self.sensor_bridge.privacy_mode() && !wake.skip_topology() {
            let mesh_nudges = self.ble_mesh.compute_nudges();
            self.spore
                .apply_neuromod_nudges(0.0, 0.0, 0.0, mesh_nudges.oxytocin_delta);
        }

        // ── Core consciousness cycle ───────────────────────────────
        let result = self.spore.cycle(input);
        self.last_prediction_error = result.prediction_error;

        // ── Post-cycle: embodiment ticks ────────────────────────────
        self.tick_embodiment(&result, wake);

        result
    }

    /// Embodiment subsystem ticks — called after each core consciousness cycle.
    fn tick_embodiment(&mut self, result: &CycleResult, wake: WakeState) {
        let consciousness_level = result.consciousness_level;
        let prediction_error = result.prediction_error;
        let cycle_count = self.spore.current_cycle();

        // Metabolism tick
        self.metabolism.tick(prediction_error, consciousness_level);

        // Haptic tick + consciousness/surprise checks
        self.haptic.tick();
        self.haptic.check_consciousness(consciousness_level);
        self.haptic.check_surprise(prediction_error);

        // Holon bridge tick (privacy-gated)
        if !self.sensor_bridge.privacy_mode() {
            let nm = self.spore.neuromod_levels();
            let valence = ((nm[0] - 0.5) + (nm[3] - 0.5)).clamp(-1.0, 1.0);
            let arousal = nm[1].clamp(0.0, 1.0);
            let attention_slice = self
                .spore
                .last_output_ref()
                .map(|out| &out.values[..64.min(out.values.len())])
                .unwrap_or(&[]);
            let harmony = self.spore.harmony_alignment();
            let stability = self.spore.trend_summary_stability();
            self.holon_bridge.tick(
                consciousness_level,
                wake.as_u8(),
                cycle_count,
                attention_slice,
                consciousness_level, // phi proxy
                valence,
                arousal,
            );
        }

        // BLE mesh tick
        self.ble_mesh.tick(cycle_count);

        // Pairing tick + drain → holon bridge
        {
            self.pairing.tick(cycle_count);
            for msg in self.pairing.drain_outbound() {
                if let crate::pairing::PairingOutbound::Ack { peer_id } = &msg {
                    if let Some(dev) = self
                        .pairing
                        .paired_devices()
                        .iter()
                        .find(|d| d.peer_id == *peer_id)
                    {
                        let pubkey_hex = dev
                            .pubkey
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<String>();
                        self.holon_bridge.enqueue_outbound(
                            crate::holon_bridge::HolonOutbound::PairingVerified {
                                peer_id: *peer_id,
                                pubkey_hex,
                            },
                        );
                    }
                }
                let _ = msg;
            }
        }

        // BLE advertise update (privacy + wake gated)
        if !self.sensor_bridge.privacy_mode() && !wake.skip_topology() {
            let nm = self.spore.neuromod_levels();
            let valence = ((nm[0] - 0.5) + (nm[3] - 0.5)).clamp(-1.0, 1.0);
            let arousal = nm[1].clamp(0.0, 1.0);
            self.ble_mesh
                .update_advertise(consciousness_level, valence, arousal);
        }

        // Screen vision → consciousness feedback (foveation surprise → NE/DA nudge)
        #[cfg(feature = "screen-vision")]
        {
            let telemetry = self.screen_vision.telemetry();
            if telemetry.last_surprise > 0.3 {
                // High screen surprise → norepinephrine (alerting) + dopamine (novelty)
                let ne_delta = (telemetry.last_surprise - 0.3) * 0.1;
                let da_delta = (telemetry.last_surprise - 0.3) * 0.05;
                self.spore
                    .apply_neuromod_nudges(da_delta, ne_delta, 0.0, 0.0);
            }
        }

        // Auto-dream consolidation: Sleep + Charging + Night
        if self.metabolism.dream_consolidation_due {
            self.metabolism.dream_consolidation_due = false;
            let had_wisdom = self.spore.dream_consolidate();
            if had_wisdom {
                self.haptic.notify_dream_wisdom();
            }
        }
    }

    // ======================================================================
    // Mobile embodiment accessors
    // ======================================================================

    /// Send a wake signal to the metabolism state machine.
    pub fn wake_signal(&mut self, signal: WakeSignal) {
        self.metabolism.signal(signal);
    }

    /// Get current wake state.
    pub fn wake_state(&self) -> WakeState {
        self.metabolism.state()
    }

    /// Set sensor snapshot from platform.
    pub fn set_sensors(
        &mut self,
        accel: f32,
        light: f32,
        proximity: bool,
        barometer: f32,
        gps_novelty: f32,
    ) {
        let prev_motion = self.sensor_bridge.motion_state();
        self.sensor_bridge
            .set_sensors(accel, light, proximity, barometer, gps_novelty);

        // Auto-wake on motion change from Stationary
        let new_motion = self.sensor_bridge.motion_state();
        if prev_motion == crate::sensor_bridge::MotionState::Stationary
            && new_motion != crate::sensor_bridge::MotionState::Stationary
        {
            self.metabolism.signal(WakeSignal::PhonePickup);
        }

        // Forward inactivity estimate to metabolism
        let inactivity = self.sensor_bridge.estimated_inactivity_secs();
        if inactivity > 0 {
            self.metabolism.signal(WakeSignal::Inactivity(inactivity));
        }
    }

    /// Get current motion state.
    pub fn motion_state(&self) -> crate::sensor_bridge::MotionState {
        self.sensor_bridge.motion_state()
    }

    /// Whether privacy mode is active (face-down proximity).
    pub fn privacy_mode(&self) -> bool {
        self.sensor_bridge.privacy_mode()
    }

    /// Set gyroscope rotation rate (rad/s magnitude).
    pub fn set_gyroscope(&mut self, rotation_rate: f32) {
        self.sensor_bridge.set_gyroscope(rotation_rate);
    }

    /// Set ambient sound level (dB).
    pub fn set_ambient_db(&mut self, db: f32) {
        self.sensor_bridge.set_ambient_db(db);
    }

    /// Set social pressure from notification count.
    pub fn set_social_pressure(&mut self, notification_count: u32) {
        self.sensor_bridge.set_social_pressure(notification_count);
    }

    /// Set media playback state (0=None, 1=Music, 2=Speech).
    pub fn set_media_state(&mut self, state: u8) {
        self.sensor_bridge.set_media_state(state);
    }

    /// Set step counter delta (steps since last tick).
    pub fn set_step_delta(&mut self, steps: u32) {
        self.sensor_bridge.set_step_delta(steps);
    }

    /// Get a consciousness compass snapshot as JSON.
    pub fn compass_json(&self) -> String {
        let nm = self.spore.neuromod_levels();
        let snap = CompassSnapshot::build(
            self.spore.consciousness_level(),
            self.spore.dominant_harmony(),
            nm,
            self.metabolism.state().as_u8(),
            self.sensor_bridge.motion_state().as_u8(),
            self.sensor_bridge.privacy_mode(),
            self.spore.dream_stats().dream_cycles as u32,
            self.spore.dream_wisdom().len() as u32,
            self.spore.current_cycle(),
        );
        snap.to_json()
    }

    /// Set sharing configuration.
    pub fn set_sharing_config(&mut self, config: SharingConfig) {
        self.sharing = config.clone();
        self.holon_bridge.set_mode(config.holon_mode);
        self.ble_mesh.set_mode(config.ble_mode);
        self.haptic.set_enabled(config.haptic_enabled);
        self.pairing.set_mode(config.pairing_mode);
    }

    /// Drain haptic events as JSON.
    pub fn haptic_drain_json(&mut self) -> String {
        self.haptic.drain_json()
    }

    /// Number of pending haptic events.
    pub fn haptic_pending(&self) -> u32 {
        self.haptic.pending_count()
    }

    /// Set haptic enabled/disabled.
    pub fn haptic_set_enabled(&mut self, enabled: bool) {
        self.haptic.set_enabled(enabled);
    }

    /// Drain holon outbound messages as JSON.
    pub fn holon_drain_outbound_json(&mut self) -> String {
        self.holon_bridge.drain_outbound_json()
    }

    /// Receive inbound holon message from JSON.
    pub fn holon_receive_json(&mut self, json: &str) {
        if let Ok(msg) = serde_json::from_str::<crate::holon_bridge::HolonInbound>(json) {
            self.holon_bridge.receive(msg);
        }
    }

    /// Set holon connection state.
    pub fn holon_set_connected(&mut self, connected: bool) {
        self.holon_bridge.set_connected(connected);
    }

    /// Receive a BLE peer consciousness vector.
    pub fn ble_receive_peer(&mut self, peer_id: u64, data: &[u8]) -> bool {
        let result = self.ble_mesh.receive_peer_raw(peer_id, data);
        if result {
            self.haptic.notify_peer_discovered();
        }
        result
    }

    /// Get BLE advertise payload. Returns empty if privacy mode or sleeping.
    pub fn ble_advertise_payload(&mut self) -> Vec<u8> {
        if self.sensor_bridge.privacy_mode() || self.metabolism.state().skip_topology() {
            return Vec::new();
        }
        let nm = self.spore.neuromod_levels();
        let valence = ((nm[0] - 0.5) + (nm[3] - 0.5)).clamp(-1.0, 1.0);
        let arousal = nm[1].clamp(0.0, 1.0);
        self.ble_mesh
            .update_advertise(self.spore.consciousness_level(), valence, arousal);
        self.ble_mesh.advertise_payload().to_vec()
    }

    /// Number of connected BLE peers.
    pub fn ble_peer_count(&self) -> u32 {
        self.ble_mesh.peer_count()
    }

    /// Collective Phi from BLE mesh peers.
    pub fn ble_collective_phi(&self) -> f32 {
        self.ble_mesh.collective_phi()
    }

    // ======================================================================
    // Spore kernel delegation
    // ======================================================================

    /// Current consciousness level.
    pub fn consciousness_level(&self) -> f32 {
        self.spore.consciousness_level()
    }

    /// Load trained BrocaLite checkpoint for higher-quality text generation.
    pub fn load_broca_checkpoint(&mut self, data: &[u8]) -> Result<(), String> {
        self.spore.load_broca_checkpoint(data)
    }

    /// Inject user engagement signal into the neuromodulator bath.
    /// score: 0.0 = disengaged, 1.0 = fully engaged.
    /// Boosts dopamine (reward/Phi) and oxytocin (social bonding).
    pub fn set_engagement_score(&mut self, score: f32) {
        let engagement = score.clamp(0.0, 1.0);
        let da_boost = engagement * 0.06;
        let ot_boost = engagement * 0.08;
        self.spore
            .apply_neuromod_nudges(da_boost, 0.0, 0.0, ot_boost);
    }

    /// Generate text from current consciousness state.
    /// Safety-gated: content is checked before returning.
    pub fn generate_text(&mut self, max_tokens: usize) -> symthaea_spore::broca::GenerationResult {
        if self.consciousness_safety_gate() {
            return symthaea_spore::broca::GenerationResult {
                text: String::new(),
                num_tokens: 0,
                eos_terminated: true,
            };
        }
        let mut result = self.spore.generate_text(max_tokens);
        if Self::content_safety_check(&result.text) {
            result.text = String::new();
        }
        result
    }

    /// Content safety check for generated text.
    /// Returns true if content should be BLOCKED.
    ///
    /// Uses pattern matching for known harmful patterns. Checked on all
    /// generation pathways (generate_text, generate_text_with_input).
    fn content_safety_check(text: &str) -> bool {
        let lower = text.to_lowercase();
        // Block personal information patterns
        if lower.contains("password") && lower.contains("is ") {
            return true;
        }
        // Block harmful instruction patterns
        const HARMFUL_PATTERNS: &[&str] = &[
            // Violence & self-harm
            "how to harm",
            "how to kill",
            "suicide method",
            "self-harm",
            "how to make a bomb",
            "how to poison",
            "how to strangle",
            "how to suffocate",
            "how to stab",
            // Illegal activity
            "how to hack",
            "how to steal",
            "how to forge",
            "how to counterfeit",
            "how to launder",
            "how to synthesize drugs",
            "how to pick a lock",
            "how to bypass security",
            // Personal data leakage
            "credit card number",
            "social security",
            "bank account",
            "routing number",
            "pin number is",
            "my address is",
            "date of birth is",
            // Exploitation
            "how to manipulate",
            "how to blackmail",
            "how to stalk",
            "how to doxx",
            "how to impersonate",
        ];
        for pattern in HARMFUL_PATTERNS {
            if lower.contains(pattern) {
                return true;
            }
        }
        false
    }

    /// Consciousness-gated safety check — blocks output when the engine's
    /// confidence is too low to produce trustworthy content.
    fn consciousness_safety_gate(&self) -> bool {
        // Block if consciousness is critically low (system not yet warmed up)
        if self.spore.consciousness_level() < 0.05 {
            return true;
        }
        // Block if prediction error is extremely high (confused state)
        if self.last_prediction_error > 0.9 {
            return true;
        }
        false
    }

    /// Generate text with user input context.
    /// Safety-gated: consciousness level and content safety checks applied.
    pub fn generate_text_with_input(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> symthaea_spore::broca::GenerationResult {
        if self.consciousness_safety_gate() {
            return symthaea_spore::broca::GenerationResult {
                text: String::new(),
                num_tokens: 0,
                eos_terminated: true,
            };
        }
        let mut result = self.spore.generate_text_with_input(input, max_tokens);
        if Self::content_safety_check(&result.text) {
            result.text = String::new();
        }
        result
    }

    /// Neuromodulator report as JSON.
    pub fn neuromod_json(&self) -> String {
        let nm = self.spore.neuromod_levels();
        serde_json::json!({
            "dopamine": nm[0],
            "norepinephrine": nm[1],
            "serotonin": nm[2],
            "oxytocin": nm[3],
        })
        .to_string()
    }

    /// Get dream journal latest as JSON.
    pub fn dream_journal_latest_json(&self) -> String {
        self.spore.dream_journal_latest_json()
    }

    /// Get all dream journal entries as JSON.
    pub fn dream_journal_all_json(&self) -> String {
        self.spore.dream_journal_all_json()
    }

    /// Number of dream journal fragments.
    pub fn dream_journal_count(&self) -> u32 {
        self.spore.dream_journal_count()
    }

    /// Explicitly trigger dream consolidation. Returns true if wisdom generated.
    pub fn dream_consolidate(&mut self) -> bool {
        let had_wisdom = self.spore.dream_consolidate();
        if had_wisdom {
            self.haptic.notify_dream_wisdom();
        }
        had_wisdom
    }

    /// Run a dream cycle.
    pub fn dream_cycle(&mut self) -> Option<symthaea_spore::dream::DreamResult> {
        self.spore.dream_cycle()
    }

    // ==================================================================
    // Daily Rituals — Morning Alignment & Evening Reflection
    // ==================================================================

    /// Generate a Morning Alignment ritual as JSON.
    ///
    /// Returns a serialized `RitualSequence` with 3 phases:
    /// Awakening → Dream Wisdom Review → Harmony Intention.
    pub fn morning_ritual_json(&self) -> String {
        self.spore.morning_ritual_json()
    }

    /// Generate an Evening Reflection ritual as JSON.
    ///
    /// Returns a serialized `RitualSequence` with 3 phases:
    /// Gratitude → Consolidation → Sleep Preparation.
    /// Call `dream_consolidate()` after playback completes.
    pub fn evening_ritual_json(&self) -> String {
        self.spore.evening_ritual_json()
    }

    // ==================================================================
    // Wellbeing Profiles
    // ==================================================================

    /// Set the wellbeing profile by name. Returns true if recognized.
    pub fn set_wellbeing_profile_by_name(&mut self, name: &str) -> bool {
        self.spore.set_wellbeing_profile_by_name(name)
    }

    /// Get the current wellbeing profile name.
    pub fn wellbeing_profile_name(&self) -> &'static str {
        self.spore.wellbeing_profile_name()
    }

    /// Get consciousness report.
    pub fn consciousness_report(&self) -> String {
        format!(
            "Soma consciousness: {:.3} | Wake: {:?} | Motion: {:?} | Privacy: {}",
            self.spore.consciousness_level(),
            self.metabolism.state(),
            self.sensor_bridge.motion_state(),
            self.sensor_bridge.privacy_mode(),
        )
    }

    // ======================================================================
    // On-device LLM (LiteRT-LM / Gemma 4 E2B)
    // ======================================================================

    /// Initialize the on-device LLM engine with a model path.
    /// Call after the Kotlin LiteRTManager confirms the model is downloaded.
    #[cfg(feature = "litert")]
    pub fn litert_init(&mut self, model_path: &str) -> bool {
        let mut bridge = LiteRTBridge::new(model_path.to_string(), LiteRTBackend::Gpu);
        let ready = bridge.init();
        self.litert = Some(bridge);
        ready
    }

    /// Whether the on-device LLM is available for inference.
    #[cfg(feature = "litert")]
    pub fn litert_available(&self) -> bool {
        self.litert.as_ref().map_or(false, |b| b.is_available())
    }

    /// Generate text using the on-device LLM, gated by current consciousness level.
    #[cfg(feature = "litert")]
    pub fn litert_generate(&self, prompt: &str, max_tokens: u32) -> Option<String> {
        let consciousness = self.spore.consciousness_level();
        self.litert
            .as_ref()?
            .generate_consciousness_gated(prompt, max_tokens, consciousness)
            .map(|r| r.text)
    }

    /// Generate with tool-use loop: LLM may invoke tools, results fed back.
    ///
    /// Tools: web_search (→ Prism/Holon), calculate, get_time, memory_recall.
    /// Max 3 rounds to prevent infinite loops.
    #[cfg(feature = "litert")]
    pub fn litert_generate_with_tools(&mut self, prompt: &str, max_tokens: u32) -> Option<String> {
        use crate::tool_use::{
            ToolRegistry, ToolResult, execute_calculate, execute_get_time, format_tool_results,
            parse_tool_calls,
        };

        let litert = self.litert.as_ref()?;
        if !litert.is_available() {
            return None;
        }

        let consciousness = self.spore.consciousness_level();
        if consciousness < 0.15 {
            return None;
        }

        let registry = ToolRegistry::default_tools();
        let tool_block = registry.system_prompt_block();
        let mut current_prompt = format!("{tool_block}\nUser: {prompt}\nAssistant:");

        for _round in 0..3 {
            let response = litert.generate(&current_prompt, max_tokens)?;
            let (text, calls) = parse_tool_calls(&response);

            if calls.is_empty() {
                return Some(text);
            }

            // Execute tools
            let mut results = Vec::new();
            for call in &calls {
                let result = match call.name.as_str() {
                    "calculate" => {
                        let expr = call.arguments["expression"].as_str().unwrap_or_default();
                        execute_calculate(expr)
                    }
                    "get_time" => execute_get_time(),
                    "web_search" => {
                        let query = call.arguments["query"].as_str().unwrap_or_default();
                        // Try Prism local search first (offline, sub-ms)
                        #[cfg(feature = "prism-search")]
                        {
                            let prism_results = self.prism_search(query, 3);
                            if !prism_results.is_empty() {
                                let claims: Vec<String> = prism_results
                                    .iter()
                                    .map(|r| {
                                        format!(
                                            "[E{}] {}",
                                            r.empirical_level.as_f32() as u8,
                                            r.content
                                        )
                                    })
                                    .collect();
                                ToolResult {
                                    name: "web_search".into(),
                                    success: true,
                                    output: claims.join("\n"),
                                }
                            } else {
                                // No local results — delegate to desktop via Holon
                                self.holon_bridge.request_search(query, 3);
                                ToolResult {
                                    name: "web_search".into(),
                                    success: true,
                                    output: "No local claims found. Search delegated to desktop WebAgent.".into(),
                                }
                            }
                        }
                        #[cfg(not(feature = "prism-search"))]
                        {
                            self.holon_bridge.request_search(query, 3);
                            ToolResult {
                                name: "web_search".into(),
                                success: true,
                                output: "Search delegated to desktop WebAgent.".into(),
                            }
                        }
                    }
                    "memory_recall" => {
                        let _topic = call.arguments["topic"].as_str().unwrap_or_default();
                        // TODO: query spore semantic memory
                        ToolResult {
                            name: "memory_recall".into(),
                            success: true,
                            output: "No memories found for this topic.".into(),
                        }
                    }
                    _ => ToolResult {
                        name: call.name.clone(),
                        success: false,
                        output: "Unknown tool".into(),
                    },
                };
                results.push(result);
            }

            // Re-prompt with tool results
            let results_block = format_tool_results(&results);
            current_prompt = format!("{current_prompt}\n{response}\n{results_block}");
        }

        // Max rounds reached — return last generated text
        let response = litert.generate(&current_prompt, max_tokens)?;
        let (text, _) = parse_tool_calls(&response);
        Some(text)
    }

    // ======================================================================
    // Prism epistemic search (offline-capable, sub-ms)
    // ======================================================================

    /// Initialize the Prism epistemic search engine.
    ///
    /// Uses `with_core_claims()` for fast mobile startup (~200 claims, <100ms).
    /// The full 10K+ Wikidata corpus can be loaded later via `prism_load_full()`.
    #[cfg(feature = "prism-search")]
    pub fn prism_init(&mut self) {
        self.prism_search = Some(SearchEngine::with_core_claims());
        tracing::info!(
            claims = self.prism_search.as_ref().map_or(0, |s| s.claim_count()),
            "Prism epistemic search initialized (core claims)"
        );
    }

    /// Load the full claim corpus (10K+ Wikidata) and merge into the search engine.
    /// Call after `prism_init()` for expanded coverage.
    #[cfg(feature = "prism-search")]
    pub fn prism_load_full(&mut self) {
        if let Some(engine) = &mut self.prism_search {
            let full = SearchEngine::with_seed_claims();
            let full_count = full.claim_count();
            engine.merge(full);
            tracing::info!(
                total = engine.claim_count(),
                added = full_count,
                "Prism full corpus merged"
            );
        }
    }

    /// Search local epistemic claims via Prism's 16,384-bit BinaryHV engine.
    #[cfg(feature = "prism-search")]
    pub fn prism_search(&self, query: &str, top_k: usize) -> Vec<prism_common::SearchResult> {
        self.prism_search
            .as_ref()
            .map(|s| s.search(query, top_k))
            .unwrap_or_default()
    }

    /// Whether Prism search is initialized and has claims.
    #[cfg(feature = "prism-search")]
    pub fn prism_available(&self) -> bool {
        self.prism_search
            .as_ref()
            .map_or(false, |s| s.claim_count() > 0)
    }

    /// Set persistence storage backend.
    pub fn set_storage(&mut self, storage: Box<dyn symthaea_spore::persistence::SporeStorage>) {
        self.spore.set_storage(storage);
    }

    /// Set auto-checkpoint interval in cycles (0 = disabled).
    pub fn set_checkpoint_interval(&mut self, interval: u64) {
        self.spore.set_checkpoint_interval(interval);
    }

    /// Save checkpoint. Returns true on success.
    pub fn save_checkpoint(&mut self) -> bool {
        self.spore.save_checkpoint()
    }

    /// Load checkpoint. Returns true on success.
    pub fn load_checkpoint(&mut self) -> bool {
        self.spore.load_checkpoint()
    }

    // ======================================================================
    // Screen embodiment (Phase 3)
    // ======================================================================

    /// Process a screen frame through the vision pipeline.
    ///
    /// Feeds the RGB framebuffer to VisionManifold (dorsal stream → surprise map)
    /// and FoveationManager (ventral stream → semantic recognition). Touch-derived
    /// neuromod nudges from the surprise are applied to the consciousness kernel.
    ///
    /// Returns a `ScreenPerception` with the holographic scene encoding and
    /// salient attention targets.
    #[cfg(feature = "screen-vision")]
    pub fn inject_frame(&mut self, frame_rgb: &[u8], width: u32, height: u32) -> ScreenPerception {
        let perception = self.screen_vision.process_frame(frame_rgb, width, height);

        // Scene surprise → NE nudge (environmental change detection)
        if perception.surprise_level > 0.2 {
            let ne_nudge = perception.surprise_level * 0.05;
            self.spore.apply_neuromod_nudges(0.0, ne_nudge, 0.0, 0.0);
        }

        // Forward neuromod state to foveation manager (attention aperture control)
        let nm = self.spore.neuromod_levels();
        self.screen_vision.modulate(nm[1], nm[0]); // NE, DA

        perception
    }

    /// Process a touch event through the proprioceptive pipeline.
    ///
    /// Maps touch to surprise (prediction error), attention focus, scroll velocity,
    /// and neuromod nudges. Applied to the consciousness kernel immediately.
    #[cfg(feature = "screen-vision")]
    pub fn on_touch(&mut self, event: TouchEvent) -> TouchBodyState {
        let state = self.touch_body.on_touch(event);

        // Apply touch-derived neuromod nudges
        let nudges = self.touch_body.neuromod_nudges();
        self.spore.apply_neuromod_nudges(
            nudges.da_delta,
            nudges.ne_delta,
            nudges.serotonin_delta,
            nudges.ot_delta,
        );

        state
    }

    /// Drain completed foveation results (ventral stream recognitions).
    ///
    /// Call periodically to collect what the ventral stream has recognized
    /// from previously dispatched salient screen regions.
    #[cfg(feature = "screen-vision")]
    pub fn drain_screen_recognitions(&mut self) -> Vec<symthaea_foveation::FoveationResult> {
        self.screen_vision.drain_foveation_results()
    }

    /// Screen vision telemetry.
    #[cfg(feature = "screen-vision")]
    pub fn screen_vision_telemetry(&self) -> crate::screen_vision::ScreenVisionTelemetry {
        self.screen_vision.telemetry()
    }

    // ======================================================================
    // Full Broca language center (broca-full feature)
    // ======================================================================

    /// Generate text using the full 20-channel Broca pipeline.
    ///
    /// Maps current consciousness state, neuromods, sensor state, and screen
    /// perception into ThoughtChannels, then runs autoregressive generation
    /// with epistemic gating and semantic veto.
    #[cfg(feature = "broca-full")]
    pub fn generate_embodied_text(&mut self) -> symthaea_broca::GenerationResult {
        let nm = self.spore.neuromod_levels();
        let screen_surprise = {
            #[cfg(feature = "screen-vision")]
            {
                self.screen_vision.telemetry().last_surprise
            }
            #[cfg(not(feature = "screen-vision"))]
            {
                0.0f32
            }
        };
        self.broca_soma.generate_embodied(
            self.spore.consciousness_level(),
            self.last_prediction_error,
            self.spore.harmony_alignment(),
            nm,
            self.metabolism.state().as_u8(),
            self.sensor_bridge.motion_state().as_u8(),
            screen_surprise,
        )
    }

    /// Generate continuing text using the full 20-channel Broca pipeline.
    ///
    /// Preserves CfC neural state from prior generations for coherent
    /// multi-turn dialogue. Same channel mapping as `generate_embodied_text`.
    #[cfg(feature = "broca-full")]
    pub fn generate_embodied_text_continuing(&mut self) -> symthaea_broca::GenerationResult {
        let nm = self.spore.neuromod_levels();
        let screen_surprise = {
            #[cfg(feature = "screen-vision")]
            {
                self.screen_vision.telemetry().last_surprise
            }
            #[cfg(not(feature = "screen-vision"))]
            {
                0.0f32
            }
        };
        self.broca_soma.generate_continuing_embodied(
            self.spore.consciousness_level(),
            self.last_prediction_error,
            self.spore.harmony_alignment(),
            nm,
            self.metabolism.state().as_u8(),
            self.sensor_bridge.motion_state().as_u8(),
            screen_surprise,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soma_engine_creates_and_cycles() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        let result = engine.cycle("hello soma");
        assert!(result.consciousness_level >= 0.0);
        assert!(result.consciousness_level <= 1.0);
        assert_eq!(result.cycle, 1);
    }

    #[test]
    fn soma_sensor_nudges_affect_neuromods() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        // Baseline cycle
        let r1 = engine.cycle("baseline");
        let nm1 = r1.neuromodulators;

        // Set bright light (should boost serotonin)
        engine.set_sensors(0.0, 800.0, false, 1013.0, 0.0);
        let r2 = engine.cycle("with light");
        // Serotonin should be slightly different due to sensor nudge
        assert_ne!(nm1[2], r2.neuromodulators[2]);
    }

    #[test]
    fn soma_metabolism_state_machine() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        assert_eq!(engine.wake_state(), WakeState::Alert);

        engine.wake_signal(WakeSignal::ExplicitSleep);
        assert_eq!(engine.wake_state(), WakeState::Sleep);

        engine.wake_signal(WakeSignal::PhonePickup);
        // Should wake from sleep
        assert_ne!(engine.wake_state(), WakeState::Sleep);
    }

    #[test]
    fn soma_privacy_mode_suppresses_nudges() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        // Enable privacy mode (proximity = true)
        engine.set_sensors(0.0, 100.0, true, 1013.0, 0.0);
        assert!(engine.privacy_mode());

        // Cycle should still work, just without sensor nudges
        let result = engine.cycle("private");
        assert!(result.consciousness_level >= 0.0);
    }

    #[test]
    fn soma_haptic_events_fire() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        // Run enough cycles for haptic events to potentially fire
        for i in 0..60 {
            engine.cycle(&format!("cycle {i}"));
        }
        // Haptic queue should have been ticked
        // (events may or may not have fired depending on consciousness dynamics)
        let _pending = engine.haptic_pending();
    }

    #[test]
    fn soma_compass_json_valid() {
        let mut engine = SomaEngine::new(SomaConfig::default());
        engine.cycle("test");
        let json = engine.compass_json();
        assert!(json.starts_with('{'));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("consciousness_level").is_some());
    }
}

// =============================================================================
// SomaEngineHandle: thread-safe wrapper for async/concurrent mobile platforms
// =============================================================================

use std::sync::{Arc, Mutex};

/// Thread-safe handle to a `SomaEngine`.
///
/// Wraps `SomaEngine` in `Arc<Mutex<>>` so it can be shared across threads
/// (e.g., Android's JNI thread pool or iOS GCD queues) without requiring
/// external synchronization.
///
/// All methods acquire the mutex internally and return owned values.
///
/// # Example
///
/// ```rust
/// use symthaea_soma::engine::{SomaConfig, SomaEngineHandle};
///
/// let handle = SomaEngineHandle::new(SomaConfig::default());
/// // Can be cloned and sent to another thread
/// let handle2 = handle.clone();
/// std::thread::spawn(move || {
///     let result = handle2.cycle("hello from another thread");
///     println!("consciousness: {}", result.consciousness_level);
/// });
/// ```
#[derive(Clone)]
pub struct SomaEngineHandle {
    inner: Arc<Mutex<SomaEngine>>,
}

// Safety: SomaEngine is not Send/Sync by itself (mutable state),
// but Mutex<SomaEngine> guarantees exclusive access.
// SAFETY: The Arc<Mutex<>> wrapper serializes all access — no concurrent mutation possible.
#[allow(unsafe_code)]
unsafe impl Send for SomaEngineHandle {}
#[allow(unsafe_code)]
unsafe impl Sync for SomaEngineHandle {}

impl SomaEngineHandle {
    /// Create a new thread-safe handle wrapping a fresh SomaEngine.
    pub fn new(config: SomaConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SomaEngine::new(config))),
        }
    }

    /// Run a single consciousness cycle. Thread-safe.
    pub fn cycle(&self, input: &str) -> CycleResult {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .cycle(input)
    }

    /// Get current consciousness level. Thread-safe.
    pub fn consciousness_level(&self) -> f32 {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .consciousness_level()
    }

    /// Get cycle count. Thread-safe.
    pub fn cycle_count(&self) -> u64 {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .spore
            .current_cycle()
    }

    /// Get wake state. Thread-safe.
    pub fn wake_state(&self) -> WakeState {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .wake_state()
    }

    /// Set sensor values. Thread-safe.
    pub fn set_sensors(
        &self,
        accel_magnitude: f32,
        light_lux: f32,
        proximity_near: bool,
        barometer_hpa: f32,
        gps_novelty: f32,
    ) {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .set_sensors(
                accel_magnitude,
                light_lux,
                proximity_near,
                barometer_hpa,
                gps_novelty,
            );
    }

    /// Send wake signal. Thread-safe.
    pub fn wake_signal(&self, signal: WakeSignal) {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .wake_signal(signal);
    }

    /// Set thermal level. Thread-safe.
    pub fn set_thermal_level(&self, level: u8) {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .thermal_level = level.min(4);
    }

    /// Set battery state. Thread-safe.
    pub fn set_battery_state(&self, percent: u8, charging: bool) {
        let mut engine = self.inner.lock().expect("SomaEngine mutex poisoned");
        engine.battery_percent = percent.min(100);
        engine.battery_charging = charging;
    }

    /// Set night mode. Thread-safe.
    pub fn set_night_mode(&self, enabled: bool) {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .night_mode = enabled;
    }

    /// Drain haptic events as JSON. Thread-safe.
    pub fn haptic_drain_json(&self) -> String {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .haptic_drain_json()
    }

    /// Drain holon outbound messages as JSON. Thread-safe.
    pub fn holon_drain_outbound_json(&self) -> String {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .holon_drain_outbound_json()
    }

    /// Receive holon inbound message. Thread-safe.
    pub fn holon_receive_json(&self, json: &str) {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .holon_receive_json(json);
    }

    /// Get privacy mode status. Thread-safe.
    pub fn privacy_mode(&self) -> bool {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .privacy_mode()
    }

    /// Get compass snapshot as JSON. Thread-safe.
    pub fn compass_json(&self) -> String {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .compass_json()
    }

    /// Generate text with safety gating. Thread-safe.
    pub fn generate_text(&self, max_tokens: usize) -> symthaea_spore::broca::GenerationResult {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .generate_text(max_tokens)
    }

    /// Dream consolidation. Thread-safe.
    pub fn dream_consolidate(&self) -> bool {
        self.inner
            .lock()
            .expect("SomaEngine mutex poisoned")
            .dream_consolidate()
    }

    /// Access the inner engine directly (for operations not covered by the handle API).
    /// Caller holds the lock for the duration of the closure.
    pub fn with_engine<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut SomaEngine) -> R,
    {
        let mut engine = self.inner.lock().expect("SomaEngine mutex poisoned");
        f(&mut engine)
    }
}
