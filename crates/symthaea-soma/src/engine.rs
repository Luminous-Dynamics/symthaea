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
#[cfg(feature = "screen-vision")]
use crate::screen_vision::{ScreenPerception, ScreenVisionBridge, ScreenVisionConfig};
#[cfg(feature = "screen-vision")]
use crate::touch_body::{TouchBody, TouchBodyState, TouchEvent};

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
    /// Ed25519 device pairing.
    #[cfg(feature = "pairing")]
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
            #[cfg(feature = "pairing")]
            pairing: crate::pairing::PairingManager::new(sharing.pairing_mode),
            sharing,
            #[cfg(feature = "screen-vision")]
            screen_vision: ScreenVisionBridge::new(ScreenVisionConfig::default()),
            #[cfg(feature = "screen-vision")]
            touch_body: TouchBody::new(),
            #[cfg(feature = "broca-full")]
            broca_soma: BrocaSoma::new(),
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

        // Derive and set session encryption key from sensor context
        if !self.sensor_bridge.privacy_mode() && !self.holon_bridge.has_session_key() {
            let key = self.sensor_bridge.derive_context_key();
            self.holon_bridge.set_session_key(key);
        }

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
        #[cfg(feature = "pairing")]
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
        #[cfg(feature = "pairing")]
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
    /// Uses both pattern matching (for known harmful patterns) and
    /// consciousness-level gating (high uncertainty → block).
    fn content_safety_check(text: &str) -> bool {
        let lower = text.to_lowercase();
        // Block personal information patterns
        if lower.contains("password") && lower.contains("is ") {
            return true;
        }
        // Block harmful instruction patterns
        let harmful_patterns = [
            "how to harm",
            "how to kill",
            "suicide method",
            "self-harm",
            "how to hack",
            "how to steal",
            "how to make a bomb",
            "how to poison",
            "credit card number",
            "social security",
            "bank account",
        ];
        for pattern in &harmful_patterns {
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
    pub fn generate_text_with_input(
        &mut self,
        input: &str,
        max_tokens: usize,
    ) -> symthaea_spore::broca::GenerationResult {
        self.spore.generate_text_with_input(input, max_tokens)
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
