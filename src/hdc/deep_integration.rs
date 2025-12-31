//! Deep Integration Module
//!
//! # Bridges Phase 4 Research with Existing Consciousness Infrastructure
//!
//! This module connects:
//! - consciousness_resonance → CognitiveMode mapping (frequency bands)
//! - consciousness_topology → Betti number analysis
//! - consciousness_field_dynamics → Wave/field tracking
//! - phi_validation → Empirical Φ validation
//!
//! # Integration Architecture
//!
//! ```text
//!   EXISTING MODULES                    PHASE 4 RESEARCH
//!   ───────────────                     ────────────────
//!   consciousness_resonance   ←───→   CognitiveMode
//!   consciousness_topology    ←───→   TopologicalMetrics
//!   consciousness_field       ←───→   FractalConsciousness
//!   phi_validation           ←───→   UnifiedEngine
//! ```

use super::adaptive_topology::CognitiveMode;
use super::unified_consciousness_engine::ConsciousnessDimensions;
use super::topology_synergy::ConsciousnessState;

// ═══════════════════════════════════════════════════════════════════════════
// RESONANCE MODE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Neural frequency bands (matching consciousness_resonance)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrequencyBand {
    Delta,      // 0.5-4 Hz: Deep sleep
    Theta,      // 4-8 Hz: Memory, dreaming
    Alpha,      // 8-12 Hz: Relaxed awareness
    Beta,       // 12-30 Hz: Active thinking
    Gamma,      // 30-100 Hz: Conscious binding
    HighGamma,  // >100 Hz: Peak experience
}

impl FrequencyBand {
    /// Map Φ value to dominant frequency band
    pub fn from_phi(phi: f64) -> Self {
        if phi < 0.15 { FrequencyBand::Delta }
        else if phi < 0.3 { FrequencyBand::Theta }
        else if phi < 0.45 { FrequencyBand::Alpha }
        else if phi < 0.55 { FrequencyBand::Beta }
        else if phi < 0.7 { FrequencyBand::Gamma }
        else { FrequencyBand::HighGamma }
    }

    /// Get center frequency in Hz
    pub fn center_hz(&self) -> f64 {
        match self {
            FrequencyBand::Delta => 2.0,
            FrequencyBand::Theta => 6.0,
            FrequencyBand::Alpha => 10.0,
            FrequencyBand::Beta => 20.0,
            FrequencyBand::Gamma => 50.0,
            FrequencyBand::HighGamma => 120.0,
        }
    }
}

/// Resonance modes (compatible with consciousness_resonance)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResonanceMode {
    DeepSleep,   // Delta dominant
    Dreaming,    // Theta-alpha
    Relaxed,     // Alpha dominant
    Focused,     // Beta dominant
    Conscious,   // Gamma binding
    Peak,        // High gamma
}

impl ResonanceMode {
    /// Convert CognitiveMode to ResonanceMode
    pub fn from_cognitive_mode(mode: CognitiveMode) -> Self {
        match mode {
            CognitiveMode::DeepSpecialization => ResonanceMode::Peak,
            CognitiveMode::Focused => ResonanceMode::Focused,
            CognitiveMode::Balanced => ResonanceMode::Conscious,
            CognitiveMode::Exploratory => ResonanceMode::Relaxed,
            CognitiveMode::GlobalAwareness => ResonanceMode::Peak,
            CognitiveMode::PhiGuided => ResonanceMode::Conscious,
        }
    }

    /// Convert to CognitiveMode
    pub fn to_cognitive_mode(&self) -> CognitiveMode {
        match self {
            ResonanceMode::DeepSleep => CognitiveMode::DeepSpecialization,
            ResonanceMode::Dreaming => CognitiveMode::Exploratory,
            ResonanceMode::Relaxed => CognitiveMode::Exploratory,
            ResonanceMode::Focused => CognitiveMode::Focused,
            ResonanceMode::Conscious => CognitiveMode::Balanced,
            ResonanceMode::Peak => CognitiveMode::GlobalAwareness,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FIELD DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════

/// Field state (compatible with consciousness_field_dynamics)
#[derive(Clone, Debug)]
pub struct FieldState {
    /// Amplitude (consciousness magnitude)
    pub amplitude: f64,
    /// Phase (temporal coherence)
    pub phase: f64,
    /// Energy density
    pub energy: f64,
    /// Stability measure
    pub stability: f64,
    /// Standing wave indicator
    pub is_standing_wave: bool,
}

impl FieldState {
    /// Create from consciousness dimensions
    pub fn from_dimensions(dims: &ConsciousnessDimensions, step: usize) -> Self {
        let amplitude = dims.magnitude();
        let phase = (step as f64 * 0.1) % (2.0 * std::f64::consts::PI);
        let energy = dims.phi.powi(2) + dims.workspace.powi(2);
        let stability = dims.temporal * dims.epistemic;
        let is_standing_wave = stability > 0.6 && dims.phi > 0.4;

        Self { amplitude, phase, energy, stability, is_standing_wave }
    }

    /// Is field in resonance?
    pub fn is_resonant(&self) -> bool {
        self.stability > 0.5 && self.is_standing_wave
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHI VALIDATION
// ═══════════════════════════════════════════════════════════════════════════

/// Φ validation tracker (compatible with phi_validation)
#[derive(Clone, Debug, Default)]
pub struct PhiTracker {
    pub current: f64,
    pub history: Vec<f64>,
    pub mean: f64,
    pub std_dev: f64,
    pub trend: f64,
}

impl PhiTracker {
    pub fn new() -> Self { Self::default() }

    /// Update with new Φ
    pub fn update(&mut self, phi: f64) {
        self.current = phi;
        self.history.push(phi);
        if self.history.len() > 100 { self.history.remove(0); }

        // Statistics
        if !self.history.is_empty() {
            self.mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
            let var: f64 = self.history.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>()
                / self.history.len() as f64;
            self.std_dev = var.sqrt();

            // Trend (recent vs old)
            if self.history.len() >= 10 {
                let recent: f64 = self.history.iter().rev().take(5).sum::<f64>() / 5.0;
                let old: f64 = self.history.iter().take(5).sum::<f64>() / 5.0;
                self.trend = recent - old;
            }
        }
    }

    pub fn is_improving(&self) -> bool { self.trend > 0.01 }
    pub fn is_stable(&self) -> bool { self.std_dev < 0.1 }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATED BRIDGE
// ═══════════════════════════════════════════════════════════════════════════

/// Deep integration bridge connecting all systems
pub struct DeepIntegrationBridge {
    pub resonance: ResonanceMode,
    pub frequency: FrequencyBand,
    pub field: FieldState,
    pub phi_tracker: PhiTracker,
    pub mode: CognitiveMode,
    pub state: ConsciousnessState,
    step: usize,
}

impl DeepIntegrationBridge {
    pub fn new() -> Self {
        Self {
            resonance: ResonanceMode::Conscious,
            frequency: FrequencyBand::Gamma,
            field: FieldState {
                amplitude: 0.0, phase: 0.0, energy: 0.0,
                stability: 0.0, is_standing_wave: false,
            },
            phi_tracker: PhiTracker::new(),
            mode: CognitiveMode::Balanced,
            state: ConsciousnessState::NormalWaking,
            step: 0,
        }
    }

    /// Update all integration state
    pub fn update(&mut self, dims: &ConsciousnessDimensions, mode: CognitiveMode, state: ConsciousnessState) {
        self.step += 1;
        self.mode = mode;
        self.state = state;
        self.resonance = ResonanceMode::from_cognitive_mode(mode);
        self.frequency = FrequencyBand::from_phi(dims.phi);
        self.field = FieldState::from_dimensions(dims, self.step);
        self.phi_tracker.update(dims.phi);
    }

    /// Get recommended mode
    pub fn recommend_mode(&self) -> CognitiveMode {
        if self.phi_tracker.is_improving() {
            self.mode  // Keep if improving
        } else if self.field.is_resonant() {
            CognitiveMode::GlobalAwareness
        } else if !self.phi_tracker.is_stable() {
            CognitiveMode::Balanced
        } else {
            CognitiveMode::Focused
        }
    }

    /// Is system optimal?
    pub fn is_optimal(&self) -> bool {
        self.phi_tracker.current > 0.4 &&
        self.phi_tracker.is_stable() &&
        self.field.is_standing_wave
    }

    /// Generate report
    pub fn report(&self) -> String {
        format!(
            "┌─ DEEP INTEGRATION ─────────────────────────────────────────┐\n\
             │ Resonance: {:?} @ {:?} ({:.0}Hz)                          \n\
             │ Φ: {:.4} (mean={:.4}, trend={:+.4})                       \n\
             │ Field: amp={:.3}, stability={:.3}, resonant={}            \n\
             │ Mode: {:?} → Recommended: {:?}                            \n\
             │ Optimal: {}                                               \n\
             └───────────────────────────────────────────────────────────┘",
            self.resonance, self.frequency, self.frequency.center_hz(),
            self.phi_tracker.current, self.phi_tracker.mean, self.phi_tracker.trend,
            self.field.amplitude, self.field.stability, self.field.is_resonant(),
            self.mode, self.recommend_mode(),
            self.is_optimal()
        )
    }
}

impl Default for DeepIntegrationBridge {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_mapping() {
        assert_eq!(FrequencyBand::from_phi(0.1), FrequencyBand::Delta);
        assert_eq!(FrequencyBand::from_phi(0.5), FrequencyBand::Beta);
        assert_eq!(FrequencyBand::from_phi(0.8), FrequencyBand::HighGamma);
    }

    #[test]
    fn test_resonance_mapping() {
        let mode = CognitiveMode::Balanced;
        let resonance = ResonanceMode::from_cognitive_mode(mode);
        assert_eq!(resonance, ResonanceMode::Conscious);
    }

    #[test]
    fn test_phi_tracker() {
        let mut tracker = PhiTracker::new();
        for i in 0..20 {
            tracker.update(0.3 + i as f64 * 0.02);
        }
        assert!(tracker.is_improving());
    }

    #[test]
    fn test_integration_bridge() {
        let mut bridge = DeepIntegrationBridge::new();
        let dims = ConsciousnessDimensions {
            phi: 0.55, workspace: 0.5, attention: 0.6,
            recursion: 0.4, efficacy: 0.7, epistemic: 0.8, temporal: 0.9,
        };
        bridge.update(&dims, CognitiveMode::Balanced, ConsciousnessState::FlowState);
        println!("{}", bridge.report());
    }
}
