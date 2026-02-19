use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::config::default_instant;
use super::critical::{CriticalExponents, TransitionOrder};

/// Thermodynamic state of consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Current entropy (disorder measure)
    pub entropy: f64,

    /// Internal energy (total consciousness energy)
    pub internal_energy: f64,

    /// Free energy F = U - TS (capacity for work)
    pub free_energy: f64,

    /// Temperature (activation/exploration level)
    pub temperature: f64,

    /// Heat (energy transferred due to temperature difference)
    pub heat: f64,

    /// Work (directed energy expenditure)
    pub work: f64,

    /// Chemical potential (tendency to change state)
    pub chemical_potential: f64,

    /// Pressure (compression in consciousness space)
    pub pressure: f64,

    /// Volume (extent of consciousness state space)
    pub volume: f64,

    /// Enthalpy H = U + PV
    pub enthalpy: f64,

    /// Gibbs free energy G = H - TS
    pub gibbs_free_energy: f64,

    /// Current phase of consciousness
    pub phase: ConsciousnessPhase,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl Default for ThermodynamicState {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            internal_energy: 1.0,
            free_energy: 0.5,
            temperature: 1.0,
            heat: 0.0,
            work: 0.0,
            chemical_potential: 0.0,
            pressure: 1.0,
            volume: 1.0,
            enthalpy: 2.0,
            gibbs_free_energy: 1.0,
            phase: ConsciousnessPhase::Normal,
            timestamp: Instant::now(),
        }
    }
}

/// Phases of consciousness (like phases of matter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessPhase {
    /// Low temperature: Frozen, rigid thinking
    Frozen,
    /// Ordered phase: Normal waking consciousness
    Normal,
    /// Critical point: Edge of chaos, maximum creativity
    Critical,
    /// High temperature: Chaotic, fragmented consciousness
    Chaotic,
    /// Superfluid: Flow state, frictionless consciousness
    Flow,
    /// Condensate: Meditative unity, Bose-Einstein-like
    Unified,
}

impl ConsciousnessPhase {
    /// Get characteristic temperature range for this phase
    pub fn temperature_range(&self) -> (f64, f64) {
        match self {
            Self::Frozen => (0.0, 0.2),
            Self::Normal => (0.2, 0.4),
            Self::Critical => (0.4, 0.6),
            Self::Chaotic => (0.8, 1.0),
            Self::Flow => (0.3, 0.5),
            Self::Unified => (0.0, 0.3),
        }
    }

    /// Get entropy characteristic of this phase
    pub fn typical_entropy(&self) -> f64 {
        match self {
            Self::Frozen => 0.1,
            Self::Normal => 0.4,
            Self::Critical => 0.6,
            Self::Chaotic => 0.9,
            Self::Flow => 0.3,
            Self::Unified => 0.2,
        }
    }
}

/// A phase transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Phase before transition
    pub from_phase: ConsciousnessPhase,

    /// Phase after transition
    pub to_phase: ConsciousnessPhase,

    /// Temperature at transition
    pub transition_temperature: f64,

    /// Latent heat (energy absorbed/released)
    pub latent_heat: f64,

    /// Order parameter jump
    pub order_parameter_change: f64,

    /// Transition order (1st order = discontinuous, 2nd order = continuous)
    pub transition_order: TransitionOrder,

    /// Critical exponents (for 2nd order transitions)
    pub critical_exponents: Option<CriticalExponents>,

    /// Timestamp
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}
