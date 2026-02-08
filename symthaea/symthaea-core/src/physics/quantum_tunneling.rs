//! # Quantum Tunneling
//!
//! This module implements quantum mechanical tunneling calculations including
//! WKB transmission coefficients, Gamow factors for nuclear physics, and
//! HDC representations of tunneling states.
//!
//! ## Key Concepts
//!
//! - **WKB Approximation**: Semi-classical method for transmission through barriers
//! - **Gamow Factor**: Coulomb barrier penetration in nuclear reactions
//! - **Alpha Decay**: Gamow-Gurney-Condon theory for radioactive decay
//!
//! ## Physical Background
//!
//! Unlike classical mechanics, quantum particles can penetrate potential barriers
//! even when their kinetic energy is less than the barrier height. The transmission
//! probability T depends exponentially on the barrier parameters:
//!
//! T ≈ exp(-2γL) where γ = √(2m(V-E))/ℏ
//!
//! ## References
//!
//! - Gamow (1928) - "Zur Quantentheorie des Atomkernes"
//! - Gurney & Condon (1929) - "Quantum Mechanics and Radioactive Disintegration"

use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use super::standard_model::PHYSICS_DIM;
use super::constants::{HBAR, E_CHARGE, M_ELECTRON, M_PROTON, C};
use serde::{Deserialize, Serialize};

/// Fine structure constant (dimensionless)
const ALPHA: f64 = 7.297_352_5693e-3;

/// Coulomb constant k_e = 1/(4πε₀)
const K_COULOMB: f64 = 8.987_551_7923e9;

/// Atomic mass unit in kg
const AMU: f64 = 1.660_539_066_60e-27;

/// Result of a tunneling calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelingResult {
    /// Transmission coefficient T (probability)
    pub transmission: f64,
    /// Reflection coefficient R = 1 - T
    pub reflection: f64,
    /// Characteristic decay constant gamma
    pub gamma: f64,
    /// WKB action integral (dimensionless)
    pub action: f64,
}

/// WKB transmission coefficient calculator
///
/// Implements the Wentzel-Kramers-Brillouin approximation for
/// quantum mechanical barrier penetration.
#[derive(Debug, Clone)]
pub struct TunnelingCalculator {
    /// Reduced Planck constant (J·s)
    hbar: f64,
    /// Particle mass (kg)
    mass: f64,
}

impl Default for TunnelingCalculator {
    fn default() -> Self {
        Self::electron()
    }
}

impl TunnelingCalculator {
    /// Create calculator for electrons
    pub fn electron() -> Self {
        Self {
            hbar: HBAR,
            mass: M_ELECTRON,
        }
    }

    /// Create calculator for protons
    pub fn proton() -> Self {
        Self {
            hbar: HBAR,
            mass: M_PROTON,
        }
    }

    /// Create calculator with custom particle mass
    pub fn with_mass(mass_kg: f64) -> Self {
        Self {
            hbar: HBAR,
            mass: mass_kg,
        }
    }

    /// Create calculator for alpha particles (He-4 nucleus)
    pub fn alpha_particle() -> Self {
        Self {
            hbar: HBAR,
            mass: 4.0 * AMU,
        }
    }

    /// Transmission coefficient for rectangular barrier
    ///
    /// For a barrier of height V and width L with particle energy E < V:
    /// T = exp(-2γL) where γ = √(2m(V-E))/ℏ
    ///
    /// # Arguments
    /// * `energy` - Particle kinetic energy (J)
    /// * `barrier_height` - Barrier potential (J)
    /// * `width` - Barrier width (m)
    ///
    /// # Returns
    /// TunnelingResult with transmission probability
    pub fn rectangular_barrier(&self, energy: f64, barrier_height: f64, width: f64) -> TunnelingResult {
        if energy >= barrier_height {
            // Classically allowed - full transmission (with some reflection)
            // For E > V, use classical transmission coefficient
            let kappa = (2.0 * self.mass * (energy - barrier_height)).sqrt() / self.hbar;
            let k = (2.0 * self.mass * energy).sqrt() / self.hbar;
            let ratio = (k - kappa) / (k + kappa);
            let t = 1.0 - ratio * ratio;
            return TunnelingResult {
                transmission: t.min(1.0),
                reflection: 1.0 - t.min(1.0),
                gamma: 0.0,
                action: 0.0,
            };
        }

        // Tunneling regime: E < V
        let gamma = (2.0 * self.mass * (barrier_height - energy)).sqrt() / self.hbar;
        let action = 2.0 * gamma * width;

        // WKB transmission coefficient
        let transmission = (-action).exp();

        TunnelingResult {
            transmission,
            reflection: 1.0 - transmission,
            gamma,
            action,
        }
    }

    /// Gamow factor for Coulomb barrier penetration
    ///
    /// The Gamow factor describes tunneling through the Coulomb barrier
    /// in nuclear reactions:
    ///
    /// G = 2πη where η = Z₁Z₂e²/(4πε₀ℏv) is the Sommerfeld parameter
    ///
    /// # Arguments
    /// * `z1` - Atomic number of projectile
    /// * `z2` - Atomic number of target
    /// * `energy` - Center-of-mass kinetic energy (J)
    ///
    /// # Returns
    /// Gamow factor (dimensionless)
    pub fn gamow_factor(&self, z1: u8, z2: u8, energy: f64) -> f64 {
        if energy <= 0.0 {
            return f64::INFINITY;
        }

        // Velocity from kinetic energy: v = √(2E/m)
        let velocity = (2.0 * energy / self.mass).sqrt();

        // Sommerfeld parameter η = Z₁Z₂α·c/v
        let eta = (z1 as f64) * (z2 as f64) * ALPHA * C / velocity;

        // Gamow factor G = 2πη
        2.0 * std::f64::consts::PI * eta
    }

    /// Gamow penetration probability
    ///
    /// P = exp(-G) where G is the Gamow factor
    pub fn gamow_probability(&self, z1: u8, z2: u8, energy: f64) -> f64 {
        let g = self.gamow_factor(z1, z2, energy);
        (-g).exp()
    }

    /// Alpha decay rate from Gamow-Gurney-Condon theory
    ///
    /// The decay rate λ for alpha emission:
    /// λ = f × P
    ///
    /// where f is the attempt frequency (oscillation frequency inside nucleus)
    /// and P is the barrier penetration probability.
    ///
    /// # Arguments
    /// * `q_value` - Q-value of decay (energy released, J)
    /// * `z_daughter` - Atomic number of daughter nucleus
    /// * `radius` - Nuclear radius (m)
    ///
    /// # Returns
    /// Decay rate λ (s⁻¹)
    pub fn alpha_decay_rate(&self, q_value: f64, z_daughter: u8, radius: f64) -> f64 {
        if q_value <= 0.0 || radius <= 0.0 {
            return 0.0;
        }

        // Alpha particle (Z=2)
        let z_alpha = 2u8;

        // Coulomb barrier height at nuclear surface
        let _v_coulomb = K_COULOMB * (z_alpha as f64) * (z_daughter as f64)
            * E_CHARGE * E_CHARGE / radius;

        // Classical turning point (where E = V)
        let r_turn = K_COULOMB * (z_alpha as f64) * (z_daughter as f64)
            * E_CHARGE * E_CHARGE / q_value;

        // WKB action integral through Coulomb barrier
        // S = (2/ℏ) ∫ √(2m(V(r)-E)) dr from R to r_turn
        // For Coulomb barrier, this gives approximately:
        let action = (2.0 * (2.0 * self.mass).sqrt() / self.hbar)
            * (K_COULOMB * (z_alpha as f64) * (z_daughter as f64) * E_CHARGE * E_CHARGE)
            .sqrt()
            * (r_turn.sqrt() * (1.0 - (radius / r_turn).sqrt()).acos()
               - (radius * (r_turn - radius)).sqrt());

        // Transmission probability
        let p_tunnel = (-2.0 * action).exp();

        // Attempt frequency: v_alpha / (2R) where v_alpha = √(2E/m)
        let v_alpha = (2.0 * q_value / self.mass).sqrt();
        let frequency = v_alpha / (2.0 * radius);

        // Decay rate
        frequency * p_tunnel
    }

    /// Half-life from decay rate
    ///
    /// t₁/₂ = ln(2) / λ
    pub fn half_life(&self, decay_rate: f64) -> f64 {
        if decay_rate <= 0.0 {
            return f64::INFINITY;
        }
        2.0_f64.ln() / decay_rate
    }

    /// WKB approximation for arbitrary potential V(x)
    ///
    /// Computes the transmission coefficient through a barrier defined by
    /// a potential function V(x) for a particle with energy E.
    ///
    /// # Arguments
    /// * `energy` - Particle energy (J)
    /// * `potential` - Function V(x) returning potential energy at position x
    /// * `x_start` - Start of integration region (m)
    /// * `x_end` - End of integration region (m)
    /// * `n_points` - Number of integration points
    ///
    /// # Returns
    /// TunnelingResult with transmission coefficient
    pub fn wkb_transmission<F>(
        &self,
        energy: f64,
        potential: F,
        x_start: f64,
        x_end: f64,
        n_points: usize,
    ) -> TunnelingResult
    where
        F: Fn(f64) -> f64,
    {
        let n = n_points.max(100);
        let dx = (x_end - x_start) / n as f64;

        // Integrate √(2m(V-E)) over classically forbidden region
        let mut action = 0.0;
        let mut in_barrier = false;
        let mut _barrier_count = 0;

        for i in 0..n {
            let x = x_start + (i as f64 + 0.5) * dx;
            let v = potential(x);

            if v > energy {
                // Inside barrier (classically forbidden)
                let kappa = (2.0 * self.mass * (v - energy)).sqrt() / self.hbar;
                action += kappa * dx;
                if !in_barrier {
                    _barrier_count += 1;
                }
                in_barrier = true;
            } else {
                in_barrier = false;
            }
        }

        // Transmission coefficient
        let transmission = if action > 0.0 {
            (-2.0 * action).exp()
        } else {
            1.0 // No barrier
        };

        let gamma = if x_end > x_start {
            action / (x_end - x_start)
        } else {
            0.0
        };

        TunnelingResult {
            transmission,
            reflection: 1.0 - transmission,
            gamma,
            action: 2.0 * action,
        }
    }

    /// Eckart barrier transmission coefficient
    ///
    /// The Eckart potential V(x) = V₀ sech²(x/a) is commonly used
    /// for chemical reaction barriers.
    ///
    /// # Arguments
    /// * `energy` - Particle energy (J)
    /// * `barrier_height` - Maximum barrier height V₀ (J)
    /// * `barrier_width` - Characteristic width a (m)
    pub fn eckart_barrier(&self, energy: f64, barrier_height: f64, barrier_width: f64) -> TunnelingResult {
        // Use WKB with Eckart potential
        let potential = |x: f64| -> f64 {
            let arg = x / barrier_width;
            barrier_height / arg.cosh().powi(2)
        };

        // Integrate over ±5 barrier widths
        self.wkb_transmission(
            energy,
            potential,
            -5.0 * barrier_width,
            5.0 * barrier_width,
            200,
        )
    }

    /// Parabolic barrier transmission coefficient
    ///
    /// For an inverted parabola V(x) = V₀ - ½mω²x², the exact
    /// transmission coefficient can be calculated analytically.
    ///
    /// # Arguments
    /// * `energy` - Particle energy (J)
    /// * `barrier_height` - Barrier height at x=0 (J)
    /// * `curvature_omega` - Curvature parameter ω (rad/s)
    pub fn parabolic_barrier(&self, energy: f64, barrier_height: f64, curvature_omega: f64) -> TunnelingResult {
        // For parabolic barrier: T = 1 / (1 + exp(-2π(E-V₀)/(ℏω)))
        let exponent = -2.0 * std::f64::consts::PI * (energy - barrier_height)
            / (self.hbar * curvature_omega);

        let transmission = 1.0 / (1.0 + exponent.exp());

        TunnelingResult {
            transmission,
            reflection: 1.0 - transmission,
            gamma: curvature_omega * self.mass / self.hbar,
            action: exponent.abs(),
        }
    }
}

/// Tunneling barrier types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierType {
    /// Rectangular barrier (step potential)
    Rectangular,
    /// Coulomb barrier (nuclear physics)
    Coulomb,
    /// Eckart barrier (chemical reactions)
    Eckart,
    /// Parabolic/inverted harmonic barrier
    Parabolic,
    /// Double-well potential
    DoubleWell,
}

/// HDC representation of tunneling states
#[derive(Debug, Clone)]
pub struct TunnelingEncoder {
    genesis: GenesisSeed,
    /// Barrier concept vector
    pub barrier: ContinuousHV,
    /// Transmission concept vector
    pub transmission: ContinuousHV,
    /// Reflection concept vector
    pub reflection: ContinuousHV,
    /// Incident wave vector
    pub incident: ContinuousHV,
    /// Transmitted wave vector
    pub transmitted: ContinuousHV,
    /// Evanescent decay vector
    pub evanescent: ContinuousHV,
    /// Quantum tunneling concept
    pub tunneling: ContinuousHV,
}

impl TunnelingEncoder {
    /// Create a new tunneling encoder from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),
            barrier: genesis.hv("tunneling::barrier", PHYSICS_DIM),
            transmission: genesis.hv("tunneling::transmission", PHYSICS_DIM),
            reflection: genesis.hv("tunneling::reflection", PHYSICS_DIM),
            incident: genesis.hv("tunneling::incident", PHYSICS_DIM),
            transmitted: genesis.hv("tunneling::transmitted", PHYSICS_DIM),
            evanescent: genesis.hv("tunneling::evanescent", PHYSICS_DIM),
            tunneling: genesis.hv("tunneling::concept", PHYSICS_DIM),
        }
    }

    /// Encode a tunneling probability into HDC space
    ///
    /// Creates a superposition weighted by transmission/reflection probabilities
    pub fn encode_tunneling_result(&self, result: &TunnelingResult) -> ContinuousHV {
        // Weight transmission and reflection components
        ContinuousHV::weighted_bundle(
            &[&self.transmission, &self.reflection],
            &[result.transmission as f32, result.reflection as f32],
        )
    }

    /// Encode barrier type
    pub fn encode_barrier_type(&self, barrier_type: BarrierType) -> ContinuousHV {
        let type_label = match barrier_type {
            BarrierType::Rectangular => "barrier::rectangular",
            BarrierType::Coulomb => "barrier::coulomb",
            BarrierType::Eckart => "barrier::eckart",
            BarrierType::Parabolic => "barrier::parabolic",
            BarrierType::DoubleWell => "barrier::double_well",
        };
        let type_hv = self.genesis.hv(type_label, PHYSICS_DIM);
        self.barrier.bind(&type_hv)
    }

    /// Encode tunneling state with barrier and result
    pub fn encode_full_state(
        &self,
        barrier_type: BarrierType,
        result: &TunnelingResult,
    ) -> ContinuousHV {
        let barrier_hv = self.encode_barrier_type(barrier_type);
        let result_hv = self.encode_tunneling_result(result);

        // Combine using binding
        self.tunneling.bind(&barrier_hv).bind(&result_hv)
    }
}

/// Tunneling time estimation
///
/// The traversal time for a particle tunneling through a barrier
/// is a subtle concept with multiple definitions.
#[derive(Debug, Clone)]
pub struct TunnelingTime;

impl TunnelingTime {
    /// Phase time (Wigner-Eisenbud time)
    ///
    /// The phase time is derived from the energy derivative of the phase:
    /// τ_φ = ℏ dφ/dE
    ///
    /// # Arguments
    /// * `calc` - Tunneling calculator
    /// * `energy` - Particle energy (J)
    /// * `barrier_height` - Barrier potential (J)
    /// * `width` - Barrier width (m)
    pub fn phase_time(
        calc: &TunnelingCalculator,
        energy: f64,
        barrier_height: f64,
        width: f64,
    ) -> f64 {
        if energy >= barrier_height {
            // No tunneling - classical traversal time
            let velocity = (2.0 * energy / calc.mass).sqrt();
            return width / velocity;
        }

        // For rectangular barrier:
        // τ_φ ≈ m L / (ℏ κ) where κ = √(2m(V-E))/ℏ
        let kappa = (2.0 * calc.mass * (barrier_height - energy)).sqrt() / calc.hbar;
        calc.mass * width / (calc.hbar * kappa)
    }

    /// Büttiker-Landauer time
    ///
    /// The time spent inside the barrier region, weighted by probability density.
    pub fn buttiker_landauer_time(
        calc: &TunnelingCalculator,
        energy: f64,
        barrier_height: f64,
        width: f64,
    ) -> f64 {
        if energy >= barrier_height {
            let velocity = (2.0 * energy / calc.mass).sqrt();
            return width / velocity;
        }

        // For rectangular barrier:
        // τ_BL = m / (ℏ κ) × [1 + (κL/2) × coth(κL)]
        let kappa = (2.0 * calc.mass * (barrier_height - energy)).sqrt() / calc.hbar;
        let kl = kappa * width;

        let coth = if kl > 20.0 {
            1.0 // Avoid overflow
        } else {
            (kl.exp() + (-kl).exp()) / (kl.exp() - (-kl).exp())
        };

        calc.mass / (calc.hbar * kappa) * (1.0 + kl * coth / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_barrier_classical() {
        let calc = TunnelingCalculator::electron();
        // Energy above barrier - mostly transmission
        let result = calc.rectangular_barrier(2.0e-19, 1.0e-19, 1.0e-9);
        assert!(result.transmission > 0.5, "Should mostly transmit when E > V");
    }

    #[test]
    fn test_rectangular_barrier_tunneling() {
        let calc = TunnelingCalculator::electron();
        // Energy below barrier - tunneling
        let result = calc.rectangular_barrier(1.0e-19, 2.0e-19, 1.0e-10);
        assert!(result.transmission > 0.0 && result.transmission < 1.0,
            "Should tunnel with 0 < T < 1");
        assert!((result.transmission + result.reflection - 1.0).abs() < 1e-10,
            "T + R should equal 1");
    }

    #[test]
    fn test_transmission_decreases_with_width() {
        let calc = TunnelingCalculator::electron();
        let t1 = calc.rectangular_barrier(1.0e-19, 2.0e-19, 1.0e-10);
        let t2 = calc.rectangular_barrier(1.0e-19, 2.0e-19, 2.0e-10);
        assert!(t2.transmission < t1.transmission,
            "Wider barrier should have lower transmission");
    }

    #[test]
    fn test_gamow_factor_positive() {
        let calc = TunnelingCalculator::proton();
        let g = calc.gamow_factor(1, 1, 1.0e-13); // 1 MeV proton
        assert!(g > 0.0, "Gamow factor should be positive");
    }

    #[test]
    fn test_gamow_probability() {
        let calc = TunnelingCalculator::proton();
        let p = calc.gamow_probability(1, 1, 1.0e-13);
        assert!(p >= 0.0 && p <= 1.0, "Probability should be in [0, 1]");
    }

    #[test]
    fn test_alpha_decay_rate() {
        let calc = TunnelingCalculator::alpha_particle();
        // Typical alpha decay parameters
        let q_value = 5.0 * 1.6e-13; // 5 MeV
        let z_daughter = 82; // Lead
        let radius = 7.0e-15; // ~7 fm

        let rate = calc.alpha_decay_rate(q_value, z_daughter, radius);
        assert!(rate > 0.0, "Decay rate should be positive");

        let half_life = calc.half_life(rate);
        assert!(half_life > 0.0, "Half-life should be positive");
    }

    #[test]
    fn test_wkb_rectangular() {
        let calc = TunnelingCalculator::electron();
        let v0 = 2.0e-19;
        let energy = 1.0e-19;
        let width = 1.0e-10;

        // Define rectangular potential
        let potential = |x: f64| -> f64 {
            if x > 0.0 && x < width { v0 } else { 0.0 }
        };

        let result = calc.wkb_transmission(energy, potential, -1.0e-10, 2.0e-10, 300);
        assert!(result.transmission > 0.0 && result.transmission < 1.0);
    }

    #[test]
    fn test_eckart_barrier() {
        let calc = TunnelingCalculator::proton();
        let result = calc.eckart_barrier(1.0e-20, 2.0e-20, 1.0e-10);
        assert!(result.transmission >= 0.0 && result.transmission <= 1.0);
    }

    #[test]
    fn test_parabolic_barrier() {
        let calc = TunnelingCalculator::electron();
        let result = calc.parabolic_barrier(1.0e-19, 2.0e-19, 1.0e14);
        assert!(result.transmission >= 0.0 && result.transmission <= 1.0);
    }

    #[test]
    fn test_tunneling_encoder() {
        let genesis = GenesisSeed::from_phrase("test");
        let encoder = TunnelingEncoder::from_genesis(&genesis);

        let result = TunnelingResult {
            transmission: 0.3,
            reflection: 0.7,
            gamma: 1e9,
            action: 2.0,
        };

        let hv = encoder.encode_tunneling_result(&result);
        assert_eq!(hv.values.len(), PHYSICS_DIM);
    }

    #[test]
    fn test_phase_time() {
        let calc = TunnelingCalculator::electron();
        let time = TunnelingTime::phase_time(&calc, 1.0e-19, 2.0e-19, 1.0e-10);
        assert!(time > 0.0, "Phase time should be positive");
    }

    #[test]
    fn test_buttiker_landauer_time() {
        let calc = TunnelingCalculator::electron();
        let time = TunnelingTime::buttiker_landauer_time(&calc, 1.0e-19, 2.0e-19, 1.0e-10);
        assert!(time > 0.0, "BL time should be positive");
    }
}
