// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Pulse Dynamics: Trigger Timing and Self-Healing Cycles
//!
//! Optimizes the temporal operation of the Spark Engine:
//! - Pulse frequency and duration
//! - Damage accumulation vs healing rate
//! - Thermal cycling management
//! - Steady-state vs burst operation
//!
//! ## Operating Modes
//!
//! ```text
//! CONTINUOUS                    PULSED
//! ─────────────────            ┌──┐  ┌──┐  ┌──┐
//! ████████████████             │  │  │  │  │  │
//! ─────────────────            └──┘  └──┘  └──┘
//! • Higher average power        • Time for healing
//! • More damage accumulation    • Lower thermal stress
//! • Simpler control             • Complex control
//! ```
//!
//! ## Key Trade-offs
//!
//! - **Faster pulses**: Higher peak power, more thermal shock
//! - **Slower pulses**: More healing time, lower average power
//! - **Duty cycle**: On-time / period, affects both power and healing

use super::radiation_damage::FusionReaction;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

/// Operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    /// Continuous operation
    Continuous,
    /// Pulsed operation with healing periods
    Pulsed,
    /// Burst mode for peak power
    Burst,
    /// Variable duty cycle
    Variable,
}

/// Pulse characteristics
#[derive(Debug, Clone)]
pub struct PulseProfile {
    /// Operating mode
    pub mode: OperatingMode,
    /// Pulse duration (s)
    pub pulse_duration_s: f64,
    /// Period between pulses (s)
    pub period_s: f64,
    /// Peak power during pulse (kW)
    pub peak_power_kw: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Average power (kW)
    pub average_power_kw: f64,
}

impl PulseProfile {
    /// Create continuous operation profile
    pub fn continuous(power_kw: f64) -> Self {
        Self {
            mode: OperatingMode::Continuous,
            pulse_duration_s: f64::INFINITY,
            period_s: f64::INFINITY,
            peak_power_kw: power_kw,
            duty_cycle: 1.0,
            average_power_kw: power_kw,
        }
    }

    /// Create pulsed operation profile
    pub fn pulsed(peak_power_kw: f64, pulse_s: f64, period_s: f64) -> Self {
        let duty = pulse_s / period_s;
        Self {
            mode: OperatingMode::Pulsed,
            pulse_duration_s: pulse_s,
            period_s,
            peak_power_kw,
            duty_cycle: duty,
            average_power_kw: peak_power_kw * duty,
        }
    }

    /// Create burst mode profile
    pub fn burst(peak_power_kw: f64, burst_s: f64, cooldown_s: f64) -> Self {
        let period = burst_s + cooldown_s;
        let duty = burst_s / period;
        Self {
            mode: OperatingMode::Burst,
            pulse_duration_s: burst_s,
            period_s: period,
            peak_power_kw,
            duty_cycle: duty,
            average_power_kw: peak_power_kw * duty,
        }
    }

    /// Pulses per hour
    pub fn pulses_per_hour(&self) -> f64 {
        if self.period_s.is_infinite() {
            0.0
        } else {
            3600.0 / self.period_s
        }
    }

    /// Energy per pulse (kJ)
    pub fn energy_per_pulse_kj(&self) -> f64 {
        self.peak_power_kw * self.pulse_duration_s
    }
}

/// Damage accumulation model
#[derive(Debug, Clone)]
pub struct DamageModel {
    /// Damage rate during operation (DPA/s)
    pub damage_rate: f64,
    /// Healing rate during rest (DPA/s healed)
    pub healing_rate: f64,
    /// Critical damage threshold (DPA)
    pub critical_dpa: f64,
    /// Current accumulated damage (DPA)
    pub current_dpa: f64,
}

impl DamageModel {
    /// Create damage model for given material and conditions
    pub fn new(
        neutron_flux: f64,          // n/cm²/s
        material_healing_rate: f64, // DPA/s healed at rest
        critical_dpa: f64,
    ) -> Self {
        // Simplified DPA rate: ~1e-21 DPA per n/cm²
        let damage_rate = neutron_flux * 1e-21;

        Self {
            damage_rate,
            healing_rate: material_healing_rate,
            critical_dpa,
            current_dpa: 0.0,
        }
    }

    /// Simulate one cycle and return new damage level
    pub fn simulate_cycle(&mut self, pulse: &PulseProfile) -> f64 {
        // Damage during pulse
        let damage_added = self.damage_rate * pulse.pulse_duration_s;

        // Healing during rest period
        let rest_time = pulse.period_s - pulse.pulse_duration_s;
        let healing = self.healing_rate * rest_time;

        // Net change
        let net_change = damage_added - healing;
        self.current_dpa = (self.current_dpa + net_change).max(0.0);

        self.current_dpa
    }

    /// Find equilibrium DPA level
    pub fn equilibrium_dpa(&self, pulse: &PulseProfile) -> f64 {
        if pulse.mode == OperatingMode::Continuous {
            // No healing, damage accumulates forever
            return f64::INFINITY;
        }

        // At equilibrium: damage_rate * on_time = healing_rate * off_time
        let damage_per_cycle = self.damage_rate * pulse.pulse_duration_s;
        let rest_time = pulse.period_s - pulse.pulse_duration_s;
        let healing_per_cycle = self.healing_rate * rest_time;

        if healing_per_cycle >= damage_per_cycle {
            // Net negative damage - equilibrium at 0
            0.0
        } else {
            // Damage accumulates but slows as DPA rises
            // (Simplified - real model would have DPA-dependent healing)
            damage_per_cycle / (healing_per_cycle / pulse.period_s + 1e-20)
        }
    }

    /// Time to critical failure (s)
    pub fn time_to_failure(&self, pulse: &PulseProfile) -> f64 {
        let eq_dpa = self.equilibrium_dpa(pulse);

        if eq_dpa < self.critical_dpa {
            // Never reaches critical - infinite lifetime
            f64::INFINITY
        } else if pulse.mode == OperatingMode::Continuous {
            self.critical_dpa / self.damage_rate
        } else {
            // Time to reach critical with pulsed operation
            let net_rate =
                self.damage_rate * pulse.duty_cycle - self.healing_rate * (1.0 - pulse.duty_cycle);
            if net_rate <= 0.0 {
                f64::INFINITY
            } else {
                self.critical_dpa / net_rate
            }
        }
    }

    /// Lifetime in years
    pub fn lifetime_years(&self, pulse: &PulseProfile) -> f64 {
        let ttf = self.time_to_failure(pulse);
        if ttf.is_infinite() {
            f64::INFINITY
        } else {
            ttf / (365.25 * 24.0 * 3600.0)
        }
    }
}

/// Thermal cycling analysis
#[derive(Debug, Clone)]
pub struct ThermalCycling {
    /// Temperature swing during cycle (K)
    pub delta_t: f64,
    /// Thermal time constant (s)
    pub time_constant: f64,
    /// Cycles to fatigue failure
    pub cycles_to_failure: f64,
    /// Fatigue safety factor
    pub fatigue_safety: f64,
}

impl ThermalCycling {
    /// Analyze thermal cycling for given pulse profile
    pub fn analyze(
        pulse: &PulseProfile,
        t_hot: f64,         // Max temperature (K)
        t_cold: f64,        // Min temperature (K)
        thermal_tau: f64,   // Thermal time constant (s)
        fatigue_coeff: f64, // Material fatigue coefficient
    ) -> Self {
        let delta_t = t_hot - t_cold;

        // Actual temperature swing depends on whether thermal equilibrium reached
        let actual_swing = if pulse.pulse_duration_s > 3.0 * thermal_tau {
            delta_t // Full swing
        } else {
            delta_t * (1.0 - (-pulse.pulse_duration_s / thermal_tau).exp())
        };

        // Coffin-Manson fatigue: N = C * (ΔT)^(-n), n ≈ 2
        let cycles_to_failure = fatigue_coeff / actual_swing.powi(2);

        // Safety factor (want > 10)
        let fatigue_safety = cycles_to_failure / (pulse.pulses_per_hour() * 8760.0 * 20.0);

        Self {
            delta_t: actual_swing,
            time_constant: thermal_tau,
            cycles_to_failure,
            fatigue_safety,
        }
    }
}

/// Pulse optimization result
#[derive(Debug, Clone)]
pub struct PulseOptimization {
    /// Optimal pulse profile
    pub profile: PulseProfile,
    /// Damage model results
    pub damage: DamageResult,
    /// Thermal cycling results
    pub thermal: ThermalCycling,
    /// Overall score (0-1)
    pub score: f64,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Damage model results
#[derive(Debug, Clone)]
pub struct DamageResult {
    /// Equilibrium DPA
    pub equilibrium_dpa: f64,
    /// Lifetime in years
    pub lifetime_years: f64,
    /// Is sustainable (equilibrium < critical)?
    pub sustainable: bool,
}

/// Pulse dynamics optimizer
#[derive(Debug, Clone)]
pub struct PulseDynamics {
    /// Concept vectors
    pub healing: ContinuousHV,
    pub damage: ContinuousHV,
    pub cycling: ContinuousHV,
    pub stability: ContinuousHV,
}

impl PulseDynamics {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            healing: genesis.hv("pulse::healing", PHYSICS_DIM),
            damage: genesis.hv("pulse::damage", PHYSICS_DIM),
            cycling: genesis.hv("pulse::cycling", PHYSICS_DIM),
            stability: genesis.hv("pulse::stability", PHYSICS_DIM),
        }
    }

    /// Optimize pulse profile for given target
    pub fn optimize(
        &self,
        target_power_kw: f64,
        reaction: FusionReaction,
        healing_rate: f64, // DPA/s healing
        critical_dpa: f64,
        thermal_tau: f64, // Thermal time constant (s)
    ) -> PulseOptimization {
        // Estimate neutron flux from power
        let energy_per_fusion = reaction.total_energy_mev() * super::constants::MEV_TO_J;
        let fusions_per_s = target_power_kw * 1000.0 / energy_per_fusion;
        let neutron_flux = fusions_per_s / 1e4; // Approximate n/cm²/s at shell

        let mut best_score = 0.0;
        let mut best_result: Option<PulseOptimization> = None;

        // Try different pulse configurations
        let configs = [
            // (pulse_s, period_s)
            (0.001, 0.01), // 10% duty, fast
            (0.01, 0.1),   // 10% duty, medium
            (0.1, 1.0),    // 10% duty, slow
            (0.1, 0.5),    // 20% duty
            (0.1, 0.2),    // 50% duty
            (1.0, 2.0),    // 50% duty, slow
            (1.0, 10.0),   // 10% duty, very slow
        ];

        for (pulse_s, period_s) in configs {
            // Scale peak power to achieve target average
            let duty = pulse_s / period_s;
            let peak_power = target_power_kw / duty;

            let profile = PulseProfile::pulsed(peak_power, pulse_s, period_s);

            // Analyze damage
            let damage_model = DamageModel::new(neutron_flux, healing_rate, critical_dpa);
            let eq_dpa = damage_model.equilibrium_dpa(&profile);
            let lifetime = damage_model.lifetime_years(&profile);
            let sustainable = eq_dpa < critical_dpa;

            let damage_result = DamageResult {
                equilibrium_dpa: eq_dpa,
                lifetime_years: lifetime,
                sustainable,
            };

            // Analyze thermal cycling
            let thermal = ThermalCycling::analyze(
                &profile,
                600.0, // T_hot estimate
                350.0, // T_cold estimate
                thermal_tau,
                1e8, // Fatigue coefficient
            );

            // Score this configuration
            let lifetime_score = if lifetime.is_infinite() {
                1.0
            } else {
                (lifetime / 50.0).min(1.0) // 50 years = perfect
            };

            let fatigue_score = (thermal.fatigue_safety / 10.0).min(1.0);
            let power_score = (duty).min(1.0); // Higher duty = higher power density

            let score = lifetime_score * 0.4 + fatigue_score * 0.3 + power_score * 0.3;

            if score > best_score {
                best_score = score;

                let mut recommendations = Vec::new();
                if sustainable {
                    recommendations
                        .push("Sustainable operation - indefinite lifetime possible".to_string());
                } else {
                    recommendations.push(format!(
                        "Finite lifetime: {lifetime:.1} years - schedule maintenance"
                    ));
                }
                if thermal.fatigue_safety < 5.0 {
                    recommendations
                        .push("Warning: Low fatigue margin - consider slower cycling".to_string());
                }
                if duty < 0.2 {
                    recommendations.push(
                        "Low duty cycle - consider higher healing rate materials".to_string(),
                    );
                }

                best_result = Some(PulseOptimization {
                    profile,
                    damage: damage_result,
                    thermal,
                    score,
                    recommendations,
                });
            }
        }

        best_result.expect("pulse optimizer: no valid configuration in profile set")
    }

    /// Analyze continuous operation
    pub fn analyze_continuous(
        &self,
        power_kw: f64,
        neutron_flux: f64,
        healing_rate: f64,
        critical_dpa: f64,
    ) -> DamageResult {
        let profile = PulseProfile::continuous(power_kw);
        let damage_model = DamageModel::new(neutron_flux, healing_rate, critical_dpa);

        DamageResult {
            equilibrium_dpa: f64::INFINITY,
            lifetime_years: damage_model.lifetime_years(&profile),
            sustainable: false,
        }
    }

    /// Calculate minimum rest time for sustainable operation
    pub fn min_rest_time(&self, pulse_duration_s: f64, damage_rate: f64, healing_rate: f64) -> f64 {
        // For sustainability: damage * pulse_time <= healing * rest_time
        // rest_time >= damage * pulse_time / healing
        damage_rate * pulse_duration_s / healing_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> PulseDynamics {
        let genesis = GenesisSeed::from_phrase("pulse test");
        PulseDynamics::from_genesis(&genesis)
    }

    #[test]
    fn test_pulse_profiles() {
        let continuous = PulseProfile::continuous(100.0);
        assert_eq!(continuous.duty_cycle, 1.0);
        assert_eq!(continuous.average_power_kw, 100.0);

        let pulsed = PulseProfile::pulsed(100.0, 0.1, 1.0);
        assert!((pulsed.duty_cycle - 0.1).abs() < 0.01);
        assert!((pulsed.average_power_kw - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_damage_model() {
        let model = DamageModel::new(1e14, 1e-8, 100.0);

        // High healing rate should result in low equilibrium
        let profile = PulseProfile::pulsed(100.0, 0.1, 10.0); // 1% duty
        let eq = model.equilibrium_dpa(&profile);

        assert!(eq < model.critical_dpa);
    }

    #[test]
    fn test_continuous_damage() {
        let model = DamageModel::new(1e14, 1e-8, 100.0);
        let profile = PulseProfile::continuous(100.0);

        let eq = model.equilibrium_dpa(&profile);
        assert!(eq.is_infinite());

        let lifetime = model.lifetime_years(&profile);
        assert!(lifetime.is_finite());
        assert!(lifetime > 0.0);
    }

    #[test]
    fn test_thermal_cycling() {
        let profile = PulseProfile::pulsed(100.0, 0.1, 1.0);
        let cycling = ThermalCycling::analyze(&profile, 600.0, 350.0, 0.5, 1e8);

        assert!(cycling.delta_t > 0.0);
        assert!(cycling.cycles_to_failure > 0.0);
    }

    #[test]
    fn test_optimization() {
        let dynamics = setup();

        let result = dynamics.optimize(
            5.0, // 5 kW target
            FusionReaction::DD,
            1e-7,  // Healing rate
            100.0, // Critical DPA
            0.5,   // Thermal time constant
        );

        assert!(result.score > 0.0);
        assert!(result.profile.average_power_kw > 0.0);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_min_rest_time() {
        let dynamics = setup();

        let rest = dynamics.min_rest_time(0.1, 1e-5, 1e-6);
        assert!(rest > 0.0);
        assert!(rest > 0.1); // Need more rest than pulse time for this ratio
    }
}
