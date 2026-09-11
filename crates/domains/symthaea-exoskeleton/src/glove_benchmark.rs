// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repeatable benchmark harness for the SX-018 powered EVA glove.
//!
//! The benchmark reports workload, energy, fatigue, tactility proxy, and a
//! mandatory zero-power backdrive probe under one declared protocol. It does
//! not establish human performance or qualification evidence by itself.

use serde::{Deserialize, Serialize};

use crate::powered_glove::{
    PoweredGloveCommand, PoweredGloveFailState, PoweredGloveFault, PoweredGloveMode,
    PoweredGloveTwin, NUM_GLOVE_DIGITS,
};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GloveBenchmarkProtocol {
    pub protocol_id: String,
    pub mode: PoweredGloveMode,
    pub cycles: u32,
    pub dt_s: f64,
    pub requested_contact_force_n: [f64; NUM_GLOVE_DIGITS],
    pub tendon_speed_m_s: [f64; NUM_GLOVE_DIGITS],
    pub exercise_resistance_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl GloveBenchmarkProtocol {
    pub fn sustained_tool_grip(mode: PoweredGloveMode) -> Self {
        Self {
            protocol_id: "sx018-sim-tool-grip-v1".to_string(),
            mode,
            cycles: 300,
            dt_s: 1.0,
            requested_contact_force_n: [20.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [0.01; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn fine_manipulation(mode: PoweredGloveMode) -> Self {
        Self {
            protocol_id: "sx018-sim-fine-manip-v1".to_string(),
            mode,
            cycles: 180,
            dt_s: 0.5,
            requested_contact_force_n: [6.0, 5.0, 3.0, 2.0, 2.0],
            tendon_speed_m_s: [0.015; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.protocol_id.trim().is_empty()
            && self.cycles > 0
            && self.dt_s.is_finite()
            && self.dt_s > 0.0
            && self
                .requested_contact_force_n
                .iter()
                .chain(self.tendon_speed_m_s.iter())
                .all(|v| v.is_finite() && *v >= 0.0)
            && self.exercise_resistance_fraction.is_finite()
            && (0.0..=1.0).contains(&self.exercise_resistance_fraction)
    }

    fn command(&self) -> PoweredGloveCommand {
        PoweredGloveCommand {
            mode: self.mode,
            requested_contact_force_n: self.requested_contact_force_n,
            tendon_speed_m_s: self.tendon_speed_m_s,
            exercise_resistance_fraction: self.exercise_resistance_fraction,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GloveBenchmarkResult {
    pub protocol_id: String,
    pub mode: PoweredGloveMode,
    pub elapsed_s: f64,
    /// Integral of total wearer force over time, N*s.
    pub human_force_impulse_n_s: f64,
    pub peak_total_human_force_n: f64,
    pub electrical_energy_wh: f64,
    pub ending_mean_fatigue: f64,
    pub minimum_tactile_fidelity: f64,
    pub degraded_steps: u32,
    pub passive_release_probe_passed: bool,
    pub evidence: ExosuitEvidenceLevel,
}

pub fn run_glove_benchmark(
    glove: &mut PoweredGloveTwin,
    protocol: &GloveBenchmarkProtocol,
) -> Result<GloveBenchmarkResult, PoweredGloveFault> {
    if !protocol.is_valid() {
        return Err(PoweredGloveFault::InvalidCommand);
    }

    let command = protocol.command();
    let mut human_force_impulse_n_s = 0.0;
    let mut peak_total_human_force_n: f64 = 0.0;
    let mut electrical_energy_wh = 0.0;
    let mut minimum_tactile_fidelity: f64 = 1.0;
    let mut degraded_steps = 0;

    for _ in 0..protocol.cycles {
        let step = glove.step(command, protocol.dt_s)?;
        let total_human_force = step.human_required_force_n.iter().sum::<f64>();
        human_force_impulse_n_s += total_human_force * protocol.dt_s;
        peak_total_human_force_n = peak_total_human_force_n.max(total_human_force);
        electrical_energy_wh += step.electrical_power_w * protocol.dt_s / 3600.0;
        if let Some(minimum) = step.tactile_fidelity.iter().copied().reduce(f64::min) {
            minimum_tactile_fidelity = minimum_tactile_fidelity.min(minimum);
        }
        if step.fail_state != PoweredGloveFailState::Nominal {
            degraded_steps += 1;
        }
    }

    let ending_mean_fatigue =
        glove.state().fatigue.iter().sum::<f64>() / NUM_GLOVE_DIGITS as f64;

    // Mandatory fail-passive probe. This mutates only the benchmark twin.
    glove.state_mut_for_fault_injection().power_available = false;
    let release_probe = glove.step(
        PoweredGloveCommand {
            mode: PoweredGloveMode::HoldAssist,
            requested_contact_force_n: [20.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [0.01; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        },
        0.1,
    )?;
    let passive_release_probe_passed = release_probe.backdrivable
        && release_probe.fail_state == PoweredGloveFailState::PassiveBackdrivable
        && release_probe.assist_force_n.iter().all(|v| *v == 0.0)
        && release_probe
            .exercise_resistance_force_n
            .iter()
            .all(|v| *v == 0.0)
        && release_probe.electrical_power_w == 0.0;

    Ok(GloveBenchmarkResult {
        protocol_id: protocol.protocol_id.clone(),
        mode: protocol.mode,
        elapsed_s: protocol.cycles as f64 * protocol.dt_s,
        human_force_impulse_n_s,
        peak_total_human_force_n,
        electrical_energy_wh,
        ending_mean_fatigue,
        minimum_tactile_fidelity,
        degraded_steps,
        passive_release_probe_passed,
        evidence: protocol.evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grip_assist_trades_energy_for_lower_human_workload_and_fatigue() {
        let mut transparent = PoweredGloveTwin::simulation_reference();
        let mut assisted = PoweredGloveTwin::simulation_reference();
        let baseline = run_glove_benchmark(
            &mut transparent,
            &GloveBenchmarkProtocol::sustained_tool_grip(PoweredGloveMode::Transparent),
        )
        .unwrap();
        let candidate = run_glove_benchmark(
            &mut assisted,
            &GloveBenchmarkProtocol::sustained_tool_grip(PoweredGloveMode::GripAssist),
        )
        .unwrap();

        assert!(candidate.human_force_impulse_n_s < baseline.human_force_impulse_n_s);
        assert!(candidate.ending_mean_fatigue < baseline.ending_mean_fatigue);
        assert!(candidate.electrical_energy_wh > baseline.electrical_energy_wh);
        assert!(baseline.passive_release_probe_passed);
        assert!(candidate.passive_release_probe_passed);
    }

    #[test]
    fn fine_mode_preserves_more_tactile_proxy_than_hold_mode() {
        let mut fine = PoweredGloveTwin::simulation_reference();
        let mut hold = PoweredGloveTwin::simulation_reference();
        let a = run_glove_benchmark(
            &mut fine,
            &GloveBenchmarkProtocol::fine_manipulation(PoweredGloveMode::FineManipulation),
        )
        .unwrap();
        let b = run_glove_benchmark(
            &mut hold,
            &GloveBenchmarkProtocol::fine_manipulation(PoweredGloveMode::HoldAssist),
        )
        .unwrap();
        assert!(a.minimum_tactile_fidelity > b.minimum_tactile_fidelity);
    }

    #[test]
    fn malformed_protocol_fails_closed() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let mut protocol = GloveBenchmarkProtocol::sustained_tool_grip(PoweredGloveMode::GripAssist);
        protocol.dt_s = f64::NAN;
        assert_eq!(
            run_glove_benchmark(&mut glove, &protocol),
            Err(PoweredGloveFault::InvalidCommand)
        );
    }
}
