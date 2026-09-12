// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-production tuning for the pure-SU(3) random-walk Metropolis proposal.
//!
//! Adaptation is deliberately confined to a dedicated warm-up phase and a
//! dedicated RNG stream. Retained production measurements must use a frozen
//! proposal width and a separate transition stream.

use crate::lattice_gauge::WilsonGaugeField;
use crate::lattice_rng::{
    LatticeChaCha8Stream, LatticeRngError, LatticeStreamCoordinates, LatticeStreamDomain,
};
use crate::lattice_sweep::{LatticeSweepError, SymmetricProposalConfig, metropolis_sweep};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProposalTuningConfig {
    pub initial_angle: f64,
    pub min_angle: f64,
    pub max_angle: f64,
    /// Caller-declared target. This module intentionally defines no universal optimum.
    pub target_acceptance: f64,
    /// Initial gain for log-angle adaptation; step size decays as gain/sqrt(t).
    pub gain: f64,
    pub tuning_sweeps: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProposalTuningStep {
    pub sweep: u64,
    pub angle_used: f64,
    pub acceptance_rate: f64,
    pub mean_acceptance_probability: f64,
    /// Actual additive change applied in log-angle space after clamping to bounds.
    pub log_angle_adjustment: f64,
    pub next_angle: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalTuningResult {
    pub tuning_stream_id: u64,
    pub target_acceptance: f64,
    pub initial_angle: f64,
    pub final_angle: f64,
    pub history: Vec<ProposalTuningStep>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProposalTuningError {
    InvalidInitialAngle(f64),
    InvalidBounds { min_angle: f64, max_angle: f64 },
    InvalidTargetAcceptance(f64),
    InvalidGain(f64),
    InvalidTuningSweeps(u64),
    TuningSweepCountTooLarge(u64),
    Rng(LatticeRngError),
    Sweep(LatticeSweepError),
}

impl From<LatticeRngError> for ProposalTuningError {
    fn from(value: LatticeRngError) -> Self { Self::Rng(value) }
}
impl From<LatticeSweepError> for ProposalTuningError {
    fn from(value: LatticeSweepError) -> Self { Self::Sweep(value) }
}

impl ProposalTuningConfig {
    pub fn validate(self) -> Result<(), ProposalTuningError> {
        if !self.min_angle.is_finite()
            || !self.max_angle.is_finite()
            || self.min_angle <= 0.0
            || self.max_angle > PI
            || self.min_angle >= self.max_angle
        {
            return Err(ProposalTuningError::InvalidBounds {
                min_angle: self.min_angle,
                max_angle: self.max_angle,
            });
        }
        if !self.initial_angle.is_finite()
            || self.initial_angle < self.min_angle
            || self.initial_angle > self.max_angle
        {
            return Err(ProposalTuningError::InvalidInitialAngle(self.initial_angle));
        }
        if !self.target_acceptance.is_finite()
            || self.target_acceptance <= 0.0
            || self.target_acceptance >= 1.0
        {
            return Err(ProposalTuningError::InvalidTargetAcceptance(
                self.target_acceptance,
            ));
        }
        if !self.gain.is_finite() || self.gain <= 0.0 {
            return Err(ProposalTuningError::InvalidGain(self.gain));
        }
        if self.tuning_sweeps == 0 {
            return Err(ProposalTuningError::InvalidTuningSweeps(0));
        }
        usize::try_from(self.tuning_sweeps)
            .map_err(|_| ProposalTuningError::TuningSweepCountTooLarge(self.tuning_sweeps))?;
        Ok(())
    }
}

fn adapt_angle(
    current_angle: f64,
    observed_acceptance: f64,
    config: ProposalTuningConfig,
    sweep_index: u64,
) -> (f64, f64) {
    let current_log = current_angle.ln();
    let step_size = config.gain / ((sweep_index + 1) as f64).sqrt();
    let requested_adjustment = step_size * (observed_acceptance - config.target_acceptance);
    let log_min = config.min_angle.ln();
    let log_max = config.max_angle.ln();
    let next_log = (current_log + requested_adjustment).clamp(log_min, log_max);
    (next_log.exp(), next_log - current_log)
}

/// Adapt proposal width during a dedicated warm-up phase.
///
/// The field is mutated by the tuning sweeps. The returned `final_angle` is the
/// value to freeze for a later production chain. A production workflow should
/// then switch to a separately domain-separated `GaugeTransition` RNG stream
/// and perform its predeclared production burn-in before retaining measurements.
/// No sample generated here has production-evidence authority.
pub fn tune_chacha8_metropolis_proposal(
    field: &mut WilsonGaugeField,
    beta: f64,
    config: ProposalTuningConfig,
    seed: [u8; 32],
    ensemble_slot: u32,
    replica: u16,
    rank: u16,
) -> Result<ProposalTuningResult, ProposalTuningError> {
    config.validate()?;
    let history_capacity = usize::try_from(config.tuning_sweeps)
        .map_err(|_| ProposalTuningError::TuningSweepCountTooLarge(config.tuning_sweeps))?;
    let coordinates = LatticeStreamCoordinates {
        domain: LatticeStreamDomain::GaugeTuning,
        ensemble_slot,
        replica,
        rank,
    };
    let mut source = LatticeChaCha8Stream::new(seed, coordinates)?;
    let tuning_stream_id = source.stream_id();
    let mut angle = config.initial_angle;
    let mut history = Vec::with_capacity(history_capacity);

    for sweep_index in 0..config.tuning_sweeps {
        let stats = metropolis_sweep(
            field,
            beta,
            SymmetricProposalConfig { max_angle: angle },
            &mut source,
        )?;
        let acceptance_rate = stats.acceptance_rate();
        let (next_angle, adjustment) = adapt_angle(angle, acceptance_rate, config, sweep_index);
        history.push(ProposalTuningStep {
            sweep: sweep_index + 1,
            angle_used: angle,
            acceptance_rate,
            mean_acceptance_probability: stats.mean_acceptance_probability,
            log_angle_adjustment: adjustment,
            next_angle,
        });
        angle = next_angle;
    }

    Ok(ProposalTuningResult {
        tuning_stream_id,
        target_acceptance: config.target_acceptance,
        initial_angle: config.initial_angle,
        final_angle: angle,
        history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ProposalTuningConfig {
        ProposalTuningConfig {
            initial_angle: 0.2,
            min_angle: 0.02,
            max_angle: 1.0,
            target_acceptance: 0.7,
            gain: 0.5,
            tuning_sweeps: 4,
        }
    }

    #[test]
    fn adaptation_direction_tracks_acceptance_error() {
        let cfg = config();
        let (larger, positive) = adapt_angle(0.2, 0.9, cfg, 0);
        let (smaller, negative) = adapt_angle(0.2, 0.5, cfg, 0);
        assert!(larger > 0.2 && positive > 0.0);
        assert!(smaller < 0.2 && negative < 0.0);
    }

    #[test]
    fn adaptation_respects_declared_bounds_and_records_applied_step() {
        let cfg = ProposalTuningConfig {
            min_angle: 0.1,
            max_angle: 0.3,
            gain: 100.0,
            ..config()
        };
        let (upper, upper_applied) = adapt_angle(0.2, 0.99, cfg, 0);
        let (lower, lower_applied) = adapt_angle(0.2, 0.01, cfg, 0);
        assert!((upper - 0.3).abs() < 1e-12);
        assert!((lower - 0.1).abs() < 1e-12);
        assert!((upper_applied - (0.3_f64 / 0.2).ln()).abs() < 1e-12);
        assert!((lower_applied - (0.1_f64 / 0.2).ln()).abs() < 1e-12);
    }

    #[test]
    fn tuning_is_deterministically_replayable_and_uses_separate_domain() {
        let mut a = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        let mut b = WilsonGaugeField::identity([2, 2, 1, 2]).unwrap();
        let seed = [0x77; 32];
        let result_a = tune_chacha8_metropolis_proposal(
            &mut a, 5.7, config(), seed, 9, 1, 0,
        ).unwrap();
        let result_b = tune_chacha8_metropolis_proposal(
            &mut b, 5.7, config(), seed, 9, 1, 0,
        ).unwrap();
        assert_eq!(result_a, result_b);
        assert_eq!(result_a.history.len(), 4);
        assert_eq!(
            result_a.tuning_stream_id >> 56,
            LatticeStreamDomain::GaugeTuning as u64
        );
        assert!((a.average_plaquette().unwrap() - b.average_plaquette().unwrap()).abs() < 1e-14);
    }

    #[test]
    fn no_universal_target_is_assumed() {
        let mut low = config();
        low.target_acceptance = 0.4;
        let mut high = config();
        high.target_acceptance = 0.9;
        assert!(low.validate().is_ok());
        assert!(high.validate().is_ok());
    }
}
