// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Local linear recovery diagnostics for scalar ecological models.
//!
//! For `dX/dt = f(X)`, the derivative `f'(X*)` at an equilibrium determines
//! local stability. Stable equilibria have an e-folding recovery time
//! `-1/f'(X*)`; the time diverges as the derivative approaches zero at a fold
//! or transcritical threshold.

use crate::error::{ModelError, require_finite};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearStability {
    Stable,
    Critical,
    Unstable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecoveryDiagnostic {
    /// Local derivative `f'(X*)`, per unit time.
    pub derivative: f64,
    pub stability: LinearStability,
    /// Stable-equilibrium e-folding time in the model's time unit.
    pub e_folding_time: Option<f64>,
}

pub fn scalar_recovery_diagnostic(derivative: f64) -> Result<RecoveryDiagnostic, ModelError> {
    require_finite("equilibrium_derivative", derivative)?;
    let tolerance = 64.0 * f64::EPSILON * derivative.abs().max(1.0);
    let stability = if derivative.abs() <= tolerance {
        LinearStability::Critical
    } else if derivative < 0.0 {
        LinearStability::Stable
    } else {
        LinearStability::Unstable
    };
    Ok(RecoveryDiagnostic {
        derivative,
        stability,
        e_folding_time: (stability == LinearStability::Stable).then(|| -1.0 / derivative),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_classification_and_timescale_are_explicit() {
        let stable = scalar_recovery_diagnostic(-0.25).unwrap();
        assert_eq!(stable.stability, LinearStability::Stable);
        assert_eq!(stable.e_folding_time, Some(4.0));
        assert_eq!(
            scalar_recovery_diagnostic(0.0).unwrap().stability,
            LinearStability::Critical
        );
        assert_eq!(
            scalar_recovery_diagnostic(0.25).unwrap().stability,
            LinearStability::Unstable
        );
    }
}
