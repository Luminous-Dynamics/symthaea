// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conservative trophic-transfer accounting.
//!
//! This is an energy or material ledger, not a food-web dynamics model. Each
//! declared transfer efficiency routes part of one level's available input to
//! production at the next level. Non-transferred material is split explicitly
//! between detrital routing and dissipative loss. The accounting closes exactly
//! up to floating-point roundoff and never treats a conventional ecological
//! efficiency as a universal constant.

use crate::error::{ModelError, require_fraction, require_non_negative};

pub const MAX_TROPHIC_TRANSFERS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrophicLevelLedger {
    pub transfer_index: usize,
    pub input: f64,
    pub transferred_production: f64,
    pub detrital_routing: f64,
    pub dissipative_loss: f64,
    pub level_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrophicTransferLedger {
    pub initial_input: f64,
    pub levels: Vec<TrophicLevelLedger>,
    pub top_level_production: f64,
    pub cumulative_detrital_routing: f64,
    pub cumulative_dissipative_loss: f64,
    pub total_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrophicTransferModel {
    transfer_efficiencies: Vec<f64>,
    detritus_fractions_of_loss: Vec<f64>,
}

impl TrophicTransferModel {
    pub fn try_new(
        transfer_efficiencies: Vec<f64>,
        detritus_fractions_of_loss: Vec<f64>,
    ) -> Result<Self, ModelError> {
        if transfer_efficiencies.is_empty() {
            return Err(ModelError::EmptySeries {
                series: "trophic transfer efficiencies",
            });
        }
        if transfer_efficiencies.len() > MAX_TROPHIC_TRANSFERS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: transfer_efficiencies.len(),
                maximum: MAX_TROPHIC_TRANSFERS,
            });
        }
        if detritus_fractions_of_loss.len() != transfer_efficiencies.len() {
            return Err(ModelError::DimensionMismatch {
                context: "trophic detritus fractions",
                expected: transfer_efficiencies.len(),
                found: detritus_fractions_of_loss.len(),
            });
        }
        let model = Self {
            transfer_efficiencies,
            detritus_fractions_of_loss,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn transfers(&self) -> usize {
        self.transfer_efficiencies.len()
    }

    pub fn transfer_efficiencies(&self) -> &[f64] {
        &self.transfer_efficiencies
    }

    pub fn detritus_fractions_of_loss(&self) -> &[f64] {
        &self.detritus_fractions_of_loss
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.transfer_efficiencies.is_empty() {
            return Err(ModelError::EmptySeries {
                series: "trophic transfer efficiencies",
            });
        }
        if self.transfer_efficiencies.len() > MAX_TROPHIC_TRANSFERS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: self.transfer_efficiencies.len(),
                maximum: MAX_TROPHIC_TRANSFERS,
            });
        }
        if self.detritus_fractions_of_loss.len() != self.transfer_efficiencies.len() {
            return Err(ModelError::DimensionMismatch {
                context: "trophic detritus fractions",
                expected: self.transfer_efficiencies.len(),
                found: self.detritus_fractions_of_loss.len(),
            });
        }
        for efficiency in &self.transfer_efficiencies {
            require_fraction("trophic_transfer_efficiency", *efficiency)?;
        }
        for fraction in &self.detritus_fractions_of_loss {
            require_fraction("detritus_fraction_of_loss", *fraction)?;
        }
        Ok(())
    }

    pub fn account(&self, initial_input: f64) -> Result<TrophicTransferLedger, ModelError> {
        self.validate()?;
        require_non_negative("initial_trophic_input", initial_input)?;
        let mut levels = Vec::with_capacity(self.transfers());
        let mut current = initial_input;
        let mut cumulative_detrital_routing = 0.0;
        let mut cumulative_dissipative_loss = 0.0;
        for (transfer_index, (&efficiency, &detritus_fraction)) in self
            .transfer_efficiencies
            .iter()
            .zip(&self.detritus_fractions_of_loss)
            .enumerate()
        {
            let transferred_production = current * efficiency;
            let non_transferred = current - transferred_production;
            let detrital_routing = non_transferred * detritus_fraction;
            let dissipative_loss = non_transferred - detrital_routing;
            let level_residual =
                current - transferred_production - detrital_routing - dissipative_loss;
            levels.push(TrophicLevelLedger {
                transfer_index,
                input: current,
                transferred_production,
                detrital_routing,
                dissipative_loss,
                level_residual,
            });
            cumulative_detrital_routing += detrital_routing;
            cumulative_dissipative_loss += dissipative_loss;
            current = transferred_production;
        }
        let total_residual =
            initial_input - current - cumulative_detrital_routing - cumulative_dissipative_loss;
        Ok(TrophicTransferLedger {
            initial_input,
            levels,
            top_level_production: current,
            cumulative_detrital_routing,
            cumulative_dissipative_loss,
            total_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_conserves_input_across_every_transfer() {
        let model =
            TrophicTransferModel::try_new(vec![0.2, 0.15, 0.1], vec![0.5, 0.4, 0.3]).unwrap();
        let ledger = model.account(1000.0).unwrap();
        assert!(ledger.total_residual.abs() < 1.0e-12);
        assert!(
            ledger
                .levels
                .iter()
                .all(|level| level.level_residual.abs() < 1.0e-12)
        );
        assert!((ledger.top_level_production - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn zero_efficiency_routes_all_input_to_declared_losses() {
        let model = TrophicTransferModel::try_new(vec![0.0], vec![0.25]).unwrap();
        let ledger = model.account(100.0).unwrap();
        assert_eq!(ledger.top_level_production, 0.0);
        assert!((ledger.cumulative_detrital_routing - 25.0).abs() < 1.0e-12);
        assert!((ledger.cumulative_dissipative_loss - 75.0).abs() < 1.0e-12);
    }

    #[test]
    fn perfect_transfer_has_no_loss() {
        let model = TrophicTransferModel::try_new(vec![1.0, 1.0], vec![0.5, 0.5]).unwrap();
        let ledger = model.account(42.0).unwrap();
        assert_eq!(ledger.top_level_production, 42.0);
        assert_eq!(ledger.cumulative_detrital_routing, 0.0);
        assert_eq!(ledger.cumulative_dissipative_loss, 0.0);
    }

    #[test]
    fn malformed_or_unphysical_efficiencies_fail_closed() {
        assert!(TrophicTransferModel::try_new(vec![], vec![]).is_err());
        assert!(TrophicTransferModel::try_new(vec![0.2], vec![]).is_err());
        assert!(TrophicTransferModel::try_new(vec![1.2], vec![0.5]).is_err());
    }
}
